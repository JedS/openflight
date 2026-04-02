"""K-LD7 angle radar tracker with ring buffer for shot correlation."""

import logging
import threading
import time
from collections import deque
from typing import Optional

from .types import KLD7Angle, KLD7Frame

logger = logging.getLogger(__name__)


def _target_to_dict(target):
    """Convert a kld7 Target namedtuple to a dict."""
    if target is None:
        return None
    return {
        "distance": target.distance,
        "speed": target.speed,
        "angle": target.angle,
        "magnitude": target.magnitude,
    }


def _find_port():
    """Auto-detect K-LD7 EVAL board USB serial port."""
    try:
        from serial.tools.list_ports import comports
    except ImportError:
        return None
    for port in comports():
        desc = (port.description or "").lower()
        mfg = (port.manufacturer or "").lower()
        if any(kw in desc for kw in ["ftdi", "cp210", "usb-serial", "uart"]):
            return port.device
        if any(kw in mfg for kw in ["ftdi", "silicon labs"]):
            return port.device
    return None


class KLD7Tracker:
    """
    K-LD7 angle radar tracker.

    Streams TDAT+PDAT frames in a background thread into a ring buffer.
    When the OPS243 detects a shot, call get_angle_for_shot() to search
    the buffer for the ball pass and extract angle data.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        range_m: int = 5,
        speed_kmh: int = 100,
        orientation: str = "vertical",
        buffer_seconds: float = 2.0,
    ):
        self.port = port
        self.range_m = range_m
        self.speed_kmh = speed_kmh
        self.orientation = orientation
        self.buffer_seconds = buffer_seconds
        self.max_buffer_frames = int(34 * buffer_seconds)

        self._radar = None
        self._stream_thread: Optional[threading.Thread] = None
        self._running = False
        self._init_ring_buffer()

    def _init_ring_buffer(self):
        """Initialize or reset the ring buffer."""
        self._ring_buffer: deque[KLD7Frame] = deque(maxlen=self.max_buffer_frames)

    def connect(self) -> bool:
        """Connect to K-LD7 and configure for golf."""
        try:
            from kld7 import KLD7
        except ImportError:
            logger.error("kld7 package not installed. Run: pip install kld7")
            return False

        port = self.port or _find_port()
        if not port:
            logger.error("No K-LD7 EVAL board detected")
            return False

        try:
            self._radar = KLD7(port, baudrate=115200)
            logger.info("K-LD7 connected on %s", port)
        except Exception as e:
            logger.error("K-LD7 connection failed: %s", e)
            return False

        self._configure_for_golf()
        return True

    def _configure_for_golf(self):
        """Configure K-LD7 parameters for golf ball detection."""
        range_settings = {5: 0, 10: 1, 30: 2, 100: 3}
        speed_settings = {12: 0, 25: 1, 50: 2, 100: 3}

        params = self._radar.params
        params.RRAI = range_settings.get(self.range_m, 0)
        params.RSPI = speed_settings.get(self.speed_kmh, 3)
        params.DEDI = 2
        params.THOF = 10
        params.TRFT = 1
        params.MIAN = -90
        params.MAAN = 90
        params.MIRA = 0
        params.MARA = 100
        params.MISP = 0
        params.MASP = 100
        params.VISU = 0

        logger.info(
            "K-LD7 configured: range=%dm, speed=%dkm/h, orientation=%s",
            self.range_m, self.speed_kmh, self.orientation,
        )

    def start(self):
        """Start the background streaming thread."""
        if self._running:
            return
        self._running = True
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        logger.info("K-LD7 streaming started (orientation=%s)", self.orientation)

    def stop(self):
        """Stop streaming and close connection."""
        self._running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=5)
            self._stream_thread = None
        if self._radar:
            try:
                self._radar.close()
            except Exception:
                pass
            try:
                self._radar._port = None
            except Exception:
                pass
            self._radar = None
        logger.info("K-LD7 stopped")

    def _stream_loop(self):
        """Background thread: stream TDAT+PDAT into ring buffer."""
        from kld7 import FrameCode

        frame_codes = FrameCode.TDAT | FrameCode.PDAT
        current_frame = KLD7Frame(timestamp=time.time())
        seen_in_frame = set()

        try:
            for code, payload in self._radar.stream_frames(frame_codes, max_count=-1):
                if not self._running:
                    break

                if code in seen_in_frame:
                    self._add_frame(current_frame)
                    current_frame = KLD7Frame(timestamp=time.time())
                    seen_in_frame = set()

                seen_in_frame.add(code)

                if code == "TDAT":
                    current_frame.tdat = _target_to_dict(payload)
                elif code == "PDAT":
                    current_frame.pdat = [_target_to_dict(t) for t in payload] if payload else []

            if seen_in_frame:
                self._add_frame(current_frame)

        except Exception as e:
            if self._running:
                logger.error("K-LD7 stream error: %s", e)

    def _add_frame(self, frame: KLD7Frame):
        """Add a frame to the ring buffer."""
        self._ring_buffer.append(frame)

    # --- Ball detection thresholds ---
    # Ball appears as fast targets at far range (in flight / hitting net)
    BALL_MIN_SPEED_KMH = 8.0
    BALL_MIN_DISTANCE_M = 3.8
    BALL_MAX_DISTANCE_M = 5.5
    BALL_MAX_BURST_GAP_S = 0.1  # Max gap between frames in a burst

    # --- Club detection thresholds ---
    # Club detected by speed transition (slow→fast) at arm's length distance
    CLUB_MIN_DISTANCE_M = 0.8
    CLUB_MAX_DISTANCE_M = 2.5
    CLUB_SPEED_THRESHOLD_KMH = 10.0

    # --- General ---
    MIN_MAGNITUDE = 500
    MIN_CONFIDENCE = 0.3

    def _extract_ball(self, shot_timestamp=None):
        """Extract ball launch angle from ring buffer.

        Ball signature: fast targets (>8 km/h) at far distance (>3.8m)
        appearing as a 1-3 frame burst. Distance-based, not speed-based,
        because K-LD7 speed aliases above 100 km/h.
        """
        # Collect qualifying targets per frame
        ball_frames = []
        for frame in self._ring_buffer:
            targets = []
            for pt in frame.pdat or []:
                if (pt is not None
                        and abs(pt.get("speed", 0)) >= self.BALL_MIN_SPEED_KMH
                        and self.BALL_MIN_DISTANCE_M <= pt.get("distance", 0) <= self.BALL_MAX_DISTANCE_M
                        and pt.get("magnitude", 0) >= self.MIN_MAGNITUDE):
                    targets.append(pt)
            # Fall back to TDAT if no qualifying PDAT
            if not targets and frame.tdat:
                td = frame.tdat
                if (abs(td.get("speed", 0)) >= self.BALL_MIN_SPEED_KMH
                        and self.BALL_MIN_DISTANCE_M <= td.get("distance", 0) <= self.BALL_MAX_DISTANCE_M
                        and td.get("magnitude", 0) >= self.MIN_MAGNITUDE):
                    targets.append(td)
            if targets:
                ball_frames.append((frame.timestamp, targets))

        if not ball_frames:
            logger.debug("K-LD7 ball: no far/fast targets in %d buffer frames",
                          len(self._ring_buffer))
            return None

        # Group into bursts (consecutive frames within BALL_MAX_BURST_GAP_S)
        bursts = []
        current_burst = [ball_frames[0]]
        for i in range(1, len(ball_frames)):
            if ball_frames[i][0] - ball_frames[i - 1][0] <= self.BALL_MAX_BURST_GAP_S:
                current_burst.append(ball_frames[i])
            else:
                bursts.append(current_burst)
                current_burst = [ball_frames[i]]
        bursts.append(current_burst)

        # Pick the best burst — prefer closest to shot_timestamp, else highest magnitude
        if shot_timestamp is not None:
            def burst_score(burst):
                avg_time = sum(f[0] for f in burst) / len(burst)
                proximity = max(0.0, 1.0 - abs(avg_time - shot_timestamp) / 2.0)
                total_mag = sum(t.get("magnitude", 0) for _, targets in burst for t in targets)
                return proximity * total_mag
            best_burst = max(bursts, key=burst_score)
        else:
            def burst_mag(burst):
                return sum(t.get("magnitude", 0) for _, targets in burst for t in targets)
            best_burst = max(bursts, key=burst_mag)

        # Extract angle from best burst (magnitude-weighted)
        total_mag = 0
        weighted_angle = 0.0
        weighted_dist = 0.0
        max_magnitude = 0
        all_angles = []

        for _, targets in best_burst:
            for t in targets:
                mag = t.get("magnitude", 0)
                if mag > 0:
                    weighted_angle += t["angle"] * mag
                    weighted_dist += t["distance"] * mag
                    total_mag += mag
                    max_magnitude = max(max_magnitude, mag)
                    all_angles.append(t["angle"])

        if total_mag == 0:
            return None

        avg_angle = weighted_angle / total_mag
        avg_distance = weighted_dist / total_mag
        num_frames = len(best_burst)

        # Confidence: frame count + magnitude + angle consistency
        frame_score = min(num_frames / 3.0, 1.0)
        mag_score = min(max_magnitude / 3000.0, 1.0)
        if len(all_angles) > 1:
            mean_a = sum(all_angles) / len(all_angles)
            std_a = (sum((a - mean_a) ** 2 for a in all_angles) / len(all_angles)) ** 0.5
            consistency = max(0.0, 1.0 - std_a / 30.0)
        else:
            consistency = 0.5
        confidence = round(min(max(
            frame_score * 0.4 + mag_score * 0.3 + consistency * 0.3,
            0.0), 1.0), 2)

        if confidence < self.MIN_CONFIDENCE:
            logger.debug("K-LD7 ball: rejected — confidence %.2f < %.2f",
                          confidence, self.MIN_CONFIDENCE)
            return None

        logger.info("K-LD7 ball: angle=%.1f° dist=%.2fm mag=%d frames=%d conf=%.2f",
                     avg_angle, avg_distance, max_magnitude, num_frames, confidence)

        if self.orientation == "vertical":
            return KLD7Angle(
                vertical_deg=round(avg_angle, 1), horizontal_deg=None,
                distance_m=round(avg_distance, 2), magnitude=max_magnitude,
                confidence=confidence, num_frames=num_frames, detection_class="ball",
            )
        return KLD7Angle(
            vertical_deg=None, horizontal_deg=round(avg_angle, 1),
            distance_m=round(avg_distance, 2), magnitude=max_magnitude,
            confidence=confidence, num_frames=num_frames, detection_class="ball",
        )

    def _extract_club(self, shot_timestamp=None):
        """Extract club angle of attack from ring buffer.

        Club signature: speed transition from <10 to >=10 km/h at close
        range (1-2.5m). The fast PDAT targets at the transition frame
        are the club head approaching the ball.
        """
        frames_list = list(self._ring_buffer)
        best_transition = None
        best_score = -1

        for fi in range(1, len(frames_list)):
            frame = frames_list[fi]
            prev_frame = frames_list[fi - 1]

            # Get max speed at close range in current and previous frame
            def _close_range_max_speed(f):
                max_spd = 0
                for pt in f.pdat or []:
                    if pt and self.CLUB_MIN_DISTANCE_M <= pt.get("distance", 0) <= self.CLUB_MAX_DISTANCE_M:
                        max_spd = max(max_spd, abs(pt.get("speed", 0)))
                if f.tdat and self.CLUB_MIN_DISTANCE_M <= f.tdat.get("distance", 0) <= self.CLUB_MAX_DISTANCE_M:
                    max_spd = max(max_spd, abs(f.tdat.get("speed", 0)))
                return max_spd

            prev_speed = _close_range_max_speed(prev_frame)
            curr_speed = _close_range_max_speed(frame)

            if curr_speed >= self.CLUB_SPEED_THRESHOLD_KMH and prev_speed < self.CLUB_SPEED_THRESHOLD_KMH:
                # Speed transition found — collect fast close-range targets
                fast_targets = []
                for pt in frame.pdat or []:
                    if (pt and abs(pt.get("speed", 0)) >= self.CLUB_SPEED_THRESHOLD_KMH
                            and self.CLUB_MIN_DISTANCE_M <= pt.get("distance", 0) <= self.CLUB_MAX_DISTANCE_M
                            and pt.get("magnitude", 0) >= self.MIN_MAGNITUDE):
                        fast_targets.append(pt)

                if not fast_targets and frame.tdat:
                    td = frame.tdat
                    if (abs(td.get("speed", 0)) >= self.CLUB_SPEED_THRESHOLD_KMH
                            and self.CLUB_MIN_DISTANCE_M <= td.get("distance", 0) <= self.CLUB_MAX_DISTANCE_M
                            and td.get("magnitude", 0) >= self.MIN_MAGNITUDE):
                        fast_targets.append(td)

                if not fast_targets:
                    continue

                # Score by proximity to shot_timestamp (if provided) + magnitude
                total_mag = sum(t.get("magnitude", 0) for t in fast_targets)
                if shot_timestamp is not None:
                    proximity = max(0.0, 1.0 - abs(frame.timestamp - shot_timestamp) / 2.0)
                    score = proximity * total_mag
                else:
                    score = total_mag

                if score > best_score:
                    best_score = score
                    best_transition = (frame, fast_targets)

        if best_transition is None:
            logger.debug("K-LD7 club: no speed transition found in %d buffer frames",
                          len(self._ring_buffer))
            return None

        frame, fast_targets = best_transition

        # Magnitude-weighted angle
        total_mag = sum(t.get("magnitude", 0) for t in fast_targets if t.get("magnitude", 0) > 0)
        if total_mag == 0:
            return None

        avg_angle = sum(t["angle"] * t["magnitude"] for t in fast_targets if t["magnitude"] > 0) / total_mag
        avg_dist = sum(t["distance"] for t in fast_targets) / len(fast_targets)
        max_magnitude = max(t.get("magnitude", 0) for t in fast_targets)

        mag_score = min(max_magnitude / 4000.0, 1.0)
        n_targets = len(fast_targets)
        target_score = min(n_targets / 3.0, 1.0)
        confidence = round(min(max(mag_score * 0.5 + target_score * 0.5, 0.0), 1.0), 2)

        if confidence < self.MIN_CONFIDENCE:
            logger.debug("K-LD7 club: rejected — confidence %.2f < %.2f",
                          confidence, self.MIN_CONFIDENCE)
            return None

        logger.info("K-LD7 club: angle=%.1f° dist=%.2fm mag=%d targets=%d conf=%.2f",
                     avg_angle, avg_dist, max_magnitude, n_targets, confidence)

        if self.orientation == "vertical":
            return KLD7Angle(
                vertical_deg=round(avg_angle, 1), horizontal_deg=None,
                distance_m=round(avg_dist, 2), magnitude=max_magnitude,
                confidence=confidence, num_frames=1, detection_class="club",
            )
        return KLD7Angle(
            vertical_deg=None, horizontal_deg=round(avg_angle, 1),
            distance_m=round(avg_dist, 2), magnitude=max_magnitude,
            confidence=confidence, num_frames=1, detection_class="club",
        )

    def get_angle_for_shot(self, shot_timestamp: Optional[float] = None) -> Optional[KLD7Angle]:
        """Search the ring buffer for the ball launch angle.

        Uses distance-based detection: ball = fast targets at >3.8m.
        """
        return self._extract_ball(shot_timestamp)

    def get_club_angle(self, shot_timestamp: Optional[float] = None) -> Optional[KLD7Angle]:
        """Search the ring buffer for the club angle of attack.

        Uses speed-transition detection at close range (1-2.5m).
        """
        return self._extract_club(shot_timestamp)

    def snapshot_buffer(self) -> list:
        """Return a serializable snapshot of the current ring buffer.

        Call this BEFORE get_angle_for_shot/reset to capture raw data
        for offline analysis alongside OPS243 shot data.
        """
        frames = []
        for frame in self._ring_buffer:
            frames.append({
                "timestamp": frame.timestamp,
                "tdat": frame.tdat,
                "pdat": frame.pdat,
            })
        return frames

    def reset(self):
        """Clear the ring buffer after a shot is processed."""
        self._ring_buffer.clear()
