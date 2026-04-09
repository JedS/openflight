"""K-LD7 angle radar tracker with ring buffer for shot correlation."""

import logging
import math
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
        angle_offset_deg: float = 0.0,
    ):
        self.port = port
        self.range_m = range_m
        self.speed_kmh = speed_kmh
        self.orientation = orientation
        self.buffer_seconds = buffer_seconds
        self.angle_offset_deg = angle_offset_deg
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
            logger.error("[KLD7] kld7 package not installed. Run: pip install kld7")
            return False

        port = self.port or _find_port()
        if not port:
            logger.error("[KLD7] No K-LD7 EVAL board detected")
            return False

        # Flush the serial port before the kld7 library handshake.
        # A prior crash can leave stale data in the K-LD7 UART buffer,
        # causing "Wrong length reply" on the next connect.
        try:
            import serial
            with serial.Serial(port, 3000000, timeout=0.1) as ser:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                # Send GBYE to reset the K-LD7 to idle state, then INIT
                ser.write(b"GBYE\r\n")
                time.sleep(0.2)
                ser.reset_input_buffer()
            logger.debug("[KLD7] Serial port flushed on %s", port)
        except Exception as e:
            logger.debug("[KLD7] Pre-flush failed (not critical): %s", e)

        try:
            self._radar = KLD7(port, baudrate=3000000)
            actual_baud = getattr(self._radar._port, 'baudrate', 'unknown') if hasattr(self._radar, '_port') else 'unknown'
            logger.info("[KLD7] Connected on %s at %s baud", port, actual_baud)
        except Exception as e:
            logger.error("[KLD7] Connection failed: %s", e, exc_info=True)
            return False

        self._configure_for_golf()
        logger.info("[KLD7] Ready: port=%s, baud=%s, range=%dm, speed=%dkm/h, orientation=%s",
                     port, actual_baud, self.range_m, self.speed_kmh, self.orientation)
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
            "[KLD7] Configured: range=%dm, speed=%dkm/h, orientation=%s",
            self.range_m, self.speed_kmh, self.orientation,
        )

    def start(self):
        """Start the background streaming thread."""
        if self._running:
            return
        self._running = True
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        logger.info("[KLD7] Streaming started (orientation=%s)", self.orientation)

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
        logger.info("[KLD7] Stopped")

    def _stream_loop(self):
        """Background thread: stream RADC+TDAT+PDAT into ring buffer."""
        from kld7 import FrameCode

        frame_codes = FrameCode.RADC | FrameCode.TDAT | FrameCode.PDAT
        current_frame = KLD7Frame(timestamp=time.time())
        seen_in_frame = set()
        frame_count = 0
        radc_count = 0

        logger.info("[KLD7] Stream started: requesting RADC+TDAT+PDAT")

        try:
            for code, payload in self._radar.stream_frames(frame_codes, max_count=-1):
                if not self._running:
                    break

                if frame_count == 0:
                    logger.info("[KLD7] First frame received: %s", code)

                if code in seen_in_frame:
                    self._add_frame(current_frame)
                    frame_count += 1
                    if frame_count == 50:
                        logger.info(
                            "[KLD7] Stream health: %d frames, %d RADC (%s)",
                            frame_count, radc_count,
                            "RADC active" if radc_count > 0 else "NO RADC",
                        )
                    current_frame = KLD7Frame(timestamp=time.time())
                    seen_in_frame = set()

                seen_in_frame.add(code)

                if code == "RADC":
                    current_frame.radc = payload
                    radc_count += 1
                    if radc_count == 1:
                        logger.info("[KLD7] First RADC frame received (%d bytes)", len(payload) if payload else 0)
                elif code == "TDAT":
                    current_frame.tdat = _target_to_dict(payload)
                elif code == "PDAT":
                    current_frame.pdat = [_target_to_dict(t) for t in payload] if payload else []

            if seen_in_frame:
                self._add_frame(current_frame)

            logger.warning("[KLD7] Stream ended (frames=%d, radc=%d, running=%s)",
                          frame_count, radc_count, self._running)

        except Exception as e:
            logger.error("[KLD7] Stream crashed after %d frames: %s", frame_count, e, exc_info=True)

    def _add_frame(self, frame: KLD7Frame):
        """Add a frame to the ring buffer."""
        self._ring_buffer.append(frame)

    # --- Ball detection thresholds ---
    # Ball appears as fast targets at far range (in flight / hitting net)
    BALL_MIN_SPEED_KMH = 8.0
    BALL_MIN_DISTANCE_M = 3.8
    BALL_MAX_DISTANCE_M = 5.5
    BALL_MAX_BURST_GAP_S = 0.1  # Max gap between frames in a burst
    BALL_AFTER_CLUB_MIN_S = 0.08
    BALL_AFTER_CLUB_MAX_S = 0.35
    MIN_SHOT_GAP_S = 3.0  # Minimum time between probable shots (suppresses double-counts)

    # --- Club detection thresholds ---
    # Club detected by speed transition (slow→fast) at arm's length distance
    CLUB_MIN_DISTANCE_M = 0.8
    CLUB_MAX_DISTANCE_M = 2.5
    CLUB_SPEED_THRESHOLD_KMH = 10.0

    # --- General ---
    MIN_MAGNITUDE = 500
    MIN_CONFIDENCE = 0.3

    def _qualifying_ball_targets(self, frame: KLD7Frame) -> list[dict]:
        """Return far-range, fast targets that match the expected ball signature."""
        targets = []
        for pt in frame.pdat or []:
            if (
                pt is not None
                and abs(pt.get("speed", 0)) >= self.BALL_MIN_SPEED_KMH
                and self.BALL_MIN_DISTANCE_M <= pt.get("distance", 0) <= self.BALL_MAX_DISTANCE_M
                and pt.get("magnitude", 0) >= self.MIN_MAGNITUDE
            ):
                targets.append(pt)

        if not targets and frame.tdat:
            td = frame.tdat
            if (
                abs(td.get("speed", 0)) >= self.BALL_MIN_SPEED_KMH
                and self.BALL_MIN_DISTANCE_M <= td.get("distance", 0) <= self.BALL_MAX_DISTANCE_M
                and td.get("magnitude", 0) >= self.MIN_MAGNITUDE
            ):
                targets.append(td)

        return targets

    def _coherent_ball_track(self, burst: list[tuple[float, list[dict]]]) -> list[dict]:
        """Pick one coherent far-target path from a multi-target burst.

        PDAT often contains several qualifying far-range detections in the same
        frame. Averaging them all together can smear the launch angle badly, so
        we score target-to-target continuity across the burst and keep the best
        single path.
        """
        per_frame_targets = []
        for _, targets in burst:
            filtered = [t for t in targets if t.get("magnitude", 0) > 0]
            if filtered:
                per_frame_targets.append(filtered)

        if not per_frame_targets:
            return []

        paths = []
        prev_scores = []
        for frame_targets in per_frame_targets:
            current_scores = []
            for target_idx, target in enumerate(frame_targets):
                best_score = math.log1p(target.get("magnitude", 0))
                best_prev_idx = None
                for prev_idx, (prev_score, _) in enumerate(prev_scores):
                    prev_target = per_frame_targets[len(paths) - 1][prev_idx]
                    angle_delta = abs(target["angle"] - prev_target["angle"])
                    dist_delta = abs(target["distance"] - prev_target["distance"])
                    penalty = angle_delta * 0.12 + dist_delta * 1.2
                    score = prev_score + math.log1p(target.get("magnitude", 0)) - penalty
                    if score > best_score:
                        best_score = score
                        best_prev_idx = prev_idx
                current_scores.append((best_score, best_prev_idx))
            paths.append(current_scores)
            prev_scores = current_scores

        # Select the best endpoint from the *last* frame only, so backtracking
        # produces a full-length path through every frame in the burst.
        last_scores = paths[-1]
        best_target_idx = 0
        best_score = float("-inf")
        for target_idx, (score, _) in enumerate(last_scores):
            if score > best_score:
                best_score = score
                best_target_idx = target_idx

        track = []
        frame_idx = len(paths) - 1
        target_idx = best_target_idx
        while frame_idx >= 0 and target_idx is not None:
            track.append(per_frame_targets[frame_idx][target_idx])
            _, target_idx = paths[frame_idx][target_idx]
            frame_idx -= 1

        track.reverse()
        return track

    def _summarize_ball_burst(self, burst: list[tuple[float, list[dict]]]) -> Optional[dict]:
        """Summarize one burst using the best coherent far-target track."""
        best_track = self._coherent_ball_track(burst)
        if not best_track:
            return None

        total_mag = sum(target.get("magnitude", 0) for target in best_track if target.get("magnitude", 0) > 0)
        if total_mag == 0:
            return None

        avg_angle = sum(target["angle"] * target["magnitude"] for target in best_track) / total_mag
        avg_distance = sum(target["distance"] * target["magnitude"] for target in best_track) / total_mag
        max_magnitude = max(target.get("magnitude", 0) for target in best_track)
        all_angles = [target["angle"] for target in best_track]
        num_frames = len(best_track)

        frame_score = min(num_frames / 3.0, 1.0)
        mag_score = min(max_magnitude / 3000.0, 1.0)
        if len(all_angles) > 1:
            mean_a = sum(all_angles) / len(all_angles)
            std_a = (sum((a - mean_a) ** 2 for a in all_angles) / len(all_angles)) ** 0.5
            consistency = max(0.0, 1.0 - std_a / 30.0)
        else:
            std_a = 0.0
            consistency = 0.5
        confidence = round(
            min(max(frame_score * 0.4 + mag_score * 0.3 + consistency * 0.3, 0.0), 1.0), 2
        )

        return {
            "burst": burst,
            "track": best_track,
            "start_time": burst[0][0],
            "end_time": burst[-1][0],
            "angle": avg_angle,
            "distance": avg_distance,
            "magnitude": max_magnitude,
            "total_magnitude": total_mag,
            "num_frames": num_frames,
            "angle_std": std_a,
            "confidence": confidence,
        }

    def _collect_ball_bursts(self) -> list[dict]:
        """Collect and summarize all plausible far-range ball bursts in the buffer."""
        ball_frames = []
        for frame in self._ring_buffer:
            targets = self._qualifying_ball_targets(frame)
            if targets:
                ball_frames.append((frame.timestamp, targets))

        if not ball_frames:
            return []

        bursts = []
        current_burst = [ball_frames[0]]
        for i in range(1, len(ball_frames)):
            if ball_frames[i][0] - ball_frames[i - 1][0] <= self.BALL_MAX_BURST_GAP_S:
                current_burst.append(ball_frames[i])
            else:
                bursts.append(current_burst)
                current_burst = [ball_frames[i]]
        bursts.append(current_burst)

        summaries = []
        for burst in bursts:
            summary = self._summarize_ball_burst(burst)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def _collect_club_candidates(self, shot_timestamp=None) -> list[dict]:
        """Collect all plausible close-range club transition events in the buffer."""
        frames_list = list(self._ring_buffer)
        candidates = []

        for fi in range(1, len(frames_list)):
            frame = frames_list[fi]
            prev_frame = frames_list[fi - 1]

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
                fast_targets = []
                for pt in frame.pdat or []:
                    if (
                        pt
                        and abs(pt.get("speed", 0)) >= self.CLUB_SPEED_THRESHOLD_KMH
                        and self.CLUB_MIN_DISTANCE_M <= pt.get("distance", 0) <= self.CLUB_MAX_DISTANCE_M
                        and pt.get("magnitude", 0) >= self.MIN_MAGNITUDE
                    ):
                        fast_targets.append(pt)

                if not fast_targets and frame.tdat:
                    td = frame.tdat
                    if (
                        abs(td.get("speed", 0)) >= self.CLUB_SPEED_THRESHOLD_KMH
                        and self.CLUB_MIN_DISTANCE_M <= td.get("distance", 0) <= self.CLUB_MAX_DISTANCE_M
                        and td.get("magnitude", 0) >= self.MIN_MAGNITUDE
                    ):
                        fast_targets.append(td)

                if not fast_targets:
                    continue

                total_mag = sum(t.get("magnitude", 0) for t in fast_targets if t.get("magnitude", 0) > 0)
                if total_mag == 0:
                    continue

                avg_angle = sum(t["angle"] * t["magnitude"] for t in fast_targets if t["magnitude"] > 0) / total_mag
                avg_dist = sum(t["distance"] for t in fast_targets) / len(fast_targets)
                max_magnitude = max(t.get("magnitude", 0) for t in fast_targets)

                mag_score = min(max_magnitude / 4000.0, 1.0)
                target_score = min(len(fast_targets) / 3.0, 1.0)
                confidence = round(min(max(mag_score * 0.5 + target_score * 0.5, 0.0), 1.0), 2)

                if shot_timestamp is not None:
                    proximity = max(0.0, 1.0 - abs(frame.timestamp - shot_timestamp) / 2.0)
                    score = proximity * total_mag
                else:
                    score = total_mag

                candidates.append({
                    "frame": frame,
                    "targets": fast_targets,
                    "timestamp": frame.timestamp,
                    "angle": avg_angle,
                    "distance": avg_dist,
                    "magnitude": max_magnitude,
                    "num_targets": len(fast_targets),
                    "confidence": confidence,
                    "score": score,
                })

        return candidates

    def _extract_ball(self, shot_timestamp=None):
        """Extract ball launch angle from ring buffer.

        Ball signature: fast targets (>8 km/h) at far distance (>3.8m)
        appearing as a 1-3 frame burst. Distance-based, not speed-based,
        because K-LD7 speed aliases above 100 km/h.
        """
        ball_bursts = self._collect_ball_bursts()
        if not ball_bursts:
            logger.debug("[KLD7] Ball: no far/fast targets in %d buffer frames",
                          len(self._ring_buffer))
            return None

        # Pick the best burst — prefer closest to shot_timestamp, else highest magnitude
        if shot_timestamp is not None:
            def burst_score(burst):
                avg_time = (burst["start_time"] + burst["end_time"]) / 2.0
                proximity = max(0.0, 1.0 - abs(avg_time - shot_timestamp) / 2.0)
                return proximity * burst["total_magnitude"]
            best_burst = max(ball_bursts, key=burst_score)
        else:
            def burst_mag(burst):
                return burst["total_magnitude"]
            best_burst = max(ball_bursts, key=burst_mag)

        avg_angle = best_burst["angle"]
        avg_distance = best_burst["distance"]
        max_magnitude = best_burst["magnitude"]
        num_frames = best_burst["num_frames"]
        confidence = best_burst["confidence"]

        if confidence < self.MIN_CONFIDENCE:
            logger.debug("[KLD7] Ball: rejected — confidence %.2f < %.2f",
                          confidence, self.MIN_CONFIDENCE)
            return None

        logger.info("[KLD7] Ball: angle=%.1f° dist=%.2fm mag=%d frames=%d conf=%.2f",
                     avg_angle, avg_distance, max_magnitude, num_frames, confidence)

        corrected_angle = round(avg_angle + getattr(self, "angle_offset_deg", 0.0), 1)

        if self.orientation == "vertical":
            return KLD7Angle(
                vertical_deg=corrected_angle, horizontal_deg=None,
                distance_m=round(avg_distance, 2), magnitude=max_magnitude,
                confidence=confidence, num_frames=num_frames, detection_class="ball",
            )
        return KLD7Angle(
            vertical_deg=None, horizontal_deg=corrected_angle,
            distance_m=round(avg_distance, 2), magnitude=max_magnitude,
            confidence=confidence, num_frames=num_frames, detection_class="ball",
        )

    def _extract_club(self, shot_timestamp=None):
        """Extract club angle of attack from ring buffer.

        Club signature: speed transition from <10 to >=10 km/h at close
        range (1-2.5m). The fast PDAT targets at the transition frame
        are the club head approaching the ball.
        """
        club_candidates = self._collect_club_candidates(shot_timestamp=shot_timestamp)
        if not club_candidates:
            logger.debug("[KLD7] Club: no speed transition found in %d buffer frames",
                          len(self._ring_buffer))
            return None

        best_transition = max(club_candidates, key=lambda candidate: candidate["score"])
        avg_angle = best_transition["angle"]
        avg_dist = best_transition["distance"]
        max_magnitude = best_transition["magnitude"]
        n_targets = best_transition["num_targets"]
        confidence = best_transition["confidence"]

        if confidence < self.MIN_CONFIDENCE:
            logger.debug("[KLD7] Club: rejected — confidence %.2f < %.2f",
                          confidence, self.MIN_CONFIDENCE)
            return None

        logger.info("[KLD7] Club: angle=%.1f° dist=%.2fm mag=%d targets=%d conf=%.2f",
                     avg_angle, avg_dist, max_magnitude, n_targets, confidence)

        corrected_angle = round(avg_angle + getattr(self, "angle_offset_deg", 0.0), 1)

        if self.orientation == "vertical":
            return KLD7Angle(
                vertical_deg=corrected_angle, horizontal_deg=None,
                distance_m=round(avg_dist, 2), magnitude=max_magnitude,
                confidence=confidence, num_frames=1, detection_class="club",
            )
        return KLD7Angle(
            vertical_deg=None, horizontal_deg=corrected_angle,
            distance_m=round(avg_dist, 2), magnitude=max_magnitude,
            confidence=confidence, num_frames=1, detection_class="club",
        )

    def _extract_ball_radc(self, ball_speed_mph: float) -> Optional[KLD7Angle]:
        """Extract ball launch angle via RADC phase interferometry.

        Uses the OPS243-measured ball speed to narrow the FFT velocity
        search band, then extracts angle from F1A/F2A phase difference.
        """
        from .radc import extract_launch_angle

        frames = [
            {"timestamp": f.timestamp, "radc": f.radc}
            for f in self._ring_buffer
            if f.radc is not None
        ]

        if not frames:
            return None

        logger.info("[KLD7] RADC: examining %d frames with RADC data, ball_speed=%.1f mph",
                     len(frames), ball_speed_mph)

        results = extract_launch_angle(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            angle_offset_deg=self.angle_offset_deg,
            speed_tolerance_mph=10.0,
        )

        if not results:
            logger.debug("[KLD7] RADC: no ball detections for %.1f mph", ball_speed_mph)
            return None

        best = results[0]
        logger.info(
            "[KLD7] RADC: angle=%.1f° speed=%.1f mph snr=%.1f conf=%.2f frames=%d",
            best["launch_angle_deg"], best["ball_speed_mph"],
            best["avg_snr_db"], best["confidence"], best["frame_count"],
        )

        if self.orientation == "vertical":
            return KLD7Angle(
                vertical_deg=best["launch_angle_deg"],
                horizontal_deg=None,
                confidence=best["confidence"],
                num_frames=best["frame_count"],
                magnitude=best["avg_snr_db"],
                detection_class="ball",
            )
        return KLD7Angle(
            vertical_deg=None,
            horizontal_deg=best["launch_angle_deg"],
            confidence=best["confidence"],
            num_frames=best["frame_count"],
            magnitude=best["avg_snr_db"],
            detection_class="ball",
        )

    def get_angle_for_shot(self, shot_timestamp: Optional[float] = None, ball_speed_mph: Optional[float] = None) -> Optional[KLD7Angle]:
        """Search the ring buffer for the ball launch angle.

        Tries RADC phase interferometry first (if ball_speed_mph is provided),
        then falls back to PDAT distance-based detection.
        """
        logger.info("[KLD7] Angle extraction: ball_speed=%s mph, buffer=%d frames",
                     "%.1f" % ball_speed_mph if ball_speed_mph else "None", len(self._ring_buffer))

        if ball_speed_mph is not None:
            try:
                radc_result = self._extract_ball_radc(ball_speed_mph)
                if radc_result is not None:
                    return radc_result
                logger.info("[KLD7] RADC extraction returned None, falling back to PDAT")
            except Exception as e:
                logger.warning("[KLD7] RADC extraction failed, falling back to PDAT: %s", e, exc_info=True)

        result = self._extract_ball(shot_timestamp)
        if result:
            logger.info("[KLD7] PDAT angle: %.1f° (conf=%.2f, %d frames)",
                         result.vertical_deg or result.horizontal_deg, result.confidence, result.num_frames)
        else:
            logger.info("[KLD7] No angle detected (RADC and PDAT both failed)")
        return result

    def get_club_angle(self, shot_timestamp: Optional[float] = None) -> Optional[KLD7Angle]:
        """Search the ring buffer for the club angle of attack.

        Uses speed-transition detection at close range (1-2.5m).
        """
        return self._extract_club(shot_timestamp)

    def find_probable_shots(self) -> list[dict]:
        """Pair close-range club transitions with following far-range ball bursts.

        This is intended for offline capture analysis, where a long `.pkl`
        recording may contain many swings. It returns one entry per plausible
        club-to-ball sequence with timing and angle summaries.
        """
        club_candidates = self._collect_club_candidates()
        ball_bursts = self._collect_ball_bursts()
        probable_shots = []
        used_ball_indices = set()
        last_shot_time = float("-inf")

        for club in club_candidates:
            if club["timestamp"] - last_shot_time < self.MIN_SHOT_GAP_S:
                continue
            viable_balls = []
            for idx, burst in enumerate(ball_bursts):
                dt = burst["start_time"] - club["timestamp"]
                if (
                    idx not in used_ball_indices
                    and self.BALL_AFTER_CLUB_MIN_S <= dt <= self.BALL_AFTER_CLUB_MAX_S
                ):
                    viable_balls.append((idx, burst))

            if not viable_balls:
                continue

            best_idx, best_burst = max(
                viable_balls,
                key=lambda item: (
                    item[1]["confidence"],
                    item[1]["magnitude"],
                    item[1]["num_frames"],
                    item[1]["total_magnitude"],
                ),
            )
            used_ball_indices.add(best_idx)
            last_shot_time = club["timestamp"]

            probable_shots.append({
                "club_time": club["timestamp"],
                "ball_time": best_burst["start_time"],
                "dt_ms": round((best_burst["start_time"] - club["timestamp"]) * 1000.0, 1),
                "club_angle_deg": round(club["angle"], 1),
                "club_distance_m": round(club["distance"], 2),
                "club_magnitude": club["magnitude"],
                "ball_angle_deg": round(best_burst["angle"], 1),
                "ball_distance_m": round(best_burst["distance"], 2),
                "ball_magnitude": best_burst["magnitude"],
                "ball_confidence": best_burst["confidence"],
                "ball_frames": best_burst["num_frames"],
            })

        return probable_shots

    def snapshot_buffer(self) -> list:
        """Return a serializable snapshot of the current ring buffer.

        Call this BEFORE get_angle_for_shot/reset to capture raw data
        for offline analysis alongside OPS243 shot data.
        """
        frames = []
        for frame in self._ring_buffer:
            entry = {
                "timestamp": frame.timestamp,
                "tdat": frame.tdat,
                "pdat": frame.pdat,
            }
            if frame.radc is not None:
                entry["has_radc"] = True
            frames.append(entry)
        return frames

    def reset(self):
        """Clear the ring buffer after a shot is processed."""
        self._ring_buffer.clear()
