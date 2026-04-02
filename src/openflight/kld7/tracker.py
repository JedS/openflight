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

    # Minimum speed (km/h) to consider a detection as a potential ball/club event.
    # Body movement is typically 1-5 km/h; golf ball/club is >10 km/h.
    MIN_SPEED_KMH = 10.0

    # Maximum angle spread (degrees) within an event cluster before it's
    # rejected as noise. Ball passes have tight angle consistency.
    MAX_ANGLE_SPREAD_DEG = 60.0

    # Maximum event duration (seconds). Ball passes are transient (<0.3s).
    # Body movement lasts seconds.
    MAX_EVENT_DURATION_S = 1.0

    def get_angle_for_shot(self) -> Optional[KLD7Angle]:
        """
        Search the ring buffer for the ball pass and extract angle data.

        Applies signal processing filters to isolate ball/club events:
        1. Speed filter: reject detections below MIN_SPEED_KMH
        2. Event clustering: group detections within 0.5s window
        3. Angle spread filter: reject clusters with wide angle variation
        4. Duration filter: reject events lasting longer than MAX_EVENT_DURATION_S
        """
        detections = []

        for frame in self._ring_buffer:
            if frame.pdat:
                for target in frame.pdat:
                    if target is not None and target.get("magnitude", 0) > 0:
                        speed = abs(target.get("speed", 0))
                        if speed >= self.MIN_SPEED_KMH:
                            detections.append((
                                frame.timestamp,
                                target["angle"],
                                target["distance"],
                                target["magnitude"],
                            ))
            elif frame.tdat and frame.tdat.get("magnitude", 0) > 0:
                speed = abs(frame.tdat.get("speed", 0))
                if speed >= self.MIN_SPEED_KMH:
                    detections.append((
                        frame.timestamp,
                        frame.tdat["angle"],
                        frame.tdat["distance"],
                        frame.tdat["magnitude"],
                    ))

        if not detections:
            return None

        peak = max(detections, key=lambda d: d[3])
        peak_time = peak[0]

        event_detections = [
            d for d in detections if abs(d[0] - peak_time) < 0.5
        ]

        if not event_detections:
            return None

        # Duration filter: reject events lasting too long
        timestamps = [d[0] for d in event_detections]
        event_duration = max(timestamps) - min(timestamps)
        if event_duration > self.MAX_EVENT_DURATION_S:
            return None

        # Angle spread filter: reject events with wide angle variation
        angles = [d[1] for d in event_detections]
        angle_spread = max(angles) - min(angles)
        if angle_spread > self.MAX_ANGLE_SPREAD_DEG:
            return None

        total_mag = sum(d[3] for d in event_detections)
        if total_mag == 0:
            return None

        avg_angle = sum(d[1] * d[3] for d in event_detections) / total_mag
        avg_distance = sum(d[2] * d[3] for d in event_detections) / total_mag
        max_magnitude = max(d[3] for d in event_detections)
        num_frames = len(set(d[0] for d in event_detections))

        frame_score = min(num_frames / 3.0, 1.0)
        mag_score = min(max_magnitude / 5000.0, 1.0)

        if len(angles) > 1:
            mean_angle = sum(angles) / len(angles)
            angle_std = (sum((a - mean_angle) ** 2 for a in angles) / len(angles)) ** 0.5
            consistency_score = max(0.0, 1.0 - angle_std / 20.0)
        else:
            consistency_score = 0.5

        confidence = frame_score * 0.4 + mag_score * 0.3 + consistency_score * 0.3
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        if self.orientation == "vertical":
            return KLD7Angle(
                vertical_deg=round(avg_angle, 1),
                horizontal_deg=None,
                distance_m=round(avg_distance, 2),
                magnitude=max_magnitude,
                confidence=confidence,
                num_frames=num_frames,
            )
        else:
            return KLD7Angle(
                vertical_deg=None,
                horizontal_deg=round(avg_angle, 1),
                distance_m=round(avg_distance, 2),
                magnitude=max_magnitude,
                confidence=confidence,
                num_frames=num_frames,
            )

    def reset(self):
        """Clear the ring buffer after a shot is processed."""
        self._ring_buffer.clear()
