"""Tests for K-LD7 angle radar integration."""

import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openflight.kld7.tracker import KLD7Tracker
from openflight.kld7.types import KLD7Angle, KLD7Frame
from openflight.launch_monitor import Shot
from openflight.server import shot_to_dict

# Path to real captured K-LD7 data (golf swings + body movement)
CAPTURE_PATH = Path(__file__).parent.parent / "session_logs" / "kld7_capture_20260329_095614.pkl"


class TestKLD7SerialIO:
    """Tests for low-level K-LD7 serial read recovery helpers."""

    def test_robust_read_packet_wraps_serial_read_failures(self, monkeypatch):
        fake_kld7 = ModuleType("kld7")

        class FakeKLD7Exception(Exception):
            pass

        fake_kld7.KLD7Exception = FakeKLD7Exception
        monkeypatch.setitem(sys.modules, "kld7", fake_kld7)

        from openflight.kld7.serial_io import install_robust_read_packet

        class FailingPort:
            def read(self, _size):
                raise RuntimeError(
                    "device reports readiness to read but returned no data "
                    "(device disconnected or multiple access on port?)"
                )

        radar = SimpleNamespace(_port=FailingPort())
        install_robust_read_packet(radar)

        with pytest.raises(FakeKLD7Exception, match="Serial read failed"):
            radar._read_packet()

    def test_robust_read_packet_wraps_non_ascii_headers(self, monkeypatch):
        fake_kld7 = ModuleType("kld7")

        class FakeKLD7Exception(Exception):
            pass

        fake_kld7.KLD7Exception = FakeKLD7Exception
        monkeypatch.setitem(sys.modules, "kld7", fake_kld7)

        from openflight.kld7.serial_io import install_robust_read_packet

        class GarbledPort:
            def read(self, _size):
                return b"\x83ABC\x00\x00\x00\x00"

        radar = SimpleNamespace(_port=GarbledPort())
        install_robust_read_packet(radar)

        with pytest.raises(FakeKLD7Exception, match="Invalid packet header"):
            radar._read_packet()

    def test_robust_read_packet_rejects_impossible_payload_lengths(self, monkeypatch):
        fake_kld7 = ModuleType("kld7")

        class FakeKLD7Exception(Exception):
            pass

        fake_kld7.KLD7Exception = FakeKLD7Exception
        monkeypatch.setitem(sys.modules, "kld7", fake_kld7)

        from openflight.kld7.serial_io import install_robust_read_packet

        class GarbledPort:
            def read(self, size):
                if size == 8:
                    return b"RADC" + (786499).to_bytes(4, "little")
                return b""

        radar = SimpleNamespace(_port=GarbledPort())
        install_robust_read_packet(radar)

        with pytest.raises(FakeKLD7Exception, match="Invalid packet length"):
            radar._read_packet()

    def test_robust_read_packet_waits_for_trailing_payload_bytes(self, monkeypatch):
        fake_kld7 = ModuleType("kld7")

        class FakeKLD7Exception(Exception):
            pass

        fake_kld7.KLD7Exception = FakeKLD7Exception
        monkeypatch.setitem(sys.modules, "kld7", fake_kld7)

        from openflight.kld7.serial_io import install_robust_read_packet

        class DelayedTailPort:
            def __init__(self):
                self.timeout = 0.5
                self.reads = []

            def read(self, size):
                self.reads.append((size, self.timeout))
                if len(self.reads) == 1:
                    return b"RADC" + (3072).to_bytes(4, "little")
                if len(self.reads) == 2:
                    return b"\x01" * 3071
                if len(self.reads) == 3:
                    return b""
                return b"\x02"

        port = DelayedTailPort()
        radar = SimpleNamespace(_port=port)
        install_robust_read_packet(radar)

        reply, payload = radar._read_packet()

        assert reply == "RADC"
        assert payload == b"\x01" * 3071 + b"\x02"
        assert port.timeout == 0.5
        assert any(timeout < 0.5 for _, timeout in port.reads)

    def test_safe_kld7_destructor_suppresses_close_failures(self):
        from openflight.kld7.serial_io import _install_safe_kld7_destructor

        class FakeKLD7:
            def __init__(self):
                self._port = object()

            def __del__(self):
                self.close()

            def close(self):
                raise RuntimeError("serial port already failed")

        _install_safe_kld7_destructor(FakeKLD7)
        radar = FakeKLD7()

        radar.__del__()

        assert radar._port is None


class TestKLD7Types:
    """Tests for K-LD7 data types."""

    def test_kld7_frame_defaults(self):
        frame = KLD7Frame(timestamp=1000.0)
        assert frame.timestamp == 1000.0
        assert frame.radc is None

    def test_kld7_frame_radc_field(self):
        frame = KLD7Frame(timestamp=1000.0)
        assert frame.radc is None
        frame_with_radc = KLD7Frame(timestamp=1000.0, radc=b"\x00" * 3072)
        assert len(frame_with_radc.radc) == 3072

    def test_kld7_angle_vertical(self):
        angle = KLD7Angle(
            vertical_deg=12.5, distance_m=2.0, magnitude=5000, confidence=0.8, num_frames=3
        )
        assert angle.vertical_deg == 12.5
        assert angle.horizontal_deg is None

    def test_kld7_angle_horizontal(self):
        angle = KLD7Angle(
            horizontal_deg=-3.2, distance_m=1.5, magnitude=4000, confidence=0.7, num_frames=2
        )
        assert angle.horizontal_deg == -3.2
        assert angle.vertical_deg is None


class TestKLD7TrackerRingBuffer:
    """Tests for ring buffer and basic operations."""

    def _make_tracker(self, orientation="vertical"):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()
        return tracker

    def test_ring_buffer_stores_frames(self):
        tracker = self._make_tracker(orientation="horizontal")
        now = time.time()
        for i in range(5):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.03))
        assert len(tracker._ring_buffer) == 5

    def test_ring_buffer_max_size(self):
        tracker = self._make_tracker(orientation="horizontal")
        tracker.max_buffer_frames = 10
        tracker._ring_buffer = __import__("collections").deque(maxlen=10)
        now = time.time()
        for i in range(20):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.03))
        assert len(tracker._ring_buffer) == 10

    def test_reset_clears_buffer(self):
        tracker = self._make_tracker()
        tracker._add_frame(KLD7Frame(timestamp=time.time()))
        assert len(tracker._ring_buffer) == 1
        tracker.reset()
        assert len(tracker._ring_buffer) == 0

    def test_snapshot_buffer(self):
        tracker = self._make_tracker()
        now = time.time()
        tracker._add_frame(KLD7Frame(timestamp=now, radc=b"\x00" * 3072))
        snap = tracker.snapshot_buffer()
        assert len(snap) == 1
        assert snap[0]["timestamp"] == now
        assert snap[0]["has_radc"] is True
        # tdat/pdat are no longer collected and should not appear in snapshot
        assert "tdat" not in snap[0]
        assert "pdat" not in snap[0]

    def test_snapshot_buffer_can_include_raw_radc_for_experiments(self):
        tracker = self._make_tracker()
        tracker._add_frame(KLD7Frame(timestamp=1000.0, radc=b"\x01\x02\x03"))

        snap = tracker.snapshot_buffer(include_radc_payload=True)

        assert snap == [
            {
                "timestamp": 1000.0,
                "has_radc": True,
                "radc_b64": "AQID",
                "radc_payload_bytes": 3,
                "radc_payload_valid": False,
            }
        ]

    def test_snapshot_buffer_omits_has_radc_when_no_radc(self):
        tracker = self._make_tracker()
        now = time.time()
        tracker._add_frame(KLD7Frame(timestamp=now))
        snap = tracker.snapshot_buffer()
        assert len(snap) == 1
        assert "has_radc" not in snap[0]

    def test_returns_none_when_no_detections(self):
        tracker = self._make_tracker()
        now = time.time()
        for i in range(5):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.03))
        assert tracker.get_angle_for_shot() is None

    def test_connect_fails_fast_when_explicit_dev_symlink_is_missing(self, monkeypatch, caplog):
        """Missing /dev/kld7_* aliases should not retry GBYE against a nonexistent path."""
        import logging

        tracker = KLD7Tracker(port="/dev/kld7_vertical", orientation="vertical")
        connect_with_recovery = Mock()
        monkeypatch.setattr(
            "openflight.kld7.tracker.find_spec",
            lambda name: object() if name == "kld7" else None,
        )
        monkeypatch.setattr(
            "openflight.kld7.tracker.Path.exists",
            lambda path: str(path) == "/dev/ttyUSB0",
        )
        monkeypatch.setattr(
            "openflight.kld7.tracker.glob.glob",
            lambda pattern: ["/dev/ttyUSB0"] if pattern == "/dev/ttyUSB*" else [],
        )
        monkeypatch.setattr(
            "openflight.kld7.serial_io.connect_with_recovery",
            connect_with_recovery,
        )

        with caplog.at_level(logging.ERROR, logger="openflight.kld7.tracker"):
            connected = tracker.connect()

        assert connected is False
        connect_with_recovery.assert_not_called()
        assert "Configured K-LD7 port does not exist: /dev/kld7_vertical" in caplog.text
        assert "/dev/ttyUSB0" in caplog.text

    def test_stream_loop_recovers_from_serial_no_data_error(self, monkeypatch):
        """A transient serial read failure should not kill the K-LD7 stream."""
        fake_kld7 = ModuleType("kld7")
        fake_kld7.FrameCode = SimpleNamespace(RADC="RADC")

        class FakeKLD7Exception(Exception):
            pass

        fake_kld7.KLD7Exception = FakeKLD7Exception
        monkeypatch.setitem(sys.modules, "kld7", fake_kld7)

        tracker = self._make_tracker(orientation="horizontal")
        tracker._running = True

        class FakeRadar:
            def __init__(self):
                self.calls = 0
                self.drain_calls = 0

            def stream_frames(self, frame_codes, max_count=-1, min_frame_interval=0):
                self.calls += 1
                yield ("RADC", bytes([self.calls]) * 3072)
                if self.calls == 1:
                    raise RuntimeError(
                        "device reports readiness to read but returned no data "
                        "(device disconnected or multiple access on port?)"
                    )
                tracker._running = False

            def _drain_serial(self):
                self.drain_calls += 1

        radar = FakeRadar()
        tracker._radar = radar
        monkeypatch.setattr("openflight.kld7.tracker.time.sleep", lambda _: None)

        tracker._stream_loop()

        assert radar.calls == 2
        assert radar.drain_calls == 1
        assert len(tracker._ring_buffer) == 2

    def test_stream_error_drain_resets_input_buffer(self, monkeypatch):
        """Recovering from a desync should discard stale streaming bytes."""
        tracker = self._make_tracker(orientation="horizontal")
        tracker._running = True

        class FakePort:
            def __init__(self):
                self.reset_calls = 0

            def reset_input_buffer(self):
                self.reset_calls += 1

        class FakeRadar:
            def __init__(self):
                self._port = FakePort()
                self.drain_calls = 0

            def _drain_serial(self):
                self.drain_calls += 1

        radar = FakeRadar()
        tracker._radar = radar
        monkeypatch.setattr("openflight.kld7.tracker.time.sleep", lambda _: None)

        tracker._drain_after_stream_error()

        assert radar.drain_calls == 1
        assert radar._port.reset_calls == 2

    def test_stream_error_drain_discards_bytes_that_arrive_during_settle(self, monkeypatch):
        """Trailing payload bytes can arrive after the first reset/drain pass."""
        tracker = self._make_tracker(orientation="horizontal")
        tracker._running = True

        class FakePort:
            def __init__(self):
                self.pending = bytearray()

            def reset_input_buffer(self):
                self.pending.clear()

        class FakeRadar:
            def __init__(self):
                self._port = FakePort()

            def _drain_serial(self):
                self._port.pending.clear()

        radar = FakeRadar()
        tracker._radar = radar

        def delayed_stale_byte(_seconds):
            radar._port.pending.extend(b"\x00")

        monkeypatch.setattr("openflight.kld7.tracker.time.sleep", delayed_stale_byte)

        tracker._drain_after_stream_error()

        assert radar._port.pending == bytearray()

    def test_stream_loop_throttles_radc_requests(self, monkeypatch):
        """RADC streaming should leave a small margin below max frame rate."""
        fake_kld7 = ModuleType("kld7")
        fake_kld7.FrameCode = SimpleNamespace(RADC="RADC")

        class FakeKLD7Exception(Exception):
            pass

        fake_kld7.KLD7Exception = FakeKLD7Exception
        monkeypatch.setitem(sys.modules, "kld7", fake_kld7)

        tracker = self._make_tracker(orientation="horizontal")
        tracker._running = True

        class FakeRadar:
            def __init__(self):
                self.min_frame_interval = None

            def stream_frames(self, frame_codes, max_count=-1, min_frame_interval=None):
                self.min_frame_interval = min_frame_interval
                yield ("RADC", b"\x7f" * 3072)
                tracker._running = False

        radar = FakeRadar()
        tracker._radar = radar
        monkeypatch.setattr("openflight.kld7.tracker.time.sleep", lambda _: None)

        tracker._stream_loop()

        assert radar.min_frame_interval == pytest.approx(0.04)

    def test_stream_loop_reconnects_after_consecutive_timeouts(self, monkeypatch):
        """Repeated command timeouts should trigger a full K-LD7 reconnect."""
        fake_kld7 = ModuleType("kld7")
        fake_kld7.FrameCode = SimpleNamespace(RADC="RADC")

        class FakeKLD7Exception(Exception):
            pass

        fake_kld7.KLD7Exception = FakeKLD7Exception
        monkeypatch.setitem(sys.modules, "kld7", fake_kld7)

        tracker = self._make_tracker(orientation="horizontal")
        tracker._running = True

        class TimeoutRadar:
            def __init__(self):
                self.calls = 0
                self.drain_calls = 0
                self.closed = False

            def stream_frames(self, frame_codes, max_count=-1, min_frame_interval=0):
                self.calls += 1
                raise FakeKLD7Exception("Timeout waiting for reply")

            def _drain_serial(self):
                self.drain_calls += 1

            def close(self):
                self.closed = True

        class RecoveredRadar:
            def stream_frames(self, frame_codes, max_count=-1, min_frame_interval=0):
                yield ("RADC", b"\x7f" * 3072)
                tracker._running = False

        timeout_radar = TimeoutRadar()
        reconnects = []

        def fake_connect():
            reconnects.append(True)
            tracker._radar = RecoveredRadar()
            return True

        tracker._radar = timeout_radar
        tracker.connect = fake_connect
        monkeypatch.setattr("openflight.kld7.tracker.time.sleep", lambda _: None)

        tracker._stream_loop()

        assert timeout_radar.calls == 10
        assert timeout_radar.drain_calls == 9
        assert timeout_radar.closed is True
        assert len(reconnects) == 1
        assert len(tracker._ring_buffer) == 1


class TestKLD7RealData:
    """Tests against real captured K-LD7 data."""

    def _make_tracker(self, orientation="vertical"):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()
        return tracker

    def _load_frames(self):
        if not CAPTURE_PATH.exists():
            pytest.skip(f"Capture file not found: {CAPTURE_PATH}")
        with open(CAPTURE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["frames"]

    def test_rejects_body_movement_from_real_data(self):
        """Body movement window should produce no ball detection."""
        raw_frames = self._load_frames()
        tracker = self._make_tracker()
        t0 = raw_frames[0]["timestamp"]
        for f in raw_frames:
            t = f["timestamp"] - t0
            if 0.4 <= t <= 4.0:
                tracker._add_frame(KLD7Frame(timestamp=f["timestamp"]))
        assert tracker.get_angle_for_shot() is None

    def test_quiet_period_produces_no_results(self):
        """A quiet period in real data should produce no results."""
        raw_frames = self._load_frames()
        tracker = self._make_tracker()
        t0 = raw_frames[0]["timestamp"]
        for f in raw_frames:
            t = f["timestamp"] - t0
            if 19.0 <= t <= 24.0:
                tracker._add_frame(KLD7Frame(timestamp=f["timestamp"]))
        assert tracker.get_angle_for_shot() is None


class TestKLD7Integration:
    """Integration tests for K-LD7 angle data flowing through to Shot."""

    def test_angle_attaches_to_shot_vertical(self):
        shot = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
            launch_angle_vertical=12.5,
            launch_angle_confidence=0.8,
            angle_source="radar",
        )
        result = shot_to_dict(shot)
        assert result["launch_angle_vertical"] == 12.5
        assert result["angle_source"] == "radar"

    def test_angle_attaches_to_shot_horizontal(self):
        shot = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
            launch_angle_horizontal=-3.5,
            launch_angle_confidence=0.7,
            angle_source="radar",
        )
        result = shot_to_dict(shot)
        assert result["launch_angle_horizontal"] == -3.5

    def test_carry_adjusts_for_vertical_angle(self):
        shot_no_angle = Shot(ball_speed_mph=150.0, timestamp=datetime.now())
        shot_with_angle = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
            launch_angle_vertical=15.0,
            launch_angle_confidence=0.8,
            angle_source="radar",
        )
        assert shot_no_angle.estimated_carry_yards != shot_with_angle.estimated_carry_yards

    def test_club_angle_in_shot_dict(self):
        shot = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
            club_angle_deg=-5.5,
        )
        result = shot_to_dict(shot)
        assert result["club_angle_deg"] == -5.5

    def test_full_tracker_to_shot_flow(self):
        """Full flow: KLD7Angle manually attached to Shot appears in shot_to_dict."""
        angle = KLD7Angle(
            vertical_deg=18.0,
            horizontal_deg=None,
            confidence=0.85,
            num_frames=3,
            magnitude=5.2,
            detection_class="ball",
        )

        shot = Shot(ball_speed_mph=150.0, timestamp=datetime.now())
        shot.launch_angle_vertical = angle.vertical_deg
        shot.launch_angle_confidence = angle.confidence
        shot.angle_source = "radar"

        result = shot_to_dict(shot)
        assert result["launch_angle_vertical"] == 18.0
        assert result["angle_source"] == "radar"


class TestRADCAngleExtraction:
    """Tests for RADC-based phase-interferometry launch angle extraction."""

    def _make_tracker(self, orientation="vertical"):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker.angle_offset_deg = 0.0
        tracker._init_ring_buffer()
        return tracker

    def _make_radc_payload_with_tone(self, velocity_kmh, angle_deg=10.0, amplitude=5000):
        """Create a synthetic RADC payload with a tone at the given velocity."""
        from openflight.kld7.radc import ANTENNA_SPACING_M, SAMPLES_PER_CHANNEL, WAVELENGTH_M

        n = SAMPLES_PER_CHANNEL  # 256
        max_speed_kmh = 100.0

        # Velocity to normalized frequency
        if velocity_kmh >= 0:
            norm_freq = velocity_kmh / (2 * max_speed_kmh)
        else:
            norm_freq = 1.0 + velocity_kmh / (2 * max_speed_kmh)

        t = np.arange(n)
        phase_per_sample = 2 * np.pi * norm_freq

        # F1A channel: reference (compute in float, then convert to uint16 with DC offset)
        f1a_i = (amplitude * np.cos(phase_per_sample * t) + 32768).astype(np.uint16)
        f1a_q = (amplitude * np.sin(phase_per_sample * t) + 32768).astype(np.uint16)

        # F2A channel: same tone shifted by angle-dependent phase
        angle_rad = np.radians(angle_deg)
        steering_phase = 2 * np.pi * ANTENNA_SPACING_M * np.sin(angle_rad) / WAVELENGTH_M
        f2a_i = (amplitude * np.cos(phase_per_sample * t + steering_phase) + 32768).astype(
            np.uint16
        )
        f2a_q = (amplitude * np.sin(phase_per_sample * t + steering_phase) + 32768).astype(
            np.uint16
        )

        # F1B channel: zeros (not used for angle)
        zeros = np.full(n, 32768, dtype=np.uint16)

        payload = b""
        for ch in [f1a_i, f1a_q, f2a_i, f2a_q, zeros, zeros]:
            payload += ch.astype(np.uint16).tobytes()
        return payload

    def _make_radc_payload_with_tones(self, tones):
        """Create a synthetic RADC payload with multiple velocity/angle tones."""
        from openflight.kld7.radc import ANTENNA_SPACING_M, SAMPLES_PER_CHANNEL, WAVELENGTH_M

        n = SAMPLES_PER_CHANNEL
        max_speed_kmh = 100.0
        t = np.arange(n)
        f1a_i = np.full(n, 32768.0)
        f1a_q = np.full(n, 32768.0)
        f2a_i = np.full(n, 32768.0)
        f2a_q = np.full(n, 32768.0)

        for velocity_kmh, angle_deg, amplitude in tones:
            if velocity_kmh >= 0:
                norm_freq = velocity_kmh / (2 * max_speed_kmh)
            else:
                norm_freq = 1.0 + velocity_kmh / (2 * max_speed_kmh)
            phase_per_sample = 2 * np.pi * norm_freq
            angle_rad = np.radians(angle_deg)
            steering_phase = 2 * np.pi * ANTENNA_SPACING_M * np.sin(angle_rad) / WAVELENGTH_M

            f1a_i += amplitude * np.cos(phase_per_sample * t)
            f1a_q += amplitude * np.sin(phase_per_sample * t)
            f2a_i += amplitude * np.cos(phase_per_sample * t + steering_phase)
            f2a_q += amplitude * np.sin(phase_per_sample * t + steering_phase)

        zeros = np.full(n, 32768, dtype=np.uint16)
        channels = [
            np.clip(f1a_i, 0, 65535).astype(np.uint16),
            np.clip(f1a_q, 0, 65535).astype(np.uint16),
            np.clip(f2a_i, 0, 65535).astype(np.uint16),
            np.clip(f2a_q, 0, 65535).astype(np.uint16),
            zeros,
            zeros,
        ]
        return b"".join(ch.tobytes() for ch in channels)

    def _make_quiet_radc_payload(self, rng=None):
        """Create a quiet RADC payload (DC + small noise, no velocity tone)."""
        from openflight.kld7.radc import SAMPLES_PER_CHANNEL

        if rng is None:
            rng = np.random.default_rng(42)
        n = SAMPLES_PER_CHANNEL
        payload = b""
        for _ in range(6):
            noise = (32768 + rng.integers(-50, 50, size=n)).astype(np.uint16)
            payload += noise.tobytes()
        return payload

    def test_extracts_angle_from_radc_with_ball_speed(self):
        """RADC extraction should find the angle at the OPS-anchored velocity bin."""
        tracker = self._make_tracker()
        now = time.time()

        ball_speed_mph = 72.0
        ball_kmh = ball_speed_mph * 1.609
        aliased_kmh = ball_kmh % 200.0
        if aliased_kmh > 100.0:
            aliased_kmh -= 200.0
        # `_make_radc_payload_with_tone` shifts F2A by +steering_phase,
        # but `per_bin_angle_deg` derives the angle from
        # f1a * conj(f2a) — so feeding angle_deg=θ into the synthetic
        # payload yields a measured angle of -θ. To assert a positive
        # output we inject the negated angle at the synth layer.
        synth_angle = -12.0
        target_angle = -synth_angle  # +12 — what the algorithm should return
        radc = self._make_radc_payload_with_tone(aliased_kmh, angle_deg=synth_angle)
        quiet = self._make_quiet_radc_payload()

        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.056, radc=quiet))
        tracker._add_frame(KLD7Frame(timestamp=now + 0.56, radc=radc))
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.62 + i * 0.056, radc=quiet))

        result = tracker.get_angle_for_shot(ball_speed_mph=ball_speed_mph)
        assert result is not None
        assert result.detection_class == "ball"
        assert result.vertical_deg == pytest.approx(target_angle, abs=3.0)
        assert result.confidence > 0.0

    def test_ops_anchor_prefers_correct_speed_peak_over_stronger_clutter(self):
        """A stronger in-band clutter peak should not beat the OPS-speed ball peak."""
        tracker = self._make_tracker(orientation="horizontal")
        now = time.time()

        ball_speed_mph = 72.0
        ball_kmh = ball_speed_mph * 1.609
        aliased_kmh = ball_kmh % 200.0
        if aliased_kmh > 100.0:
            aliased_kmh -= 200.0

        target_angle = 4.0
        clutter_angle = -14.0
        radc = self._make_radc_payload_with_tones(
            [
                # See test_extracts_angle_from_radc_with_ball_speed for the
                # synthetic sign flip. The lower-amplitude ball peak is near
                # the OPS-expected bin; the stronger clutter peak is elsewhere
                # inside the wider speed-tolerance band.
                (aliased_kmh, -target_angle, 3000.0),
                (-72.0, -clutter_angle, 8000.0),
            ]
        )
        quiet = self._make_quiet_radc_payload()

        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.056, radc=quiet))
        tracker._add_frame(KLD7Frame(timestamp=now + 0.56, radc=radc))
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.62 + i * 0.056, radc=quiet))

        result = tracker.get_angle_for_shot(ball_speed_mph=ball_speed_mph)

        assert result is not None
        assert result.horizontal_deg == pytest.approx(target_angle, abs=4.0)
        assert abs(result.horizontal_deg - clutter_angle) > 8.0

    def test_returns_none_without_ball_speed(self):
        """When ball_speed_mph is None, should return None (RADC requires speed anchor)."""
        tracker = self._make_tracker()
        now = time.time()

        for i in range(3):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.033))

        result = tracker.get_angle_for_shot(ball_speed_mph=None)
        assert result is None

    def test_get_angle_for_shot_filters_stale_radc_frames(self, monkeypatch):
        """Shot timestamp filtering prevents stale ring-buffer frames from influencing a shot."""
        tracker = self._make_tracker()
        tracker.buffer_seconds = 2.0
        shot_ts = 1000.0

        tracker._add_frame(KLD7Frame(timestamp=shot_ts - 10.0, radc=b"stale-old"))
        tracker._add_frame(KLD7Frame(timestamp=shot_ts - 1.2, radc=b"fresh-a"))
        tracker._add_frame(KLD7Frame(timestamp=shot_ts + 0.4, radc=b"fresh-b"))
        tracker._add_frame(KLD7Frame(timestamp=shot_ts + 1.0, radc=b"stale-future"))

        seen_timestamps = []

        def fake_extract_launch_angle(frames, **kwargs):
            seen_timestamps.extend(frame["timestamp"] for frame in frames)
            return [
                {
                    "launch_angle_deg": 2.5,
                    "ball_speed_mph": 80.0,
                    "avg_snr_db": 8.0,
                    "confidence": 0.8,
                    "frame_count": 2,
                    "impact_frames": [1],
                },
                {
                    "launch_angle_deg": 7.5,
                    "ball_speed_mph": 80.0,
                    "avg_snr_db": 8.0,
                    "confidence": 0.8,
                    "frame_count": 2,
                    "impact_frames": [3],
                },
            ]

        monkeypatch.setattr(
            "openflight.kld7.radc.extract_launch_angle",
            fake_extract_launch_angle,
        )

        result = tracker.get_angle_for_shot(
            shot_timestamp=shot_ts,
            ball_speed_mph=80.0,
        )

        assert seen_timestamps == [shot_ts - 1.2, shot_ts + 0.4]
        assert result is not None
        assert result.vertical_deg == pytest.approx(7.5)
        assert result.frames_examined == 2
        assert result.frames_available == 4
        assert result.frames_ignored_stale == 2

    def test_get_angle_for_shot_uses_all_radc_frames_without_timestamp(self, monkeypatch):
        """Legacy callers without a shot timestamp retain all-frame extraction behavior."""
        tracker = self._make_tracker()
        tracker._add_frame(KLD7Frame(timestamp=1000.0, radc=b"a"))
        tracker._add_frame(KLD7Frame(timestamp=1010.0, radc=b"b"))

        frame_counts = []

        def fake_extract_launch_angle(frames, **kwargs):
            frame_counts.append(len(frames))
            return [
                {
                    "launch_angle_deg": 7.5,
                    "ball_speed_mph": 80.0,
                    "avg_snr_db": 8.0,
                    "confidence": 0.8,
                    "frame_count": 2,
                }
            ]

        monkeypatch.setattr(
            "openflight.kld7.radc.extract_launch_angle",
            fake_extract_launch_angle,
        )

        result = tracker.get_angle_for_shot(ball_speed_mph=80.0)

        assert frame_counts == [2]
        assert result is not None
        assert result.frames_examined == 2
        assert result.frames_available == 2
        assert result.frames_ignored_stale == 0

    def test_angle_offset_applied_to_radc(self):
        """Angle offset should be applied to RADC-extracted angle."""
        tracker = self._make_tracker()
        tracker.angle_offset_deg = 5.0
        now = time.time()

        ball_speed_mph = 72.0
        ball_kmh = ball_speed_mph * 1.609
        aliased_kmh = ball_kmh % 200.0
        if aliased_kmh > 100.0:
            aliased_kmh -= 200.0

        # See test_extracts_angle_from_radc_with_ball_speed for the
        # sign-flip explanation. Inject -10° to get a measured +10° pre-offset.
        radc = self._make_radc_payload_with_tone(aliased_kmh, angle_deg=-10.0)
        quiet = self._make_quiet_radc_payload()

        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.056, radc=quiet))
        tracker._add_frame(KLD7Frame(timestamp=now + 0.56, radc=radc))
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.62 + i * 0.056, radc=quiet))

        result = tracker.get_angle_for_shot(ball_speed_mph=ball_speed_mph)
        assert result is not None
        # measured +10 + offset 5 = +15
        assert result.vertical_deg == pytest.approx(15.0, abs=3.0)

    def test_horizontal_radc_retries_with_relaxed_energy_threshold(self, monkeypatch):
        """Horizontal extraction should retry low-energy coherent misses."""
        tracker = self._make_tracker(orientation="horizontal")
        tracker._add_frame(KLD7Frame(timestamp=time.time(), radc=b"\x00" * 3072))
        calls = []

        def fake_extract_launch_angle(frames, **kwargs):
            calls.append(kwargs["impact_energy_threshold"])
            if kwargs["impact_energy_threshold"] == 0.5:
                return [
                    {
                        "launch_angle_deg": 2.4,
                        "ball_speed_mph": 80.0,
                        "avg_snr_db": 2.3,
                        "confidence": 0.72,
                        "frame_count": 12,
                    }
                ]
            return []

        monkeypatch.setattr(
            "openflight.kld7.radc.extract_launch_angle",
            fake_extract_launch_angle,
        )

        result = tracker.get_angle_for_shot(ball_speed_mph=80.0)

        assert calls == [1.85, 0.5]
        assert result is not None
        assert result.horizontal_deg == pytest.approx(2.4)
        assert result.confidence == pytest.approx(0.45)

    def test_horizontal_radc_retries_weak_wall_candidate(self, monkeypatch):
        """Sparse, low-SNR wall-angle hits should not block the relaxed retry."""
        tracker = self._make_tracker(orientation="horizontal")
        tracker._add_frame(KLD7Frame(timestamp=time.time(), radc=b"\x00" * 3072))
        calls = []

        def fake_extract_launch_angle(frames, **kwargs):
            calls.append(kwargs["impact_energy_threshold"])
            if kwargs["impact_energy_threshold"] == 1.85:
                return [
                    {
                        "launch_angle_deg": 14.4,
                        "ball_speed_mph": 80.0,
                        "avg_snr_db": 2.6,
                        "confidence": 0.54,
                        "frame_count": 2,
                    }
                ]
            return [
                {
                    "launch_angle_deg": -3.1,
                    "ball_speed_mph": 80.0,
                    "avg_snr_db": 2.5,
                    "confidence": 0.63,
                    "frame_count": 18,
                }
            ]

        monkeypatch.setattr(
            "openflight.kld7.radc.extract_launch_angle",
            fake_extract_launch_angle,
        )

        result = tracker.get_angle_for_shot(ball_speed_mph=80.0)

        assert calls == [1.85, 0.5]
        assert result is not None
        assert result.horizontal_deg == pytest.approx(-3.1)
        assert result.confidence == pytest.approx(0.45)

    def test_horizontal_radc_keeps_strong_wall_candidate(self, monkeypatch):
        """A well-supported edge angle can still be a real shot shape."""
        tracker = self._make_tracker(orientation="horizontal")
        tracker._add_frame(KLD7Frame(timestamp=time.time(), radc=b"\x00" * 3072))
        calls = []

        def fake_extract_launch_angle(frames, **kwargs):
            calls.append(kwargs["impact_energy_threshold"])
            return [
                {
                    "launch_angle_deg": 13.2,
                    "ball_speed_mph": 80.0,
                    "avg_snr_db": 6.0,
                    "confidence": 0.72,
                    "frame_count": 5,
                }
            ]

        monkeypatch.setattr(
            "openflight.kld7.radc.extract_launch_angle",
            fake_extract_launch_angle,
        )

        result = tracker.get_angle_for_shot(ball_speed_mph=80.0)

        assert calls == [1.85]
        assert result is not None
        assert result.horizontal_deg == pytest.approx(13.2)
        assert result.confidence == pytest.approx(0.72)

    def test_vertical_radc_does_not_use_low_energy_retry(self, monkeypatch):
        """The relaxed retry is intentionally horizontal-only."""
        tracker = self._make_tracker(orientation="vertical")
        tracker._add_frame(KLD7Frame(timestamp=time.time(), radc=b"\x00" * 3072))
        calls = []

        def fake_extract_launch_angle(frames, **kwargs):
            calls.append(kwargs["impact_energy_threshold"])
            return []

        monkeypatch.setattr(
            "openflight.kld7.radc.extract_launch_angle",
            fake_extract_launch_angle,
        )

        result = tracker.get_angle_for_shot(ball_speed_mph=80.0)

        assert calls == [3.0]
        assert result is None

    def test_radc_extraction_passes_tunable_parameters(self, monkeypatch):
        """Live extraction should use the same knobs as the replay tool."""
        tracker = self._make_tracker(orientation="vertical")
        tracker.radc_speed_tolerance_mph = 8.0
        tracker.radc_centroid_floor_frac = 0.65
        tracker.radc_ops_bin_outlier_tol = 12
        tracker.radc_ops_bin_outlier_penalty = 4.0
        tracker.radc_ops_anchored_peak_min_snr = 2.5
        tracker.radc_vertical_impact_energy_threshold = 2.5
        tracker.radc_horizontal_angle_limit_deg = 30.0
        tracker._add_frame(KLD7Frame(timestamp=time.time(), radc=b"\x00" * 3072))
        captured_kwargs = []

        def fake_extract_launch_angle(frames, **kwargs):
            captured_kwargs.append(kwargs)
            return [
                {
                    "launch_angle_deg": 8.0,
                    "ball_speed_mph": 80.0,
                    "avg_snr_db": 5.0,
                    "confidence": 0.8,
                    "frame_count": 4,
                }
            ]

        monkeypatch.setattr(
            "openflight.kld7.radc.extract_launch_angle",
            fake_extract_launch_angle,
        )

        result = tracker.get_angle_for_shot(ball_speed_mph=80.0)

        assert result is not None
        assert captured_kwargs == [
            {
                "ops243_ball_speed_mph": 80.0,
                "angle_offset_deg": 0.0,
                "speed_tolerance_mph": 8.0,
                "impact_energy_threshold": 2.5,
                "centroid_floor_frac": 0.65,
                "ops_bin_outlier_tol": 12,
                "ops_bin_outlier_penalty": 4.0,
                "ops_anchored_peak_min_snr": 2.5,
                "horizontal_angle_limit_deg": 30.0,
                "orientation": "vertical",
            }
        ]
