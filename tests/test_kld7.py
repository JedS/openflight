"""Tests for K-LD7 angle radar integration."""

import time
from datetime import datetime

import pytest

from openflight.kld7.types import KLD7Angle, KLD7Frame
from openflight.kld7.tracker import KLD7Tracker
from openflight.launch_monitor import Shot, ClubType
from openflight.server import shot_to_dict


class TestKLD7Types:
    """Tests for K-LD7 data types."""

    def test_kld7_frame_defaults(self):
        frame = KLD7Frame(timestamp=1000.0)
        assert frame.timestamp == 1000.0
        assert frame.tdat is None
        assert frame.pdat == []

    def test_kld7_angle_vertical(self):
        angle = KLD7Angle(vertical_deg=12.5, distance_m=2.0, magnitude=5000, confidence=0.8, num_frames=3)
        assert angle.vertical_deg == 12.5
        assert angle.horizontal_deg is None

    def test_kld7_angle_horizontal(self):
        angle = KLD7Angle(horizontal_deg=-3.2, distance_m=1.5, magnitude=4000, confidence=0.7, num_frames=2)
        assert angle.horizontal_deg == -3.2
        assert angle.vertical_deg is None


class TestKLD7TrackerRingBuffer:
    """Tests for ring buffer and angle extraction logic (no hardware)."""

    def _make_tracker(self, orientation="vertical"):
        """Create a tracker without connecting to hardware."""
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()
        return tracker

    def test_ring_buffer_stores_frames(self):
        tracker = self._make_tracker()
        now = time.time()
        for i in range(5):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.03,
                tdat={"distance": 1.0, "speed": 5.0, "angle": 10.0 + i, "magnitude": 3000 + i * 100},
                pdat=[],
            ))
        assert len(tracker._ring_buffer) == 5

    def test_ring_buffer_max_size(self):
        tracker = self._make_tracker()
        tracker.max_buffer_frames = 10
        tracker._ring_buffer = __import__('collections').deque(maxlen=10)
        now = time.time()
        for i in range(20):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.03,
                tdat={"distance": 1.0, "speed": 5.0, "angle": 0.0, "magnitude": 1000},
                pdat=[],
            ))
        assert len(tracker._ring_buffer) == 10

    def test_get_angle_finds_highest_magnitude_event(self):
        tracker = self._make_tracker(orientation="vertical")
        now = time.time()
        # Background noise frames (no detections)
        for i in range(10):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.03,
                tdat=None,
                pdat=[],
            ))
        # Ball pass: 3 frames with high magnitude at angle ~15°
        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.30 + i * 0.03,
                tdat={"distance": 2.0, "speed": 50.0, "angle": 14.0 + i, "magnitude": 5000 + i * 100},
                pdat=[{"distance": 2.0, "speed": 50.0, "angle": 14.0 + i, "magnitude": 5000 + i * 100}],
            ))
        # More noise after
        for i in range(5):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.50 + i * 0.03,
                tdat=None,
                pdat=[],
            ))

        result = tracker.get_angle_for_shot()
        assert result is not None
        assert result.vertical_deg is not None
        assert 13.0 < result.vertical_deg < 17.0
        assert result.horizontal_deg is None
        assert result.num_frames == 3
        assert result.confidence > 0.0
        assert result.distance_m > 0.0

    def test_get_angle_returns_none_when_no_detections(self):
        tracker = self._make_tracker()
        now = time.time()
        for i in range(5):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.03, tdat=None, pdat=[]))
        result = tracker.get_angle_for_shot()
        assert result is None

    def test_get_angle_horizontal_orientation(self):
        tracker = self._make_tracker(orientation="horizontal")
        now = time.time()
        tracker._add_frame(KLD7Frame(
            timestamp=now,
            tdat={"distance": 1.5, "speed": 30.0, "angle": -5.0, "magnitude": 4500},
            pdat=[{"distance": 1.5, "speed": 30.0, "angle": -5.0, "magnitude": 4500}],
        ))
        result = tracker.get_angle_for_shot()
        assert result is not None
        assert result.horizontal_deg is not None
        assert result.vertical_deg is None

    def test_reset_clears_buffer(self):
        tracker = self._make_tracker()
        tracker._add_frame(KLD7Frame(timestamp=time.time(), tdat={"distance": 1.0, "speed": 5.0, "angle": 0.0, "magnitude": 3000}, pdat=[]))
        assert len(tracker._ring_buffer) == 1
        tracker.reset()
        assert len(tracker._ring_buffer) == 0

    def test_prefers_pdat_over_tdat(self):
        """PDAT raw detections should be preferred for angle extraction."""
        tracker = self._make_tracker(orientation="vertical")
        now = time.time()
        # Frame with TDAT at 10° but PDAT at 20° (higher magnitude)
        # Both speeds above MIN_SPEED_KMH (10 km/h) to pass speed filter
        tracker._add_frame(KLD7Frame(
            timestamp=now,
            tdat={"distance": 1.0, "speed": 30.0, "angle": 10.0, "magnitude": 3000},
            pdat=[{"distance": 1.5, "speed": 40.0, "angle": 20.0, "magnitude": 5000}],
        ))
        result = tracker.get_angle_for_shot()
        assert result is not None
        assert abs(result.vertical_deg - 20.0) < 1.0


class TestKLD7NoiseFiltering:
    """Tests for signal processing: rejecting noise, accepting ball events."""

    def _make_tracker(self, orientation="vertical"):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()
        return tracker

    def test_rejects_slow_body_movement(self):
        """Body movement at ~1.6 km/h should be rejected even with high magnitude."""
        tracker = self._make_tracker()
        now = time.time()
        # Simulate body movement: slow speed, wide angle spread, many frames
        for i in range(30):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                tdat={"distance": 1.5, "speed": 1.6, "angle": -40.0 + i * 3.0, "magnitude": 4000},
                pdat=[{"distance": 1.5, "speed": 1.6, "angle": -40.0 + i * 3.0, "magnitude": 4000}],
            ))
        result = tracker.get_angle_for_shot()
        assert result is None, "Body movement at 1.6 km/h should be rejected"

    def test_rejects_wide_angle_spread_events(self):
        """Events with >60° angle spread are noise (body/arm movement)."""
        tracker = self._make_tracker()
        now = time.time()
        # Wide angle spread event: angles from -50 to +50
        angles = [-50, -30, -10, 10, 30, 50]
        for i, ang in enumerate(angles):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                tdat={"distance": 2.0, "speed": 5.0, "angle": ang, "magnitude": 5000},
                pdat=[{"distance": 2.0, "speed": 5.0, "angle": ang, "magnitude": 5000}],
            ))
        result = tracker.get_angle_for_shot()
        assert result is None, "Wide angle spread (100°) should be rejected as noise"

    def test_rejects_long_duration_events(self):
        """Events lasting >1 second are body movement, not a ball pass."""
        tracker = self._make_tracker()
        now = time.time()
        # 2-second continuous event (body walking through beam)
        for i in range(60):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                tdat={"distance": 1.5 + i * 0.02, "speed": 3.0, "angle": 10.0 + (i % 5), "magnitude": 3500},
                pdat=[],
            ))
        result = tracker.get_angle_for_shot()
        assert result is None, "2-second continuous event should be rejected"

    def test_accepts_transient_high_speed_ball(self):
        """A short, high-speed, high-magnitude event should be accepted."""
        tracker = self._make_tracker()
        now = time.time()
        # Background noise
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.033, tdat=None, pdat=[]))
        # Ball pass: 2 frames, high speed, tight angle
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.33 + i * 0.033,
                tdat={"distance": 2.0, "speed": 50.0, "angle": 12.0 + i * 0.5, "magnitude": 5500},
                pdat=[{"distance": 2.0, "speed": 50.0, "angle": 12.0 + i * 0.5, "magnitude": 5500}],
            ))
        # More empty frames
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.5 + i * 0.033, tdat=None, pdat=[]))

        result = tracker.get_angle_for_shot()
        assert result is not None, "Short high-speed ball event should be accepted"
        assert 11.0 < result.vertical_deg < 14.0

    def test_ball_extracted_from_noisy_buffer(self):
        """Ball event should be found even when surrounded by noise frames."""
        tracker = self._make_tracker()
        now = time.time()
        # Noise: slow body movement
        for i in range(15):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                tdat={"distance": 1.5, "speed": 1.6, "angle": -20.0 + i * 2.0, "magnitude": 3000},
                pdat=[],
            ))
        # Gap
        for i in range(5):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.5 + i * 0.033, tdat=None, pdat=[]))
        # Ball: high speed, tight angle, transient
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.7 + i * 0.033,
                tdat={"distance": 2.5, "speed": 45.0, "angle": 15.0, "magnitude": 5000},
                pdat=[{"distance": 2.5, "speed": 45.0, "angle": 15.0, "magnitude": 5000}],
            ))
        # More noise after
        for i in range(10):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 1.0 + i * 0.033,
                tdat={"distance": 1.2, "speed": 1.6, "angle": 30.0 + i, "magnitude": 2800},
                pdat=[],
            ))

        result = tracker.get_angle_for_shot()
        assert result is not None, "Ball event should be found amid noise"
        assert 14.0 < result.vertical_deg < 16.0

    def test_confidence_low_for_single_frame_detection(self):
        """Single-frame detections should have lower confidence."""
        tracker = self._make_tracker()
        now = time.time()
        tracker._add_frame(KLD7Frame(
            timestamp=now,
            tdat={"distance": 2.0, "speed": 40.0, "angle": 10.0, "magnitude": 4500},
            pdat=[{"distance": 2.0, "speed": 40.0, "angle": 10.0, "magnitude": 4500}],
        ))
        result = tracker.get_angle_for_shot()
        assert result is not None
        assert result.confidence < 0.7, "Single frame should have lower confidence"

    def test_rejects_low_magnitude_detections(self):
        """Low magnitude detections (below threshold) should be rejected."""
        tracker = self._make_tracker()
        now = time.time()
        # Weak detections — magnitude below minimum threshold
        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                tdat={"distance": 2.0, "speed": 30.0, "angle": 10.0, "magnitude": 500},
                pdat=[{"distance": 2.0, "speed": 30.0, "angle": 10.0, "magnitude": 500}],
            ))
        result = tracker.get_angle_for_shot()
        assert result is None, "Low magnitude detections should be rejected"

    def test_rejects_very_close_range_reflections(self):
        """Detections at <0.3m are likely antenna reflections, not real targets."""
        tracker = self._make_tracker()
        now = time.time()
        tracker._add_frame(KLD7Frame(
            timestamp=now,
            tdat={"distance": 0.1, "speed": 50.0, "angle": 5.0, "magnitude": 5000},
            pdat=[{"distance": 0.1, "speed": 50.0, "angle": 5.0, "magnitude": 5000}],
        ))
        result = tracker.get_angle_for_shot()
        assert result is None, "Very close range (<0.3m) should be rejected as reflection"


class TestKLD7Integration:
    """Integration tests for K-LD7 angle data flowing through to Shot."""

    def test_angle_attaches_to_shot_vertical(self):
        """K-LD7 vertical angle should attach to Shot correctly."""
        shot = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
            launch_angle_vertical=12.5,
            launch_angle_confidence=0.8,
            angle_source="radar",
        )
        result = shot_to_dict(shot)
        assert result["launch_angle_vertical"] == 12.5
        assert result["launch_angle_confidence"] == 0.8
        assert result["angle_source"] == "radar"

    def test_angle_attaches_to_shot_horizontal(self):
        """K-LD7 horizontal angle should attach to Shot correctly."""
        shot = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
            launch_angle_horizontal=-3.5,
            launch_angle_confidence=0.7,
            angle_source="radar",
        )
        result = shot_to_dict(shot)
        assert result["launch_angle_horizontal"] == -3.5
        assert result["angle_source"] == "radar"

    def test_carry_adjusts_for_vertical_angle(self):
        """Shot carry should adjust when vertical angle is provided."""
        shot_no_angle = Shot(ball_speed_mph=150.0, timestamp=datetime.now())
        shot_with_angle = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
            launch_angle_vertical=15.0,
            launch_angle_confidence=0.8,
            angle_source="radar",
        )
        assert shot_no_angle.estimated_carry_yards > 0
        assert shot_with_angle.estimated_carry_yards > 0
        assert shot_no_angle.estimated_carry_yards != shot_with_angle.estimated_carry_yards

    def test_tracker_angle_to_shot_flow(self):
        """Full flow: KLD7Tracker ring buffer -> get_angle -> attach to Shot."""
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = "vertical"
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()

        now = time.time()
        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.03,
                tdat={"distance": 2.0, "speed": 50.0, "angle": 12.0, "magnitude": 5000},
                pdat=[{"distance": 2.0, "speed": 50.0, "angle": 12.0, "magnitude": 5000}],
            ))

        angle = tracker.get_angle_for_shot()
        assert angle is not None

        shot = Shot(
            ball_speed_mph=150.0,
            timestamp=datetime.now(),
        )
        shot.launch_angle_vertical = angle.vertical_deg
        shot.launch_angle_confidence = angle.confidence
        shot.angle_source = "radar"

        result = shot_to_dict(shot)
        assert result["launch_angle_vertical"] == 12.0
        assert result["angle_source"] == "radar"
        assert result["launch_angle_confidence"] > 0.0
