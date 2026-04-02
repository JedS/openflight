"""Tests for K-LD7 angle radar integration."""

import pickle
import time
from datetime import datetime
from pathlib import Path

import pytest

from openflight.kld7.types import KLD7Angle, KLD7Frame
from openflight.kld7.tracker import KLD7Tracker
from openflight.launch_monitor import Shot, ClubType
from openflight.server import shot_to_dict

# Path to real captured K-LD7 data (golf swings + body movement)
CAPTURE_PATH = Path(__file__).parent.parent / "session_logs" / "kld7_capture_20260329_095614.pkl"


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
    """Tests for ring buffer and basic operations."""

    def _make_tracker(self, orientation="vertical"):
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
                timestamp=now + i * 0.03, tdat=None, pdat=[],
            ))
        assert len(tracker._ring_buffer) == 5

    def test_ring_buffer_max_size(self):
        tracker = self._make_tracker()
        tracker.max_buffer_frames = 10
        tracker._ring_buffer = __import__('collections').deque(maxlen=10)
        now = time.time()
        for i in range(20):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.03, tdat=None, pdat=[]))
        assert len(tracker._ring_buffer) == 10

    def test_reset_clears_buffer(self):
        tracker = self._make_tracker()
        tracker._add_frame(KLD7Frame(timestamp=time.time(), tdat=None, pdat=[]))
        assert len(tracker._ring_buffer) == 1
        tracker.reset()
        assert len(tracker._ring_buffer) == 0

    def test_snapshot_buffer(self):
        tracker = self._make_tracker()
        now = time.time()
        tracker._add_frame(KLD7Frame(timestamp=now, tdat={"distance": 1.0, "speed": 5.0, "angle": 0.0, "magnitude": 3000}, pdat=[]))
        snap = tracker.snapshot_buffer()
        assert len(snap) == 1
        assert snap[0]["tdat"]["distance"] == 1.0

    def test_returns_none_when_no_detections(self):
        tracker = self._make_tracker()
        now = time.time()
        for i in range(5):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.03, tdat=None, pdat=[]))
        assert tracker.get_angle_for_shot() is None
        assert tracker.get_club_angle() is None


class TestBallDetection:
    """Tests for ball launch angle extraction (distance-based)."""

    def _make_tracker(self, orientation="vertical"):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()
        return tracker

    def _ball_target(self, angle=15.0, dist=4.2, speed=25.0, mag=2500):
        """Create a target dict matching ball signature (far, fast)."""
        return {"distance": dist, "speed": speed, "angle": angle, "magnitude": mag}

    def _body_target(self, angle=0.0, dist=1.5, speed=2.0, mag=3500):
        """Create a target dict matching body noise (close, slow)."""
        return {"distance": dist, "speed": speed, "angle": angle, "magnitude": mag}

    def test_detects_ball_at_far_range(self):
        """Ball burst: fast targets at >3.8m should be detected."""
        tracker = self._make_tracker()
        now = time.time()
        # Empty frames
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.033, tdat=None, pdat=[]))
        # Ball burst: 2 frames at ~4.2m
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.33 + i * 0.033,
                tdat=None,
                pdat=[self._ball_target(angle=18.0 + i, dist=4.2 + i * 0.1)],
            ))
        # More empty frames
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.5 + i * 0.033, tdat=None, pdat=[]))

        result = tracker.get_angle_for_shot()
        assert result is not None
        assert result.detection_class == "ball"
        assert 17.0 < result.vertical_deg < 20.0
        assert result.distance_m > 3.8

    def test_rejects_slow_targets_at_far_range(self):
        """Slow targets at far range (net reflections) should be rejected."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(5):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 4.0, "speed": 2.0, "angle": 5.0, "magnitude": 3000}],
            ))
        assert tracker.get_angle_for_shot() is None

    def test_rejects_fast_targets_at_close_range(self):
        """Fast targets at close range are club/body, not ball."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 1.5, "speed": 30.0, "angle": -5.0, "magnitude": 4000}],
            ))
        assert tracker.get_angle_for_shot() is None

    def test_rejects_body_movement(self):
        """Body movement at 1.5m, slow speed should produce no ball detection."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(30):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[self._body_target(angle=-40 + i * 3)],
            ))
        assert tracker.get_angle_for_shot() is None

    def test_ball_found_amid_body_noise(self):
        """Ball burst should be found even surrounded by body noise."""
        tracker = self._make_tracker()
        now = time.time()
        # Body noise
        for i in range(15):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[self._body_target(angle=-20 + i * 2)],
            ))
        # Gap
        for i in range(5):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.5 + i * 0.033, tdat=None, pdat=[]))
        # Ball burst
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.7 + i * 0.033, tdat=None,
                pdat=[self._ball_target(angle=20.0, dist=4.3)],
            ))
        # More body noise
        for i in range(10):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 1.0 + i * 0.033, tdat=None,
                pdat=[self._body_target(angle=30 + i)],
            ))

        result = tracker.get_angle_for_shot()
        assert result is not None
        assert 19.0 < result.vertical_deg < 21.0

    def test_rejects_low_magnitude(self):
        """Targets below minimum magnitude should be rejected."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[self._ball_target(mag=200)],
            ))
        assert tracker.get_angle_for_shot() is None

    def test_shot_timestamp_prefers_nearby_burst(self):
        """When shot_timestamp is given, prefer burst closest to that time."""
        tracker = self._make_tracker()
        now = time.time()
        # Earlier burst at angle 30°
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[self._ball_target(angle=30.0, mag=3000)],
            ))
        # Gap
        for i in range(30):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.5 + i * 0.033, tdat=None, pdat=[]))
        # Later burst at angle 12°
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 1.5 + i * 0.033, tdat=None,
                pdat=[self._ball_target(angle=12.0, mag=2000)],
            ))

        # Without timestamp: picks higher magnitude (30°)
        result = tracker.get_angle_for_shot()
        assert result is not None
        assert abs(result.vertical_deg - 30.0) < 2.0

        # With timestamp near later burst: picks 12°
        result_ts = tracker.get_angle_for_shot(shot_timestamp=now + 1.55)
        assert result_ts is not None
        assert abs(result_ts.vertical_deg - 12.0) < 2.0

    def test_horizontal_orientation(self):
        """Ball detection should use horizontal_deg for horizontal orientation."""
        tracker = self._make_tracker(orientation="horizontal")
        now = time.time()
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[self._ball_target(angle=-3.0)],
            ))
        result = tracker.get_angle_for_shot()
        assert result is not None
        assert result.horizontal_deg is not None
        assert result.vertical_deg is None

    def test_tdat_fallback_when_no_pdat(self):
        """Should fall back to TDAT when no PDAT targets qualify."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                tdat={"distance": 4.2, "speed": 20.0, "angle": 15.0, "magnitude": 2500},
                pdat=[],
            ))
        result = tracker.get_angle_for_shot()
        assert result is not None
        assert 14.0 < result.vertical_deg < 16.0

    def test_multi_frame_higher_confidence(self):
        """3-frame burst should have higher confidence than 1-frame."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[self._ball_target(angle=15.0, mag=3000)],
            ))
        result_3 = tracker.get_angle_for_shot()

        tracker2 = self._make_tracker()
        tracker2._add_frame(KLD7Frame(
            timestamp=now, tdat=None,
            pdat=[self._ball_target(angle=15.0, mag=3000)],
        ))
        result_1 = tracker2.get_angle_for_shot()

        assert result_3 is not None
        assert result_1 is not None
        assert result_3.confidence > result_1.confidence


class TestClubDetection:
    """Tests for club angle of attack extraction (speed-transition based)."""

    def _make_tracker(self, orientation="vertical"):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()
        return tracker

    def test_detects_speed_transition(self):
        """Club detected by speed jump from <10 to >=10 km/h at close range."""
        tracker = self._make_tracker()
        now = time.time()
        # Slow frames (body/setup)
        for i in range(5):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 1.5, "speed": 3.0, "angle": -10.0, "magnitude": 3500}],
            ))
        # Speed transition: club approaching ball
        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.2, tdat=None,
            pdat=[
                {"distance": 1.3, "speed": 12.0, "angle": -6.0, "magnitude": 4000},
                {"distance": 1.4, "speed": 11.0, "angle": -5.0, "magnitude": 3800},
            ],
        ))

        result = tracker.get_club_angle()
        assert result is not None
        assert result.detection_class == "club"
        assert -8.0 < result.vertical_deg < -4.0

    def test_rejects_no_speed_transition(self):
        """Continuous slow movement should not trigger club detection."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(20):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 1.5, "speed": 5.0, "angle": -10.0 + i, "magnitude": 3500}],
            ))
        assert tracker.get_club_angle() is None

    def test_rejects_speed_transition_at_far_range(self):
        """Speed transition at >2.5m is not club (ball or net reflection)."""
        tracker = self._make_tracker()
        now = time.time()
        for i in range(5):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 4.0, "speed": 3.0, "angle": 5.0, "magnitude": 3000}],
            ))
        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.2, tdat=None,
            pdat=[{"distance": 4.0, "speed": 15.0, "angle": 5.0, "magnitude": 3000}],
        ))
        assert tracker.get_club_angle() is None

    def test_club_uses_tdat_fallback(self):
        """Club detection should use TDAT when PDAT is empty."""
        tracker = self._make_tracker()
        now = time.time()
        # Slow
        tracker._add_frame(KLD7Frame(
            timestamp=now,
            tdat={"distance": 1.5, "speed": 3.0, "angle": -10.0, "magnitude": 3500},
            pdat=[],
        ))
        # Fast (transition)
        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.033,
            tdat={"distance": 1.3, "speed": 12.0, "angle": -7.0, "magnitude": 4000},
            pdat=[],
        ))
        result = tracker.get_club_angle()
        assert result is not None
        assert result.detection_class == "club"

    def test_ball_and_club_independent(self):
        """Ball and club detection work independently on same buffer."""
        tracker = self._make_tracker()
        now = time.time()
        # Club approach (slow → fast at close range)
        for i in range(5):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 1.5, "speed": 3.0, "angle": -8.0, "magnitude": 3500}],
            ))
        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.2, tdat=None,
            pdat=[{"distance": 1.3, "speed": 12.0, "angle": -6.0, "magnitude": 4000}],
        ))
        # Gap
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.3 + i * 0.033, tdat=None, pdat=[]))
        # Ball burst (far, fast)
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.7 + i * 0.033, tdat=None,
                pdat=[{"distance": 4.2, "speed": 25.0, "angle": 18.0, "magnitude": 2500}],
            ))

        ball = tracker.get_angle_for_shot()
        club = tracker.get_club_angle()

        assert ball is not None
        assert ball.detection_class == "ball"
        assert ball.vertical_deg > 0

        assert club is not None
        assert club.detection_class == "club"
        assert club.vertical_deg < 0


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
                tracker._add_frame(KLD7Frame(
                    timestamp=f["timestamp"],
                    tdat=f.get("tdat"),
                    pdat=f.get("pdat", []),
                ))
        assert tracker.get_angle_for_shot() is None

    def test_quiet_period_produces_no_results(self):
        """A quiet period in real data should produce no results."""
        raw_frames = self._load_frames()
        tracker = self._make_tracker()
        t0 = raw_frames[0]["timestamp"]
        for f in raw_frames:
            t = f["timestamp"] - t0
            if 19.0 <= t <= 24.0:
                tracker._add_frame(KLD7Frame(
                    timestamp=f["timestamp"],
                    tdat=f.get("tdat"),
                    pdat=f.get("pdat", []),
                ))
        assert tracker.get_angle_for_shot() is None


class TestKLD7Integration:
    """Integration tests for K-LD7 angle data flowing through to Shot."""

    def test_angle_attaches_to_shot_vertical(self):
        shot = Shot(
            ball_speed_mph=150.0, timestamp=datetime.now(),
            launch_angle_vertical=12.5, launch_angle_confidence=0.8, angle_source="radar",
        )
        result = shot_to_dict(shot)
        assert result["launch_angle_vertical"] == 12.5
        assert result["angle_source"] == "radar"

    def test_angle_attaches_to_shot_horizontal(self):
        shot = Shot(
            ball_speed_mph=150.0, timestamp=datetime.now(),
            launch_angle_horizontal=-3.5, launch_angle_confidence=0.7, angle_source="radar",
        )
        result = shot_to_dict(shot)
        assert result["launch_angle_horizontal"] == -3.5

    def test_carry_adjusts_for_vertical_angle(self):
        shot_no_angle = Shot(ball_speed_mph=150.0, timestamp=datetime.now())
        shot_with_angle = Shot(
            ball_speed_mph=150.0, timestamp=datetime.now(),
            launch_angle_vertical=15.0, launch_angle_confidence=0.8, angle_source="radar",
        )
        assert shot_no_angle.estimated_carry_yards != shot_with_angle.estimated_carry_yards

    def test_club_angle_in_shot_dict(self):
        shot = Shot(
            ball_speed_mph=150.0, timestamp=datetime.now(),
            club_angle_deg=-5.5,
        )
        result = shot_to_dict(shot)
        assert result["club_angle_deg"] == -5.5

    def test_full_tracker_to_shot_flow(self):
        """Full flow: ball burst in buffer → get_angle → attach to Shot."""
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = "vertical"
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()

        now = time.time()
        # Ball burst at far range
        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 4.3, "speed": 20.0, "angle": 18.0, "magnitude": 2500}],
            ))

        angle = tracker.get_angle_for_shot(shot_timestamp=now + 0.1)
        assert angle is not None
        assert angle.detection_class == "ball"

        shot = Shot(ball_speed_mph=150.0, timestamp=datetime.now())
        shot.launch_angle_vertical = angle.vertical_deg
        shot.launch_angle_confidence = angle.confidence
        shot.angle_source = "radar"

        result = shot_to_dict(shot)
        assert result["launch_angle_vertical"] == 18.0
        assert result["angle_source"] == "radar"

    def test_get_angle_after_reset_returns_none(self):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = "vertical"
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()

        now = time.time()
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[{"distance": 4.3, "speed": 20.0, "angle": 15.0, "magnitude": 2500}],
            ))
        assert tracker.get_angle_for_shot() is not None

        tracker.reset()
        assert tracker.get_angle_for_shot() is None
