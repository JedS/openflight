"""Tests for K-LD7 angle radar integration."""

import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from openflight.kld7.types import KLD7Angle, KLD7Frame
from openflight.kld7.tracker import KLD7Tracker
from openflight.launch_monitor import Shot, ClubType
from openflight.server import shot_to_dict

# Path to real captured K-LD7 data (golf swings + body movement)
CAPTURE_PATH = Path(__file__).parent.parent / "session_logs" / "kld7_capture_20260329_095614.pkl"
LABELED_CAPTURE_PATH = Path(__file__).parent.parent / "session_logs" / "kld7_capture_20260402_135117-wedge.pkl"


class TestKLD7Types:
    """Tests for K-LD7 data types."""

    def test_kld7_frame_defaults(self):
        frame = KLD7Frame(timestamp=1000.0)
        assert frame.timestamp == 1000.0
        assert frame.tdat is None
        assert frame.pdat == []

    def test_kld7_frame_radc_field(self):
        frame = KLD7Frame(timestamp=1000.0)
        assert frame.radc is None
        frame_with_radc = KLD7Frame(timestamp=1000.0, radc=b"\x00" * 3072)
        assert len(frame_with_radc.radc) == 3072

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

    def test_angle_offset_applied_to_ball(self):
        """Angle offset should shift the reported ball angle."""
        tracker = self._make_tracker()
        tracker.angle_offset_deg = 12.0
        now = time.time()
        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033, tdat=None,
                pdat=[self._ball_target(angle=5.0)],
            ))
        result = tracker.get_angle_for_shot()
        assert result is not None
        assert result.vertical_deg == pytest.approx(17.0, abs=0.2)

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

    def test_prefers_coherent_track_within_noisy_far_burst(self):
        """Mixed far targets should resolve to one coherent launch path."""
        tracker = self._make_tracker()
        now = time.time()

        tracker._add_frame(KLD7Frame(
            timestamp=now,
            tdat=None,
            pdat=[
                self._ball_target(angle=7.0, dist=4.2, speed=20.0, mag=2600),
                self._ball_target(angle=60.0, dist=4.4, speed=22.0, mag=2300),
            ],
        ))
        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.033,
            tdat=None,
            pdat=[
                self._ball_target(angle=8.0, dist=4.3, speed=18.0, mag=2550),
                self._ball_target(angle=-62.0, dist=4.5, speed=19.0, mag=2250),
            ],
        ))

        result = tracker.get_angle_for_shot()

        assert result is not None
        assert 6.5 < result.vertical_deg < 8.5

    def test_coherent_track_uses_all_frames_in_3_frame_burst(self):
        """DP backtracking must traverse the full burst, not stop at best-scoring frame.

        When the last frame only has a target with a large angle jump from the
        coherent path, the cumulative score at the last frame can be lower than
        the score at the middle frame.  The DP must still select the best path
        ending at the last frame so all frames are represented.
        """
        tracker = self._make_tracker()
        now = time.time()

        # Frames 0-1: strong coherent path at ~10°
        tracker._add_frame(KLD7Frame(
            timestamp=now,
            tdat=None,
            pdat=[
                self._ball_target(angle=10.0, dist=4.2, speed=20.0, mag=5000),
            ],
        ))
        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.033,
            tdat=None,
            pdat=[
                self._ball_target(angle=11.0, dist=4.25, speed=19.0, mag=5000),
            ],
        ))
        # Frame 2: only a weak target at a very different angle — big
        # continuity penalty makes its cumulative score lower than frame 1's.
        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.066,
            tdat=None,
            pdat=[
                self._ball_target(angle=70.0, dist=4.8, speed=18.0, mag=600),
            ],
        ))

        result = tracker.get_angle_for_shot()

        assert result is not None
        # The path must include all 3 frames
        assert result.num_frames == 3


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


class TestProbableShotPairing:
    """Tests for offline club-to-ball pairing on buffered K-LD7 data."""

    def _make_tracker(self, orientation="vertical"):
        tracker = KLD7Tracker.__new__(KLD7Tracker)
        tracker.orientation = orientation
        tracker.buffer_seconds = 2.0
        tracker.max_buffer_frames = 70
        tracker._init_ring_buffer()
        return tracker

    def test_pairs_club_transition_to_following_ball_burst(self):
        """A close-range club event should pair with the following far-range burst."""
        tracker = self._make_tracker()
        now = time.time()

        for i in range(5):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                tdat=None,
                pdat=[{"distance": 1.5, "speed": 3.0, "angle": -8.0, "magnitude": 3500}],
            ))

        tracker._add_frame(KLD7Frame(
            timestamp=now + 0.2,
            tdat=None,
            pdat=[{"distance": 1.3, "speed": 12.0, "angle": -6.0, "magnitude": 4000}],
        ))

        for i in range(2):
            tracker._add_frame(KLD7Frame(
                timestamp=now + 0.4 + i * 0.033,
                tdat=None,
                pdat=[{"distance": 4.2, "speed": 25.0, "angle": 18.0, "magnitude": 2500}],
            ))

        probable_shots = tracker.find_probable_shots()

        assert len(probable_shots) == 1
        assert probable_shots[0]["dt_ms"] == pytest.approx(200.0, abs=40.0)
        assert probable_shots[0]["ball_angle_deg"] == pytest.approx(18.0, abs=1.0)

    def test_suppresses_double_counted_shots_within_min_gap(self):
        """Two club transitions from the same swing should not produce two shots.

        In real captures, a single swing often produces multiple close-range
        speed transitions (club approach + follow-through rebound). Frames
        are in temporal order with slow frames between transitions, so both
        trigger as valid club candidates. The pairing logic must suppress
        the second one.
        """
        tracker = self._make_tracker()
        tracker.max_buffer_frames = 500
        tracker._init_ring_buffer()
        now = time.time()

        def _add_swing(t_start):
            """Add frames in temporal order: body → club → ball → slow → second club → second burst."""
            # Slow body movement
            for i in range(5):
                tracker._add_frame(KLD7Frame(
                    timestamp=t_start + i * 0.033,
                    tdat=None,
                    pdat=[{"distance": 1.5, "speed": 3.0, "angle": -8.0, "magnitude": 3500}],
                ))
            # First club transition (real)
            tracker._add_frame(KLD7Frame(
                timestamp=t_start + 0.2,
                tdat=None,
                pdat=[{"distance": 1.3, "speed": 12.0, "angle": -6.0, "magnitude": 4000}],
            ))
            # Ball burst
            for i in range(2):
                tracker._add_frame(KLD7Frame(
                    timestamp=t_start + 0.4 + i * 0.033,
                    tdat=None,
                    pdat=[{"distance": 4.2, "speed": 25.0, "angle": 18.0, "magnitude": 2500}],
                ))
            # Slow frames (follow-through settling)
            for i in range(5):
                tracker._add_frame(KLD7Frame(
                    timestamp=t_start + 0.6 + i * 0.033,
                    tdat=None,
                    pdat=[{"distance": 1.6, "speed": 2.0, "angle": 5.0, "magnitude": 3000}],
                ))
            # Second club transition (follow-through / rebound — false)
            tracker._add_frame(KLD7Frame(
                timestamp=t_start + 1.0,
                tdat=None,
                pdat=[{"distance": 1.4, "speed": 15.0, "angle": 10.0, "magnitude": 3800}],
            ))
            # A second far burst (net reflection / noise)
            tracker._add_frame(KLD7Frame(
                timestamp=t_start + 1.2,
                tdat=None,
                pdat=[{"distance": 4.5, "speed": 20.0, "angle": 55.0, "magnitude": 2100}],
            ))

        _add_swing(now)
        _add_swing(now + 10.0)  # second swing well-separated

        probable_shots = tracker.find_probable_shots()

        # Should find 2 shots (one per swing), not 4
        assert len(probable_shots) == 2
        assert probable_shots[1]["club_time"] - probable_shots[0]["club_time"] > 5.0


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

    def _load_labeled_capture(self):
        if not LABELED_CAPTURE_PATH.exists():
            pytest.skip(f"Capture file not found: {LABELED_CAPTURE_PATH}")
        with open(LABELED_CAPTURE_PATH, "rb") as f:
            return pickle.load(f)

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

    def test_probable_shots_match_expected_count_on_labeled_capture(self):
        """Labeled wedge capture should produce the expected number of probable shots."""
        capture = self._load_labeled_capture()
        tracker = self._make_tracker()
        tracker.max_buffer_frames = max(len(capture["frames"]), 70)
        tracker._init_ring_buffer()

        for f in capture["frames"]:
            tracker._add_frame(KLD7Frame(
                timestamp=f["timestamp"],
                tdat=f.get("tdat"),
                pdat=f.get("pdat", []),
            ))

        probable_shots = tracker.find_probable_shots()

        assert capture["metadata"]["expected_shots"] == 5
        assert len(probable_shots) == capture["metadata"]["expected_shots"]
        assert all(80 <= shot["dt_ms"] <= 350 for shot in probable_shots)


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
        from openflight.kld7.radc import ANTENNA_SPACING_M, WAVELENGTH_M, SAMPLES_PER_CHANNEL

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
        f2a_i = (amplitude * np.cos(phase_per_sample * t + steering_phase) + 32768).astype(np.uint16)
        f2a_q = (amplitude * np.sin(phase_per_sample * t + steering_phase) + 32768).astype(np.uint16)

        # F1B channel: zeros (not used for angle)
        zeros = np.full(n, 32768, dtype=np.uint16)

        payload = b""
        for ch in [f1a_i, f1a_q, f2a_i, f2a_q, zeros, zeros]:
            payload += ch.astype(np.uint16).tobytes()
        return payload

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
        target_angle = 12.0

        radc = self._make_radc_payload_with_tone(aliased_kmh, angle_deg=target_angle)
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

    def test_falls_back_to_pdat_without_radc(self):
        """When no RADC frames are present, should fall back to PDAT-based detection."""
        tracker = self._make_tracker()
        now = time.time()

        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                pdat=[{"distance": 4.2, "speed": 25.0, "angle": 15.0, "magnitude": 2500}],
            ))

        result = tracker.get_angle_for_shot(ball_speed_mph=72.0)
        assert result is not None
        assert result.detection_class == "ball"
        assert 14.0 < result.vertical_deg < 16.0

    def test_falls_back_to_pdat_without_ball_speed(self):
        """When ball_speed_mph is None, should use PDAT-based detection."""
        tracker = self._make_tracker()
        now = time.time()

        for i in range(3):
            tracker._add_frame(KLD7Frame(
                timestamp=now + i * 0.033,
                pdat=[{"distance": 4.2, "speed": 25.0, "angle": 15.0, "magnitude": 2500}],
            ))

        result = tracker.get_angle_for_shot(ball_speed_mph=None)
        assert result is not None

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

        radc = self._make_radc_payload_with_tone(aliased_kmh, angle_deg=10.0)
        quiet = self._make_quiet_radc_payload()

        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + i * 0.056, radc=quiet))
        tracker._add_frame(KLD7Frame(timestamp=now + 0.56, radc=radc))
        for i in range(10):
            tracker._add_frame(KLD7Frame(timestamp=now + 0.62 + i * 0.056, radc=quiet))

        result = tracker.get_angle_for_shot(ball_speed_mph=ball_speed_mph)
        assert result is not None
        assert result.vertical_deg == pytest.approx(15.0, abs=3.0)
