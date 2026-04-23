"""
Golf Launch Monitor data types and carry estimation.

Provides Shot, ClubType, and carry distance estimation used by
RollingBufferMonitor and the Flask server.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from .ops243 import SpeedReading

# Spin confidence threshold for "high" quality — used across modules.
# Measured spin is trusted for physics simulation only above this level.
SPIN_CONFIDENCE_HIGH = 0.7


class ClubType(Enum):
    """Golf club types for distance estimation."""

    DRIVER = "driver"
    WOOD_3 = "3-wood"
    WOOD_5 = "5-wood"
    WOOD_7 = "7-wood"
    HYBRID_3 = "3-hybrid"
    HYBRID_5 = "5-hybrid"
    HYBRID_7 = "7-hybrid"
    HYBRID_9 = "9-hybrid"
    IRON_2 = "2-iron"
    IRON_3 = "3-iron"
    IRON_4 = "4-iron"
    IRON_5 = "5-iron"
    IRON_6 = "6-iron"
    IRON_7 = "7-iron"
    IRON_8 = "8-iron"
    IRON_9 = "9-iron"
    PW = "pw"
    GW = "gw"
    SW = "sw"
    LW = "lw"
    UNKNOWN = "unknown"


# Optimal launch angles by club (from TrackMan data)
_OPTIMAL_LAUNCH = {
    ClubType.DRIVER: 11.0,
    ClubType.WOOD_3: 12.5,
    ClubType.WOOD_5: 14.0,
    ClubType.WOOD_7: 15.5,
    ClubType.HYBRID_3: 13.5,
    ClubType.HYBRID_5: 15.0,
    ClubType.HYBRID_7: 16.5,
    ClubType.HYBRID_9: 18.0,
    ClubType.IRON_2: 13.0,
    ClubType.IRON_3: 14.5,
    ClubType.IRON_4: 16.0,
    ClubType.IRON_5: 17.5,
    ClubType.IRON_6: 19.0,
    ClubType.IRON_7: 20.5,
    ClubType.IRON_8: 23.0,
    ClubType.IRON_9: 25.5,
    ClubType.PW: 28.0,
    ClubType.GW: 30.0,
    ClubType.SW: 32.0,
    ClubType.LW: 35.0,
    ClubType.UNKNOWN: 18.0,
}


def estimate_carry_distance(ball_speed_mph: float, club: ClubType = ClubType.DRIVER) -> float:
    """
    Estimate carry distance from ball speed using TrackMan-derived data.

    This uses interpolation from real-world data assuming optimal launch
    conditions (10-14° launch angle, appropriate spin for ball speed).

    Data sources:
    - TrackMan PGA Tour averages
    - pitchmarks.com ball speed to distance tables

    Args:
        ball_speed_mph: Ball speed in mph
        club: Club type (affects the model used)

    Returns:
        Estimated carry distance in yards

    Note:
        Without launch angle and spin rate, this is an approximation.
        Actual distance can vary ±10-15% based on:
        - Launch angle (optimal: 10-14° for driver)
        - Spin rate (optimal: 2000-3000 rpm for driver)
        - Weather conditions
        - Altitude
    """
    # Driver ball speed to carry distance lookup table
    # Based on TrackMan data assuming optimal launch conditions
    # Format: (ball_speed_mph, carry_yards_low, carry_yards_high)
    DRIVER_TABLE = [
        (100, 130, 142),
        (110, 157, 170),
        (120, 183, 197),
        (130, 207, 223),
        (140, 231, 249),
        (150, 254, 275),
        (160, 276, 301),
        (167, 275, 285),  # PGA Tour average
        (170, 298, 325),
        (180, 320, 349),
        (190, 342, 372),
        (200, 360, 389),
        (210, 383, 408),
    ]
    # Some source rows are mixed (e.g. PGA average vs optimal carry) and can
    # create local dips in the midpoint curve. Clamp midpoints to a
    # non-decreasing sequence so carry does not decrease as ball speed rises.
    monotonic_driver_curve = []
    max_carry_so_far = 0.0
    for speed, carry_min, carry_max in DRIVER_TABLE:
        midpoint = (carry_min + carry_max) / 2
        max_carry_so_far = max(max_carry_so_far, midpoint)
        monotonic_driver_curve.append((speed, max_carry_so_far))

    # No club factor applied — ball speed already reflects the club's smash
    # factor. Club-specific carry differences come from launch angle and spin
    # adjustments applied downstream.

    # Interpolate from driver table
    if ball_speed_mph <= monotonic_driver_curve[0][0]:
        # Below minimum - extrapolate linearly
        ratio = ball_speed_mph / monotonic_driver_curve[0][0]
        base_carry = monotonic_driver_curve[0][1]
        carry = base_carry * ratio
    elif ball_speed_mph >= monotonic_driver_curve[-1][0]:
        # Above maximum - extrapolate conservatively
        # Use ~1.8 yards per mph above 210 mph
        base_carry = monotonic_driver_curve[-1][1]
        carry = base_carry + (ball_speed_mph - monotonic_driver_curve[-1][0]) * 1.8
    else:
        # Interpolate between table entries
        for i in range(len(monotonic_driver_curve) - 1):
            if monotonic_driver_curve[i][0] <= ball_speed_mph < monotonic_driver_curve[i + 1][0]:
                # Linear interpolation
                speed_low, carry_low = monotonic_driver_curve[i]
                speed_high, carry_high = monotonic_driver_curve[i + 1]

                # Interpolate
                t = (ball_speed_mph - speed_low) / (speed_high - speed_low)
                carry = carry_low + t * (carry_high - carry_low)
                break
        else:
            # Fallback (shouldn't reach here)
            carry = ball_speed_mph * 1.65

    return carry


def adjust_carry_for_launch_angle(
    base_carry: float,
    launch_angle: float,
    club: ClubType = ClubType.DRIVER,
    confidence: float = 1.0,
) -> float:
    """
    Adjust carry distance based on launch angle deviation from optimal.

    Deviation from optimal costs carry:
    - Too low: -2.0 yards per degree (ball doesn't get enough height)
    - Too high: -1.5 yards per degree (ball balloons, less severe)
    - Penalty is scaled by confidence and capped at 10% of base carry.

    Args:
        base_carry: Base carry distance in yards (from estimate_carry_distance)
        launch_angle: Measured vertical launch angle in degrees
        club: Club type (determines optimal launch angle)
        confidence: Confidence in the launch angle measurement (0-1)

    Returns:
        Adjusted carry distance in yards
    """
    optimal = _OPTIMAL_LAUNCH.get(club, 18.0)
    angle_delta = launch_angle - optimal

    if angle_delta < 0:
        raw_penalty = abs(angle_delta) * 2.0
    else:
        raw_penalty = angle_delta * 1.5

    penalty = raw_penalty * confidence
    max_penalty = base_carry * 0.10
    penalty = min(penalty, max_penalty)

    return base_carry - penalty


@dataclass
class Shot:
    """
    Represents a detected golf shot with ball and club data.

    Attributes:
        ball_speed_mph: Peak ball speed detected (mph)
        club_speed_mph: Peak club head speed detected (mph), if available
        smash_factor: Ratio of ball speed to club speed (typically 1.4-1.5 for driver)
        timestamp: When the shot was detected
        impact_timestamp: Epoch timestamp of the peak ball reading from OPS243
        peak_magnitude: Signal strength of strongest reading
        readings: All raw speed readings for this shot
        club: Club type for distance estimation
        launch_angle_vertical: Vertical launch angle in degrees (from camera)
        launch_angle_horizontal: Horizontal launch angle in degrees (from camera)
        launch_angle_confidence: Confidence in launch angle measurement (0-1)
        spin_rpm: Spin rate in RPM (from rolling buffer mode)
        spin_confidence: Confidence in spin measurement (0-1)
        carry_spin_adjusted: Carry distance adjusted for spin (yards)
        mode: Shot source — "streaming", "rolling-buffer", or "mock"
        readings_data: Serialized readings for session logging
    """

    ball_speed_mph: float
    timestamp: datetime
    impact_timestamp: Optional[float] = None
    club_speed_mph: Optional[float] = None
    peak_magnitude: Optional[float] = None
    readings: List[SpeedReading] = field(default_factory=list)
    club: ClubType = ClubType.DRIVER
    launch_angle_vertical: Optional[float] = None
    launch_angle_horizontal: Optional[float] = None
    launch_angle_confidence: Optional[float] = None
    spin_rpm: Optional[float] = None
    spin_confidence: Optional[float] = None
    carry_spin_adjusted: Optional[float] = None
    mode: str = "rolling-buffer"
    readings_data: Optional[list] = None
    angle_source: Optional[str] = None  # "radar", "camera", "estimated", or None
    club_angle_deg: Optional[float] = None  # Club angle of attack from K-LD7 (vertical)
    club_path_deg: Optional[float] = None  # Club path from K-LD7 (horizontal)
    spin_axis_deg: Optional[float] = None  # Spin axis tilt: 0=backspin, +right(fade), -left(draw)

    @property
    def ball_speed_ms(self) -> float:
        """Ball speed in meters per second."""
        return self.ball_speed_mph * 0.44704

    @property
    def club_speed_ms(self) -> Optional[float]:
        """Club speed in meters per second."""
        if self.club_speed_mph is None:
            return None
        return self.club_speed_mph * 0.44704

    @property
    def smash_factor(self) -> Optional[float]:
        """
        Smash factor: ratio of ball speed to club speed.

        Indicates quality of contact:
        - Driver: 1.44-1.50 (optimal)
        - Irons: 1.30-1.38
        - Wedges: 1.20-1.25

        Returns:
            Smash factor or None if club speed not available
        """
        if self.club_speed_mph is None or self.club_speed_mph == 0:
            return None
        return self.ball_speed_mph / self.club_speed_mph

    @property
    def estimated_carry_yards(self) -> float:
        """Estimated carry distance based on ball speed, club type, and launch angle."""
        base = estimate_carry_distance(self.ball_speed_mph, self.club)
        if self.launch_angle_vertical is not None:
            return adjust_carry_for_launch_angle(
                base,
                self.launch_angle_vertical,
                self.club,
                self.launch_angle_confidence or 0.2,
            )
        return base

    @property
    def estimated_carry_range(self) -> tuple:
        """
        Return (min, max) carry distance estimate to show uncertainty.

        Returns:
            Tuple of (low_estimate, high_estimate) in yards
        """
        base = self.estimated_carry_yards
        # ±10% uncertainty without launch angle/spin data
        # Reduce to ±5% if we have launch angle data
        if self.has_launch_angle:
            return (base * 0.95, base * 1.05)
        return (base * 0.90, base * 1.10)

    @property
    def has_launch_angle(self) -> bool:
        """Check if launch angle data is available for this shot."""
        return self.launch_angle_vertical is not None

    @property
    def has_spin(self) -> bool:
        """Check if spin data is available for this shot."""
        return self.spin_rpm is not None

    @property
    def spin_quality(self) -> Optional[str]:
        """
        Get spin measurement quality as a string.

        Returns:
            "high", "medium", "low", or None if no spin data
        """
        if self.spin_confidence is None:
            return None
        if self.spin_confidence >= SPIN_CONFIDENCE_HIGH:
            return "high"
        if self.spin_confidence >= 0.4:
            return "medium"
        return "low"


