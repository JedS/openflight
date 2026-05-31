"""K-LD7 raw ADC (RADC) signal processing for the openflight package.

Core functions for FFT-based velocity detection and phase interferometry
angle extraction from K-LD7 24 GHz radar raw ADC data.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)

RADC_PAYLOAD_BYTES = 3072
SAMPLES_PER_CHANNEL = 256

DC_MASK_BINS = 8  # Zero out bins near DC to suppress residual leakage

# Half-width (in FFT bins) of the neighborhood around the per-frame peak
# bin used for the magnitude²-weighted centroid angle. A real ball peak
# spreads across a handful of bins (Hann-window leakage + intra-frame
# Doppler smear); 16 bins on either side comfortably covers the peak
# shape without picking up noise elsewhere in the ball band.
CENTROID_SEARCH_BINS = 16

# Minimum linear SNR for accepting the OPS-expected-bin local peak over the
# strongest in-band peak. This must be higher than the per-frame detection
# floor because Hann-window side-lobes from a strong off-anchor target can
# exceed 2x the full-spectrum median and otherwise masquerade as a valid
# anchored ball return.
OPS_ANCHORED_PEAK_MIN_SNR = 5.0

# Broad default ball band used when no OPS243 ball speed is available.
# At RSPI=3 the K-LD7's unambiguous velocity span is +/-100 km/h, so
# typical golf ball speeds alias into this negative-velocity window.
DEFAULT_BALL_ALIASED_MIN_KMH = -39.0
DEFAULT_BALL_ALIASED_MAX_KMH = -7.0

# K-LD7 antenna parameters (24 GHz)
WAVELENGTH_M = 3e8 / 24.125e9  # ~12.43 mm
ANTENNA_SPACING_M = 8.0e-3  # ~0.64λ, calibrated against PDAT reference data


def parse_radc_payload(payload: bytes) -> dict[str, np.ndarray]:
    """Parse a 3072-byte RADC payload into six uint16 channel arrays.

    Layout (each segment = 256 × uint16 = 512 bytes):
        [0:512]     F1 Freq A — I channel
        [512:1024]  F1 Freq A — Q channel
        [1024:1536] F2 Freq A — I channel
        [1536:2048] F2 Freq A — Q channel
        [2048:2560] F1 Freq B — I channel
        [2560:3072] F1 Freq B — Q channel
    """
    if len(payload) != RADC_PAYLOAD_BYTES:
        raise ValueError(f"RADC payload must be {RADC_PAYLOAD_BYTES} bytes, got {len(payload)}")
    seg = 512  # bytes per segment
    return {
        "f1a_i": np.frombuffer(payload[0:seg], dtype=np.uint16).copy(),
        "f1a_q": np.frombuffer(payload[seg : 2 * seg], dtype=np.uint16).copy(),
        "f2a_i": np.frombuffer(payload[2 * seg : 3 * seg], dtype=np.uint16).copy(),
        "f2a_q": np.frombuffer(payload[3 * seg : 4 * seg], dtype=np.uint16).copy(),
        "f1b_i": np.frombuffer(payload[4 * seg : 5 * seg], dtype=np.uint16).copy(),
        "f1b_q": np.frombuffer(payload[5 * seg : 6 * seg], dtype=np.uint16).copy(),
    }


def to_complex_iq(i_channel: np.ndarray, q_channel: np.ndarray) -> np.ndarray:
    """Convert uint16 I/Q arrays to complex float, removing DC offset.

    Uses per-channel mean removal instead of a fixed midpoint, since the
    K-LD7 ADC bias varies across channels and units.
    """
    i_float = i_channel.astype(np.float64) - np.mean(i_channel.astype(np.float64))
    q_float = q_channel.astype(np.float64) - np.mean(q_channel.astype(np.float64))
    return i_float + 1j * q_float


def compute_spectrum(
    iq: np.ndarray, fft_size: int = 2048, dc_mask_bins: int = DC_MASK_BINS
) -> np.ndarray:
    """Compute magnitude spectrum from complex I/Q with Hann window and zero-padding.

    Args:
        iq: Complex I/Q array (256 samples from RADC)
        fft_size: FFT length (zero-padded if > len(iq))
        dc_mask_bins: Number of bins around DC to zero out (both ends)

    Returns:
        Magnitude spectrum (linear scale), length = fft_size
    """
    windowed = iq * np.hanning(len(iq))
    padded = np.zeros(fft_size, dtype=np.complex128)
    padded[: len(windowed)] = windowed
    fft_result = np.fft.fft(padded)
    magnitude = np.abs(fft_result)
    # Mask DC leakage at both ends of the spectrum
    if dc_mask_bins > 0:
        magnitude[:dc_mask_bins] = 0.0
        magnitude[-dc_mask_bins:] = 0.0
    return magnitude


def compute_fft_complex(
    iq: np.ndarray, fft_size: int = 2048, dc_mask_bins: int = DC_MASK_BINS
) -> np.ndarray:
    """Compute complex FFT output (not magnitude) for phase-based processing."""
    windowed = iq * np.hanning(len(iq))
    padded = np.zeros(fft_size, dtype=np.complex128)
    padded[: len(windowed)] = windowed
    result = np.fft.fft(padded)
    if dc_mask_bins > 0:
        result[:dc_mask_bins] = 0.0
        result[-dc_mask_bins:] = 0.0
    return result


@dataclass(frozen=True)
class CFARDetection:
    bin_index: int
    magnitude: float
    snr_db: float


@dataclass(frozen=True)
class RADCChannelStats:
    """Per-channel raw ADC health metrics."""

    mean: float
    std: float
    min_code: int
    max_code: int
    dynamic_range: int
    clipped_low_frac: float
    clipped_high_frac: float


@dataclass(frozen=True)
class RADCFrameDiagnostics:
    """Frame-level diagnostics for raw K-LD7 ADC data.

    This intentionally mirrors the live angle path's target-band and
    centroid logic, but exposes the intermediate signal-quality checks
    needed to diagnose bad horizontal/vertical launch angle estimates.
    """

    frame_index: int
    timestamp: float | None
    has_radc: bool
    valid_payload: bool
    reason: str | None
    target_bands: tuple[tuple[int, int], ...]
    expected_bin: int | None
    peak_bin: int | None
    peak_velocity_kmh: float | None
    peak_ball_speed_mph: float | None
    speed_error_mph: float | None
    peak_magnitude: float
    noise_floor: float
    snr_linear: float
    snr_db: float
    bin_error: int | None
    angle_peak_deg: float | None
    angle_centroid_deg: float | None
    phase_coherence: float | None
    peak_width_bins: int
    channel_stats: dict[str, RADCChannelStats]
    iq_stats: dict[str, dict[str, float]]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly representation."""
        out = asdict(self)
        out["target_bands"] = [list(b) for b in self.target_bands]
        out["warnings"] = list(self.warnings)
        return out


def cfar_detect(
    spectrum: np.ndarray,
    guard_cells: int = 4,
    training_cells: int = 16,
    threshold_factor: float = 8.0,
) -> list[CFARDetection]:
    """Ordered-statistic CFAR detection on a magnitude spectrum.

    For each bin, estimates the noise level from surrounding training cells
    (excluding guard cells) and declares a detection if the bin exceeds
    threshold_factor × noise_estimate.

    Args:
        spectrum: Magnitude spectrum (1D array)
        guard_cells: Number of guard cells on each side of the cell under test
        training_cells: Number of training cells on each side (outside guard)
        threshold_factor: Detection threshold as multiple of noise estimate

    Returns:
        List of detections sorted by magnitude (descending)
    """
    n = len(spectrum)
    margin = guard_cells + training_cells
    detections = []

    for i in range(margin, n - margin):
        left_train = spectrum[i - margin : i - guard_cells]
        right_train = spectrum[i + guard_cells + 1 : i + margin + 1]
        training = np.concatenate([left_train, right_train])
        # Use median (OS-CFAR) for robustness against interfering targets
        noise_estimate = np.median(training)

        if noise_estimate <= 0:
            continue

        if spectrum[i] > threshold_factor * noise_estimate:
            snr_db = 10.0 * np.log10(spectrum[i] / noise_estimate)
            detections.append(
                CFARDetection(
                    bin_index=i,
                    magnitude=float(spectrum[i]),
                    snr_db=float(snr_db),
                )
            )

    detections.sort(key=lambda d: d.magnitude, reverse=True)
    return detections


def per_bin_angle_deg(
    f1a_fft: np.ndarray,
    f2a_fft: np.ndarray,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
) -> np.ndarray:
    """Compute angle of arrival at each FFT bin from phase difference between Rx channels.

    Uses the interferometric formula: θ = arcsin(Δφ * λ / (2π * d))
    where Δφ is the phase difference, λ is wavelength, d is antenna spacing.

    Returns array of angles in degrees, one per bin. Bins with no signal return 0.
    """
    cross = f1a_fft * np.conj(f2a_fft)
    phase_diff = np.angle(cross)
    # arcsin argument must be in [-1, 1]
    sin_theta = phase_diff * wavelength_m / (2.0 * np.pi * antenna_spacing_m)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_theta))


def bin_to_velocity_kmh(bin_index: int, fft_size: int, max_speed_kmh: float) -> float:
    """Convert FFT bin index to velocity in km/h.

    Bins 0..N/2 = 0..+max_speed (outbound).
    Bins N/2..N = -max_speed..0 (inbound, aliased).
    """
    if bin_index <= fft_size // 2:
        return bin_index * max_speed_kmh / (fft_size // 2)
    else:
        return (bin_index - fft_size) * max_speed_kmh / (fft_size // 2)


def _velocity_to_bin(
    velocity_kmh: float, fft_size: int = 2048, max_speed_kmh: float = 100.0
) -> int:
    """Convert velocity in km/h to FFT bin index."""
    if velocity_kmh >= 0:
        return int(velocity_kmh * (fft_size // 2) / max_speed_kmh)
    return int(fft_size + velocity_kmh * (fft_size // 2) / max_speed_kmh)


def aliased_velocity_from_ball_speed_mph(
    ball_speed_mph: float,
    max_speed_kmh: float = 100.0,
) -> float:
    """Map true ball speed to the K-LD7 aliased Doppler velocity in km/h."""
    ball_speed_kmh = ball_speed_mph * 1.609
    unambiguous_range = max_speed_kmh * 2.0
    aliased_kmh = ball_speed_kmh % unambiguous_range
    if aliased_kmh > max_speed_kmh:
        aliased_kmh -= unambiguous_range
    return float(aliased_kmh)


def expected_ball_bin_from_speed(
    ball_speed_mph: float,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
) -> int:
    """Return the FFT bin where an OPS243-measured ball speed should peak."""
    aliased_kmh = aliased_velocity_from_ball_speed_mph(ball_speed_mph, max_speed_kmh)
    return _velocity_to_bin(aliased_kmh, fft_size, max_speed_kmh)


def circular_bin_distance(a: int, b: int, fft_size: int = 2048) -> int:
    """Shortest distance between FFT bins, accounting for circular wrap."""
    distance = abs(int(a) - int(b))
    return int(min(distance, fft_size - distance))


def default_ball_bin_ranges(
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
) -> list[tuple[int, int]]:
    """Return the broad default ball velocity band used without OPS243 anchoring."""
    return [
        (
            _velocity_to_bin(DEFAULT_BALL_ALIASED_MIN_KMH, fft_size, max_speed_kmh),
            _velocity_to_bin(DEFAULT_BALL_ALIASED_MAX_KMH, fft_size, max_speed_kmh),
        )
    ]


def ball_bin_range_from_speed(
    ball_speed_mph: float,
    tolerance_mph: float = 10.0,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
) -> list[tuple[int, int]]:
    """Return FFT bin ranges for a specific ball speed.

    Uses the OPS243-measured ball speed to compute exactly where in the
    aliased spectrum the ball return should appear. Much more precise
    than the broad default range — eliminates club/multipath
    contamination.

    Returns a list of one or two ``(lo, hi)`` half-open ranges. Two
    ranges are returned when the search window straddles the FFT
    wraparound boundary at ±max_speed_kmh (i.e. the aliased ball
    velocity is within ``tolerance_mph`` of ±max_speed_kmh): in that
    case the band physically wraps around DC (bin 0), and the
    correct representation is two sub-ranges
    ``[N - tol_bins, N] ∪ [0, tol_bins]`` rather than a single
    spectrum-spanning range. Each sub-range satisfies ``0 ≤ lo < hi
    ≤ fft_size`` and is suitable for direct slicing
    (``spec[lo:hi]``).

    Args:
        ball_speed_mph: Measured ball speed from OPS243
        tolerance_mph: Search window around the expected velocity (±)
    """
    aliased_kmh = aliased_velocity_from_ball_speed_mph(ball_speed_mph, max_speed_kmh)
    tol_kmh = tolerance_mph * 1.609
    lo_vel = aliased_kmh - tol_kmh
    hi_vel = aliased_kmh + tol_kmh

    # The standard FFT layout places positive velocities in bins
    # [1, N/2] and negative velocities in bins [N/2, N-1], with
    # bin 0 = DC and bin N/2 = ±Nyquist. A *velocity* range that
    # crosses 0 km/h is therefore a *bin* range that wraps around
    # the array boundary at bin 0 / bin N. The single-range
    # representation [_velocity_to_bin(lo), _velocity_to_bin(hi)]
    # would describe the COMPLEMENT of the actual band in this case
    # (the entire spectrum *between* the two wrap-around ends).
    #
    # The window crosses 0 km/h when sign(lo_vel) != sign(hi_vel)
    # and 0 lies strictly inside (lo_vel, hi_vel).
    crosses_zero = lo_vel < 0 < hi_vel

    if not crosses_zero:
        lo_bin = _velocity_to_bin(lo_vel, fft_size, max_speed_kmh)
        hi_bin = _velocity_to_bin(hi_vel, fft_size, max_speed_kmh)
        if lo_bin > hi_bin:
            lo_bin, hi_bin = hi_bin, lo_bin
        return [(lo_bin, hi_bin)]

    # Wrap-around case. Split the window into:
    #   negative half: [lo_vel, 0) → bins (N/2, N) near the top
    #                                of the array
    #   positive half: (0, hi_vel] → bins (0, N/2) near the bottom
    # _velocity_to_bin(0) = 0 and _velocity_to_bin(-eps) = N - eps.
    #
    # We use a small bin offset rather than 0 / N to avoid emitting
    # a degenerate (0, 0) range when the window is exactly ±tol.
    neg_lo_bin = _velocity_to_bin(lo_vel, fft_size, max_speed_kmh)
    pos_hi_bin = _velocity_to_bin(hi_vel, fft_size, max_speed_kmh)

    ranges: list[tuple[int, int]] = []
    # Negative half: (neg_lo_bin, fft_size) — top of array.
    # _velocity_to_bin(lo_vel) returns a high bin (e.g. 1794 for
    # lo_vel=-25 km/h, fft_size=2048, max=100). The range is
    # [neg_lo_bin, fft_size).
    if neg_lo_bin < fft_size:
        ranges.append((neg_lo_bin, fft_size))
    # Positive half: [0, pos_hi_bin) — bottom of array.
    if pos_hi_bin > 0:
        ranges.append((0, pos_hi_bin))

    if not ranges:
        # Pathological: tolerance produced no valid bins. Fall back
        # to the broad default ball range.
        return default_ball_bin_ranges(fft_size, max_speed_kmh)
    return ranges


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_bin_ranges(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    fft_size: int,
) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for lo, hi in ranges:
        lo_i = max(0, min(fft_size, int(lo)))
        hi_i = max(0, min(fft_size, int(hi)))
        if lo_i < hi_i:
            out.append((lo_i, hi_i))
    return tuple(out)


def _target_bands_for_ball(
    ops243_ball_speed_mph: float | None,
    speed_tolerance_mph: float,
    fft_size: int,
    max_speed_kmh: float,
    target_bands: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    if target_bands is not None:
        return _normalize_bin_ranges(target_bands, fft_size)
    if ops243_ball_speed_mph is not None:
        return _normalize_bin_ranges(
            ball_bin_range_from_speed(
                ops243_ball_speed_mph,
                speed_tolerance_mph,
                fft_size,
                max_speed_kmh,
            ),
            fft_size,
        )
    return _normalize_bin_ranges(default_ball_bin_ranges(fft_size, max_speed_kmh), fft_size)


def _channel_stats(channel: np.ndarray) -> RADCChannelStats:
    arr = np.asarray(channel)
    if arr.size == 0:
        return RADCChannelStats(
            mean=0.0,
            std=0.0,
            min_code=0,
            max_code=0,
            dynamic_range=0,
            clipped_low_frac=0.0,
            clipped_high_frac=0.0,
        )

    arr_float = arr.astype(np.float64)
    min_code = int(np.min(arr_float))
    max_code = int(np.max(arr_float))
    return RADCChannelStats(
        mean=float(np.mean(arr_float)),
        std=float(np.std(arr_float)),
        min_code=min_code,
        max_code=max_code,
        dynamic_range=max_code - min_code,
        clipped_low_frac=float(np.mean(arr_float <= 0.0)),
        clipped_high_frac=float(np.mean(arr_float >= 65535.0)),
    )


def _iq_pair_stats(i_channel: np.ndarray, q_channel: np.ndarray) -> dict[str, float]:
    i_float = np.asarray(i_channel, dtype=np.float64)
    q_float = np.asarray(q_channel, dtype=np.float64)
    if i_float.size == 0 or q_float.size == 0:
        return {
            "i_std": 0.0,
            "q_std": 0.0,
            "q_to_i_std_ratio": 0.0,
            "iq_correlation": 0.0,
        }

    i_centered = i_float - float(np.mean(i_float))
    q_centered = q_float - float(np.mean(q_float))
    i_std = float(np.std(i_centered))
    q_std = float(np.std(q_centered))
    if i_std > 0:
        q_to_i = q_std / i_std
    elif q_std > 0:
        q_to_i = float("inf")
    else:
        q_to_i = 0.0

    denom = float(np.sqrt(np.sum(i_centered**2) * np.sum(q_centered**2)))
    corr = float(np.sum(i_centered * q_centered) / denom) if denom > 0 else 0.0
    return {
        "i_std": i_std,
        "q_std": q_std,
        "q_to_i_std_ratio": float(q_to_i),
        "iq_correlation": corr,
    }


def _find_peak_in_bands(
    spectrum: np.ndarray,
    bands: tuple[tuple[int, int], ...],
) -> tuple[int | None, float, tuple[int, int] | None]:
    peak_bin: int | None = None
    peak_val = 0.0
    peak_band: tuple[int, int] | None = None
    for sub_lo, sub_hi in bands:
        sub = spectrum[sub_lo:sub_hi]
        if sub.size == 0:
            continue
        sub_idx = int(np.argmax(sub))
        sub_max = float(sub[sub_idx])
        if sub_max > peak_val:
            peak_val = sub_max
            peak_bin = sub_lo + sub_idx
            peak_band = (sub_lo, sub_hi)
    return peak_bin, peak_val, peak_band


def _find_peak_near_expected_bin(
    spectrum: np.ndarray,
    bands: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    expected_bin: int,
    tolerance_bins: int,
    fft_size: int,
) -> tuple[int | None, float, tuple[int, int] | None]:
    """Find the strongest peak near the OPS-expected bin inside the target bands."""
    peak_bin: int | None = None
    peak_val = 0.0
    peak_band: tuple[int, int] | None = None
    tolerance = max(0, int(tolerance_bins))

    for sub_lo, sub_hi in bands:
        if sub_hi <= sub_lo:
            continue
        indices = np.arange(sub_lo, sub_hi, dtype=int)
        mask = np.array(
            [circular_bin_distance(idx, expected_bin, fft_size) <= tolerance for idx in indices]
        )
        if not mask.any():
            continue
        near_indices = indices[mask]
        near_values = spectrum[near_indices]
        if near_values.size == 0:
            continue
        local_idx = int(np.argmax(near_values))
        local_val = float(near_values[local_idx])
        if local_val > peak_val:
            peak_val = local_val
            peak_bin = int(near_indices[local_idx])
            peak_band = (sub_lo, sub_hi)

    return peak_bin, peak_val, peak_band


def _peak_neighborhood_indices(
    spectrum: np.ndarray,
    peak_bin: int,
    peak_val: float,
    peak_band: tuple[int, int] | None,
    half_width: int,
    floor_frac: float | None,
) -> np.ndarray:
    fft_size = len(spectrum)
    sub_lo, sub_hi = peak_band if peak_band is not None else (0, fft_size)
    lo_n = max(sub_lo, peak_bin - max(0, half_width))
    hi_n = min(sub_hi, peak_bin + max(0, half_width) + 1)
    if hi_n <= lo_n:
        return np.array([peak_bin], dtype=int)

    indices = np.arange(lo_n, hi_n, dtype=int)
    if floor_frac is not None:
        mask = spectrum[indices] >= peak_val * floor_frac
        indices = indices[mask]
    if indices.size == 0:
        return np.array([peak_bin], dtype=int)
    return indices


def _centroid_angle_for_peak(
    angles: np.ndarray,
    spectrum: np.ndarray,
    peak_bin: int,
    peak_val: float,
    peak_band: tuple[int, int] | None,
    centroid_floor_frac: float,
) -> tuple[float, int]:
    if centroid_floor_frac >= 1.0:
        return float(angles[peak_bin]), 1

    indices = _peak_neighborhood_indices(
        spectrum,
        peak_bin,
        peak_val,
        peak_band,
        CENTROID_SEARCH_BINS,
        centroid_floor_frac,
    )
    weights = spectrum[indices] ** 2
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(angles[peak_bin]), int(indices.size)
    return float(np.sum(angles[indices] * weights) / weight_sum), int(indices.size)


def _phase_coherence_for_peak(
    f1a_fft: np.ndarray,
    f2a_fft: np.ndarray,
    spectrum: np.ndarray,
    peak_bin: int,
    peak_val: float,
    peak_band: tuple[int, int] | None,
    coherence_bins: int,
) -> float | None:
    indices = _peak_neighborhood_indices(
        spectrum,
        peak_bin,
        peak_val,
        peak_band,
        coherence_bins,
        floor_frac=None,
    )
    weights = np.maximum(spectrum[indices], 0.0)
    if float(np.sum(weights)) <= 0:
        weights = np.ones_like(weights)

    f1 = f1a_fft[indices]
    f2 = f2a_fft[indices]
    cross = np.sum(weights * f1 * np.conj(f2))
    p1 = float(np.sum(weights * np.abs(f1) ** 2))
    p2 = float(np.sum(weights * np.abs(f2) ** 2))
    if p1 <= 0 or p2 <= 0:
        return None
    coherence = float(np.abs(cross) / np.sqrt(p1 * p2))
    return float(np.clip(coherence, 0.0, 1.0))


def radc_frame_diagnostics(
    frame: dict,
    frame_index: int = 0,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    ops243_ball_speed_mph: float | None = None,
    speed_tolerance_mph: float = 10.0,
    target_bands: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    orientation: str | None = None,
    centroid_floor_frac: float = 0.5,
    coherence_bins: int = 4,
    snr_warn_floor: float = 2.0,
    coherence_warn_floor: float = 0.65,
    ops_bin_warn_tol: int = 25,
    clipping_warn_frac: float = 0.005,
) -> RADCFrameDiagnostics:
    """Inspect one raw ADC frame and expose the live angle-path decisions.

    The returned diagnostics are designed for offline analysis of captured
    RADC frames. They answer: did the expected OPS-anchored bin contain the
    strongest target, was the peak coherent across the two receive channels,
    and did raw ADC health look suspicious before FFT processing?
    """
    timestamp = _optional_float(frame.get("timestamp"))
    bands = _target_bands_for_ball(
        ops243_ball_speed_mph,
        speed_tolerance_mph,
        fft_size,
        max_speed_kmh,
        target_bands,
    )
    expected_bin = (
        expected_ball_bin_from_speed(ops243_ball_speed_mph, fft_size, max_speed_kmh)
        if ops243_ball_speed_mph is not None
        else None
    )

    def empty(reason: str, has_radc: bool, warnings: tuple[str, ...]) -> RADCFrameDiagnostics:
        return RADCFrameDiagnostics(
            frame_index=frame_index,
            timestamp=timestamp,
            has_radc=has_radc,
            valid_payload=False,
            reason=reason,
            target_bands=bands,
            expected_bin=expected_bin,
            peak_bin=None,
            peak_velocity_kmh=None,
            peak_ball_speed_mph=None,
            speed_error_mph=None,
            peak_magnitude=0.0,
            noise_floor=0.0,
            snr_linear=0.0,
            snr_db=0.0,
            bin_error=None,
            angle_peak_deg=None,
            angle_centroid_deg=None,
            phase_coherence=None,
            peak_width_bins=0,
            channel_stats={},
            iq_stats={},
            warnings=warnings,
        )

    radc_raw = frame.get("radc")
    if radc_raw is None:
        return empty("missing_radc", has_radc=False, warnings=("missing_radc",))

    try:
        channels = parse_radc_payload(radc_raw) if isinstance(radc_raw, bytes) else radc_raw
    except ValueError:
        return empty("invalid_payload_size", has_radc=True, warnings=("invalid_payload",))
    if not isinstance(channels, dict):
        return empty("invalid_payload_type", has_radc=True, warnings=("invalid_payload",))

    required = ("f1a_i", "f1a_q", "f2a_i", "f2a_q")
    missing = [name for name in required if name not in channels]
    if missing:
        return empty(
            f"missing_channels:{','.join(missing)}",
            has_radc=True,
            warnings=("missing_channels",),
        )

    channel_stats = {
        name: _channel_stats(np.asarray(channel))
        for name, channel in channels.items()
        if isinstance(channel, np.ndarray) or hasattr(channel, "__array__")
    }
    iq_stats = {
        "f1a": _iq_pair_stats(channels["f1a_i"], channels["f1a_q"]),
        "f2a": _iq_pair_stats(channels["f2a_i"], channels["f2a_q"]),
    }

    f1a_iq = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
    f2a_iq = to_complex_iq(channels["f2a_i"], channels["f2a_q"])
    spectrum = compute_spectrum(f1a_iq, fft_size=fft_size)
    peak_bin, peak_val, peak_band = _find_peak_in_bands(spectrum, bands)

    positive = spectrum[spectrum > 0]
    noise_floor = float(np.median(positive)) if positive.size else 0.0
    snr_linear = peak_val / noise_floor if noise_floor > 0 else 0.0
    snr_db = float(10.0 * np.log10(snr_linear)) if snr_linear > 0 else 0.0

    warnings: list[str] = []
    for name, stats in channel_stats.items():
        clipped = max(stats.clipped_low_frac, stats.clipped_high_frac)
        if clipped > clipping_warn_frac:
            warnings.append(f"adc_clipping:{name}")
    for name, stats in iq_stats.items():
        ratio = stats["q_to_i_std_ratio"]
        if ratio > 0 and (ratio < 0.5 or ratio > 2.0):
            warnings.append(f"iq_imbalance:{name}")

    if peak_bin is None or peak_val <= 0:
        return RADCFrameDiagnostics(
            frame_index=frame_index,
            timestamp=timestamp,
            has_radc=True,
            valid_payload=True,
            reason="no_peak_in_target_band",
            target_bands=bands,
            expected_bin=expected_bin,
            peak_bin=None,
            peak_velocity_kmh=None,
            peak_ball_speed_mph=None,
            speed_error_mph=None,
            peak_magnitude=0.0,
            noise_floor=noise_floor,
            snr_linear=snr_linear,
            snr_db=snr_db,
            bin_error=None,
            angle_peak_deg=None,
            angle_centroid_deg=None,
            phase_coherence=None,
            peak_width_bins=0,
            channel_stats=channel_stats,
            iq_stats=iq_stats,
            warnings=tuple(warnings + ["no_peak_in_target_band"]),
        )

    f1a_fft = compute_fft_complex(f1a_iq, fft_size=fft_size)
    f2a_fft = compute_fft_complex(f2a_iq, fft_size=fft_size)
    angles = per_bin_angle_deg(f1a_fft, f2a_fft)
    angle_peak = float(angles[peak_bin])
    angle_centroid, peak_width = _centroid_angle_for_peak(
        angles,
        spectrum,
        peak_bin,
        peak_val,
        peak_band,
        centroid_floor_frac,
    )
    phase_coherence = _phase_coherence_for_peak(
        f1a_fft,
        f2a_fft,
        spectrum,
        peak_bin,
        peak_val,
        peak_band,
        coherence_bins,
    )
    bin_error = (
        circular_bin_distance(peak_bin, expected_bin, fft_size)
        if expected_bin is not None
        else None
    )
    peak_velocity_kmh = bin_to_velocity_kmh(peak_bin, fft_size, max_speed_kmh)
    peak_ball_speed_mph = (2.0 * max_speed_kmh + peak_velocity_kmh) / 1.609
    speed_error_mph = (
        peak_ball_speed_mph - ops243_ball_speed_mph if ops243_ball_speed_mph is not None else None
    )

    if snr_linear < snr_warn_floor:
        warnings.append("low_snr")
    if phase_coherence is not None and phase_coherence < coherence_warn_floor:
        warnings.append("low_phase_coherence")
    if bin_error is not None and bin_error > ops_bin_warn_tol:
        warnings.append("far_from_ops_bin")
    if orientation == "vertical" and (angle_centroid < 0.0 or angle_centroid > 45.0):
        warnings.append("outside_vertical_bounds")
    if orientation == "horizontal" and abs(angle_centroid) > 15.0:
        warnings.append("outside_horizontal_bounds")

    return RADCFrameDiagnostics(
        frame_index=frame_index,
        timestamp=timestamp,
        has_radc=True,
        valid_payload=True,
        reason=None,
        target_bands=bands,
        expected_bin=expected_bin,
        peak_bin=peak_bin,
        peak_velocity_kmh=peak_velocity_kmh,
        peak_ball_speed_mph=peak_ball_speed_mph,
        speed_error_mph=speed_error_mph,
        peak_magnitude=float(peak_val),
        noise_floor=noise_floor,
        snr_linear=float(snr_linear),
        snr_db=snr_db,
        bin_error=bin_error,
        angle_peak_deg=angle_peak,
        angle_centroid_deg=angle_centroid,
        phase_coherence=phase_coherence,
        peak_width_bins=peak_width,
        channel_stats=channel_stats,
        iq_stats=iq_stats,
        warnings=tuple(warnings),
    )


def _median_or_none(values: list[float]) -> float | None:
    clean = [v for v in values if np.isfinite(v)]
    return float(np.median(clean)) if clean else None


def summarize_radc_diagnostics(
    diagnostics: list[RADCFrameDiagnostics],
    top_bins: int = 8,
) -> dict[str, object]:
    """Summarize a sequence of raw ADC frame diagnostics."""
    valid = [d for d in diagnostics if d.valid_payload]
    peaks = [d for d in valid if d.peak_bin is not None]

    warnings_by_type: dict[str, int] = {}
    for d in diagnostics:
        for warning in d.warnings:
            warnings_by_type[warning] = warnings_by_type.get(warning, 0) + 1

    peak_bins: dict[int, int] = {}
    for d in peaks:
        assert d.peak_bin is not None
        peak_bins[d.peak_bin] = peak_bins.get(d.peak_bin, 0) + 1
    peak_bin_histogram = [
        {"bin": int(bin_index), "count": int(count)}
        for bin_index, count in sorted(
            peak_bins.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_bins]
    ]

    channel_clip_max: dict[str, dict[str, float]] = {}
    for d in valid:
        for name, stats in d.channel_stats.items():
            entry = channel_clip_max.setdefault(
                name,
                {"low": 0.0, "high": 0.0, "dynamic_range": 0.0},
            )
            entry["low"] = max(entry["low"], stats.clipped_low_frac)
            entry["high"] = max(entry["high"], stats.clipped_high_frac)
            entry["dynamic_range"] = max(entry["dynamic_range"], float(stats.dynamic_range))

    return {
        "frame_count": len(diagnostics),
        "radc_frame_count": sum(1 for d in diagnostics if d.has_radc),
        "valid_payload_count": len(valid),
        "peak_frame_count": len(peaks),
        "target_bands": [list(b) for b in diagnostics[0].target_bands] if diagnostics else [],
        "expected_bin": diagnostics[0].expected_bin if diagnostics else None,
        "median_snr_db": _median_or_none([d.snr_db for d in peaks]),
        "max_snr_db": float(max((d.snr_db for d in peaks), default=0.0)),
        "median_phase_coherence": _median_or_none(
            [d.phase_coherence for d in peaks if d.phase_coherence is not None]
        ),
        "median_abs_bin_error": _median_or_none(
            [float(d.bin_error) for d in peaks if d.bin_error is not None]
        ),
        "median_abs_speed_error_mph": _median_or_none(
            [abs(float(d.speed_error_mph)) for d in peaks if d.speed_error_mph is not None]
        ),
        "median_peak_width_bins": _median_or_none([float(d.peak_width_bins) for d in peaks]),
        "peak_bin_histogram_top": peak_bin_histogram,
        "channel_clip_max": channel_clip_max,
        "warnings_by_type": warnings_by_type,
    }


def radc_capture_diagnostics(
    frames: list[dict],
    **kwargs: object,
) -> tuple[list[RADCFrameDiagnostics], dict[str, object]]:
    """Run raw ADC diagnostics for a capture and return rows plus summary."""
    diagnostics = [
        radc_frame_diagnostics(frame, frame_index=i, **kwargs) for i, frame in enumerate(frames)
    ]
    return diagnostics, summarize_radc_diagnostics(diagnostics)


def find_impact_frames(
    frames: list[dict],
    fft_size: int = 2048,
    min_velocity_bin: int = 150,
    energy_threshold: float = 3.0,
    ball_bands: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Find frames with sudden high-velocity energy (impact events).

    Looks for frames where the high-velocity portion of the spectrum
    has significantly more energy than the surrounding frames.

    Checks both positive-velocity bins (min_velocity_bin to N/2, for club)
    and the ball velocity band(s). When ``ball_bands`` is provided, sums
    energy across all listed sub-ranges (typically one (lo, hi); two for
    the wrap-around case). When None, defaults to the full
    negative-velocity half N/2 to N. Golf ball speeds alias into the
    negative velocity range at RSPI=100 km/h, so checking only positive
    bins misses ball impacts.
    """
    energies = []
    for frame in frames:
        radc = frame.get("radc")
        if radc is None:
            energies.append(0.0)
            continue
        try:
            channels = parse_radc_payload(radc) if isinstance(radc, bytes) else radc
            iq = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
        except (KeyError, TypeError, ValueError):
            energies.append(0.0)
            continue
        spec = compute_spectrum(iq, fft_size=fft_size)
        # Energy in positive high-velocity bins (club swing)
        pos_energy = float(np.sum(spec[min_velocity_bin : fft_size // 2] ** 2))
        # Energy in aliased negative-velocity bins (ball)
        if ball_bands:
            neg_energy = sum(float(np.sum(spec[lo:hi] ** 2)) for lo, hi in ball_bands)
        else:
            neg_energy = float(np.sum(spec[fft_size // 2 + min_velocity_bin :] ** 2))
        energies.append(pos_energy + neg_energy)

    energies = np.array(energies)
    if np.median(energies) <= 0:
        return []

    # Frames where high-velocity energy exceeds median by threshold factor
    median_energy = np.median(energies[energies > 0])
    impact_indices = []
    for i, e in enumerate(energies):
        if e > energy_threshold * median_energy:
            impact_indices.append(i)
    return impact_indices


def extract_launch_angle(
    frames: list[dict],
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    cfar_threshold: float = 2.5,
    impact_energy_threshold: float = 3.0,
    angle_offset_deg: float = 0.0,
    ops243_ball_speed_mph: float | None = None,
    speed_tolerance_mph: float = 10.0,
    orientation: str | None = None,
    ops_bin_outlier_tol: int = 25,
    ops_bin_outlier_penalty: float = 10.0,
    centroid_floor_frac: float = 0.5,
    ops_anchored_peak_min_snr: float = OPS_ANCHORED_PEAK_MIN_SNR,
    require_ops_anchored_peak: bool = False,
    horizontal_angle_limit_deg: float = 15.0,
) -> list[dict]:
    """Extract vertical launch angle per shot from RADC frames.

    Pipeline:
    1. Find impact frames (high-velocity energy spikes)
    2. Group consecutive impacts into shot events
    3. For each shot, run band-limited CFAR in the ball velocity range
    4. Per-bin interferometric angle estimation on ball detections.
       The per-frame angle is the magnitude²-weighted centroid of the
       per-bin angles inside the spectral peak (bins whose magnitude
       exceeds `centroid_floor_frac` of the peak), rather than the raw
       angle at the single peak bin. For range-spread targets like a
       golf ball whose energy spreads across multiple FFT bins due to
       Hann-window leakage and intra-frame Doppler smear, this is much
       more robust to noise than reading a single bin.
    5. SNR²-weighted average angle across frames. When the OPS243 ball
       speed is supplied, frames whose peak bin is more than
       ops_bin_outlier_tol away from the OPS-expected bin have their
       weight reduced by ops_bin_outlier_penalty (default 10×). This
       downweights persistent clutter stripes that sit inside the ball
       velocity band but far from the actual ball location. Horizontal
       extraction is stricter: it skips frames without a usable OPS-bin
       peak instead of falling back to a far in-band peak, because Test2
       TrackMan replay showed those fallback peaks drive launch-direction
       noise.
    6. Apply angle offset

    Args:
        ops243_ball_speed_mph: If provided (live path), narrows the velocity
            search to a tight band around this speed. This eliminates
            club/multipath contamination and works for any club/player.
            If None (offline analysis), uses the broad default ball range.
        speed_tolerance_mph: Search window ± around ops243_ball_speed_mph.
        ops_bin_outlier_tol: When ops243_ball_speed_mph is provided, frames
            whose peak bin is more than this many bins from the
            OPS-expected bin are downweighted in the final average.
            Has no effect when ops243_ball_speed_mph is None.
        ops_bin_outlier_penalty: Weight divisor for outlier frames
            (default 10×). Set to 1.0 to disable the soft check.
        centroid_floor_frac: Bins inside the ball band whose magnitude
            is at least this fraction of the peak are included in the
            per-frame magnitude²-weighted angle centroid (default 0.5,
            i.e. all bins above the half-power point of the peak). Set
            to 1.0 to revert to single-peak-bin angle extraction.
        ops_anchored_peak_min_snr: Minimum linear SNR required for the
            OPS-expected local peak before using that frame. Default
            preserves production behavior; lower experimental replay
            values can admit weak near-OPS horizontal peaks.
        require_ops_anchored_peak: When true, skip frames whose local peak near
            the OPS243-expected ball-speed bin is missing or below
            ops_anchored_peak_min_snr, instead of falling back to the strongest
            in-band peak. This is useful for TrackMan replay sessions where
            vertical fallback often latches onto clutter stripes.
        horizontal_angle_limit_deg: Symmetric horizontal bound in degrees
            used to reject obvious side-angle false positives. Default
            remains ±15° for production; experimental TrackMan replay can
            raise this when validating wider horizontal launch targets.

    Returns a list of shot dicts, one per detected shot. Each contains
    launch_angle_deg, ball_speed_mph, confidence, and supporting data.
    Returns empty list if no shots found.
    """
    # Velocity band: narrow (OPS243-anchored) or broad (offline default).
    # The band is a list of (lo, hi) sub-ranges. For most ball speeds
    # this is a single range; for speeds whose aliased velocity is
    # within ±tolerance of 0 km/h, the band wraps around DC and is
    # represented as two sub-ranges [N-tol, N) ∪ [0, tol).
    ball_bands: list[tuple[int, int]]
    if ops243_ball_speed_mph is not None:
        ball_bands = ball_bin_range_from_speed(
            ops243_ball_speed_mph,
            speed_tolerance_mph,
            fft_size,
            max_speed_kmh,
        )
        # Where the ball *should* peak, given the OPS243 speed. Used as a
        # soft anchor for the SNR²-weighted average below.
        ops_expected_bin: int | None = expected_ball_bin_from_speed(
            ops243_ball_speed_mph,
            fft_size,
            max_speed_kmh,
        )
    else:
        # Broad default ball velocity range for offline analysis.
        # Ball 100-120 mph aliases to -39 to -7 km/h at RSPI=3 (100 km/h max).
        ball_bands = default_ball_bin_ranges(fft_size, max_speed_kmh)
        ops_expected_bin = None

    min_velocity_bin = 150  # skip low-velocity body/clutter
    impact_indices = find_impact_frames(
        frames,
        fft_size=fft_size,
        min_velocity_bin=min_velocity_bin,
        energy_threshold=impact_energy_threshold,
        ball_bands=ball_bands,
    )
    if not impact_indices:
        import logging

        logging.getLogger("openflight.kld7.radc").info(
            "[KLD7-RADC] No impact frames found (energy_threshold=%.1f, ball_bands=%s, %d frames)",
            impact_energy_threshold,
            ball_bands,
            len(frames),
        )
        return []

    # Group consecutive impact frames into shot events
    shot_groups: list[list[int]] = []
    for idx in impact_indices:
        if not shot_groups or idx - shot_groups[-1][-1] > 5:
            shot_groups.append([idx])
        else:
            shot_groups[-1].append(idx)

    results = []
    for shot_idx, impact_group in enumerate(shot_groups):
        # Expand to impact -1 before, +2 after (ball appears slightly after)
        frame_set = set()
        for idx in impact_group:
            for offset in range(-1, 3):
                fi = idx + offset
                if 0 <= fi < len(frames):
                    frame_set.add(fi)

        # Peak-bin extraction: for each frame, find the single strongest
        # bin in the ball velocity band and take the angle at that bin only.
        # This avoids averaging across noisy weak detections.
        peak_angles = []
        peak_snrs = []
        peak_speeds_mph = []
        peak_bins: list[int] = []

        for fi in sorted(frame_set):
            radc_raw = frames[fi].get("radc")
            if radc_raw is None:
                continue
            try:
                channels = parse_radc_payload(radc_raw) if isinstance(radc_raw, bytes) else radc_raw
            except (KeyError, TypeError, ValueError):
                continue

            f1a_iq = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
            f2a_iq = to_complex_iq(channels["f2a_i"], channels["f2a_q"])

            spec = compute_spectrum(f1a_iq, fft_size=fft_size)
            # SNR of the peak bin vs full-spectrum noise floor
            full_median = float(np.median(spec[spec > 0]))
            peak_bin: int | None = None
            peak_val = 0.0
            peak_band: tuple[int, int] | None = None

            if ops_expected_bin is not None:
                peak_bin, peak_val, peak_band = _find_peak_near_expected_bin(
                    spec,
                    ball_bands,
                    ops_expected_bin,
                    ops_bin_outlier_tol,
                    fft_size,
                )
                anchored_snr = peak_val / full_median if full_median > 0 else 0.0
                if peak_bin is None or anchored_snr < ops_anchored_peak_min_snr:
                    if orientation == "horizontal" or require_ops_anchored_peak:
                        continue
                    peak_bin, peak_val, peak_band = _find_peak_in_bands(spec, tuple(ball_bands))
            else:
                peak_bin, peak_val, peak_band = _find_peak_in_bands(spec, tuple(ball_bands))

            if peak_val <= 0 or peak_bin is None:
                continue

            snr = peak_val / full_median if full_median > 0 else 0.0
            if snr < 2.0:
                continue

            # Per-bin angle at the peak
            f1a_fft = compute_fft_complex(f1a_iq, fft_size=fft_size)
            f2a_fft = compute_fft_complex(f2a_iq, fft_size=fft_size)
            angles = per_bin_angle_deg(f1a_fft, f2a_fft)

            # Magnitude²-weighted centroid of the per-bin angles across
            # the spectral peak, rather than the raw angle at a single
            # bin. Search a small neighborhood (`centroid_search_bins`)
            # around the peak and include bins whose magnitude is at
            # least `centroid_floor_frac` of the peak. For a range-
            # spread target this integrates the angle estimate across
            # all the energy in the peak; restricting to a neighborhood
            # prevents random noise bins elsewhere in the band (which
            # have similar magnitudes when there is no real ball signal)
            # from contributing. This is the wideband monopulse
            # formulation (Zhang et al., Sensors 2016).
            if centroid_floor_frac < 1.0:
                # Clip the centroid neighborhood to the sub-band that
                # contains the peak. This prevents the neighborhood
                # from spilling across the wrap into an unrelated
                # spectral region when the ball band wraps around DC.
                # Fall back to FFT bounds if peak_bin sits outside any
                # listed sub-band (defensive — shouldn't happen).
                sub_for_peak = next(
                    (sub for sub in ball_bands if sub[0] <= peak_bin < sub[1]),
                    (0, fft_size),
                )
                sub_lo, sub_hi = sub_for_peak
                lo_n = max(sub_lo, peak_bin - CENTROID_SEARCH_BINS)
                hi_n = min(sub_hi, peak_bin + CENTROID_SEARCH_BINS + 1)
                neigh = spec[lo_n:hi_n]
                neigh_mask = neigh >= peak_val * centroid_floor_frac
                if neigh_mask.any():
                    neigh_indices = np.flatnonzero(neigh_mask) + lo_n
                    neigh_w = neigh[neigh_mask] ** 2
                    neigh_w_sum = float(neigh_w.sum())
                    if neigh_w_sum > 0:
                        centroid_angle = float(
                            np.sum(angles[neigh_indices] * neigh_w) / neigh_w_sum
                        )
                    else:
                        centroid_angle = float(angles[peak_bin])
                else:
                    centroid_angle = float(angles[peak_bin])
            else:
                # Disabled (frac=1.0) — fall back to the legacy
                # single-peak-bin angle for exact backward compatibility.
                centroid_angle = float(angles[peak_bin])

            peak_angles.append(centroid_angle)
            peak_snrs.append(snr)
            peak_bins.append(peak_bin)
            vel = bin_to_velocity_kmh(peak_bin, fft_size, max_speed_kmh)
            peak_speeds_mph.append((200.0 + vel) / 1.609)

        if not peak_angles:
            continue

        angs = np.array(peak_angles)
        snrs = np.array(peak_snrs)
        bins_arr = np.array(peak_bins, dtype=int)

        if len(angs) == 1:
            # Single-frame detection — accept if SNR is strong.
            # Golf balls transit the K-LD7 beam in ~1 frame at 18 FPS,
            # so a single high-SNR frame is the expected case.
            if snrs[0] < 5.0:
                continue
            clean_angs = angs
            clean_snrs = snrs
            clean_bins = bins_arr
        else:
            # Multi-frame: outlier rejection.
            #
            # Drop the frame furthest from the median angle, *unless*
            # one frame's SNR is dramatically larger than the others.
            # In that case the median is being set by low-SNR noise
            # frames around a single high-SNR ball frame, and dropping
            # the angular outlier would discard the only real
            # detection. Instead we drop the lowest-SNR frame.
            clean_mask = np.ones(len(angs), dtype=bool)
            if len(angs) >= 3:
                max_snr = float(snrs.max())
                med_snr = float(np.median(snrs))
                snr_dominant = max_snr > 10.0 * max(med_snr, 1.0)
                if snr_dominant:
                    worst = int(np.argmin(snrs))
                else:
                    med = float(np.median(angs))
                    worst = int(np.argmax(np.abs(angs - med)))
                clean_mask[worst] = False
            clean_angs = angs[clean_mask]
            clean_snrs = snrs[clean_mask]
            clean_bins = bins_arr[clean_mask]

        # SNR²-weighted average of surviving peaks. When the OPS-expected
        # bin is known, frames whose peak bin is far from it (likely
        # clutter latched onto a persistent stripe) get a weight penalty.
        w = clean_snrs**2
        if (
            ops_expected_bin is not None
            and ops_bin_outlier_penalty > 1.0
            and ops_bin_outlier_tol >= 0
        ):
            bin_distances = np.array(
                [circular_bin_distance(b, ops_expected_bin, fft_size) for b in clean_bins]
            )
            outlier = bin_distances > ops_bin_outlier_tol
            if outlier.any():
                w = w.astype(float).copy()
                w[outlier] = w[outlier] / ops_bin_outlier_penalty
                penalty_count = int(outlier.sum())
                pct = 100.0 * penalty_count / len(clean_bins)
                # When ≥50% of frames are penalized, this almost always
                # indicates a setup problem (radar mounted off-axis or
                # locked onto a persistent clutter source) rather than a
                # rare outlier. Surface as WARNING so it shows up in
                # production logs without needing to replay.
                log_fn = logger.warning if pct >= 50.0 else logger.info
                log_fn(
                    "[RADC] OPS-bin penalty: %d/%d frames (%.0f%%) > %d "
                    "bins from expected bin %d (peak bins: %s, weight "
                    "/%.1f). High rate suggests a radar mounting or "
                    "clutter issue — see docs/kld7-troubleshooting.md.",
                    penalty_count,
                    len(clean_bins),
                    pct,
                    ops_bin_outlier_tol,
                    ops_expected_bin,
                    list(map(int, clean_bins)),
                    ops_bin_outlier_penalty,
                )
        total_w = float(np.sum(w))
        if total_w <= 0:
            continue
        weighted_angle = float(np.sum(clean_angs * w) / total_w)
        corrected_angle = weighted_angle + angle_offset_deg

        # Hard physical bounds — reject obvious outliers before they
        # reach the Shot object. Orientation-aware: vertical [0°, 45°],
        # horizontal [-15°, +15°]. When orientation is None (offline
        # analysis), skip bounds filtering.
        if orientation == "vertical" and (corrected_angle < 0.0 or corrected_angle > 45.0):
            logger.info(
                "[RADC] Vertical angle %.1f° outside [0, 45] — rejected",
                corrected_angle,
            )
            continue
        horizontal_limit = max(float(horizontal_angle_limit_deg), 0.0)
        if orientation == "horizontal" and abs(corrected_angle) > horizontal_limit:
            logger.info(
                "[RADC] Horizontal angle %.1f° outside ±%.1f° — rejected",
                corrected_angle,
                horizontal_limit,
            )
            continue

        avg_speed_mph = float(np.mean(peak_speeds_mph))
        angle_std = float(np.std(clean_angs))
        avg_snr = float(np.mean(clean_snrs))

        # Confidence based primarily on SNR. For RADC, single-frame
        # detection is the expected case (ball transits in ~56ms at 18 FPS),
        # so frame count shouldn't penalize confidence. Multi-frame
        # detections get a bonus from angle consistency.
        frame_count = len(clean_angs)
        snr_score = min(avg_snr / 10.0, 1.0)
        if frame_count == 1:
            # Single frame: confidence driven by SNR alone
            # SNR 5 → 0.50, SNR 10 → 0.75, SNR 15+ → 0.90
            confidence = round(0.40 + snr_score * 0.50, 2)
        else:
            # Multi-frame: SNR + angle consistency bonus
            std_score = max(0.0, 1.0 - angle_std / 15.0)
            confidence = round(
                snr_score * 0.5 + std_score * 0.3 + min(frame_count / 3.0, 1.0) * 0.2, 2
            )

        results.append(
            {
                "shot_index": shot_idx,
                "launch_angle_deg": round(corrected_angle, 1),
                "raw_angle_deg": round(weighted_angle, 1),
                "angle_offset_deg": angle_offset_deg,
                "ball_speed_mph": round(avg_speed_mph, 1),
                "confidence": confidence,
                "detection_count": len(peak_angles),
                "frame_count": frame_count,
                "angle_std_deg": round(angle_std, 1),
                "avg_snr_db": round(avg_snr, 1),
                "impact_frames": impact_group,
            }
        )

    return results


def select_best_shot_result(results: list[dict]) -> dict:
    """Select the candidate nearest the triggering OPS shot.

    RADC extraction can return multiple chronological energy groups from the
    before-heavy live ring buffer. The OPS-triggered shot is closest to the end
    of that buffer, so prefer the candidate whose impact frames occur latest.
    """
    if not results:
        raise ValueError("select_best_shot_result requires at least one result")
    return max(results, key=lambda result: max(result.get("impact_frames") or [-1]))
