"""Standalone helpers for K-LD7 raw ADC (RADC) signal processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RADC_PAYLOAD_BYTES = 3072
SAMPLES_PER_CHANNEL = 256
ADC_MIDPOINT = 32768  # uint16 midpoint for DC offset removal


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
        raise ValueError(
            f"RADC payload must be {RADC_PAYLOAD_BYTES} bytes, got {len(payload)}"
        )
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
    """Convert uint16 I/Q arrays to complex float, removing DC offset."""
    i_float = i_channel.astype(np.float64) - ADC_MIDPOINT
    q_float = q_channel.astype(np.float64) - ADC_MIDPOINT
    return i_float + 1j * q_float


def compute_spectrum(iq: np.ndarray, fft_size: int = 2048) -> np.ndarray:
    """Compute magnitude spectrum from complex I/Q with Hann window and zero-padding.

    Args:
        iq: Complex I/Q array (256 samples from RADC)
        fft_size: FFT length (zero-padded if > len(iq))

    Returns:
        Magnitude spectrum (linear scale), length = fft_size
    """
    windowed = iq * np.hanning(len(iq))
    padded = np.zeros(fft_size, dtype=np.complex128)
    padded[: len(windowed)] = windowed
    fft_result = np.fft.fft(padded)
    return np.abs(fft_result)


@dataclass(frozen=True)
class CFARDetection:
    bin_index: int
    magnitude: float
    snr_db: float


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


@dataclass(frozen=True)
class RADCDetection:
    frame_index: int
    timestamp: float
    distance_m: float
    velocity_kmh: float
    angle_deg: float
    magnitude: float
    snr_db: float
    bin_index: int


def bin_to_velocity_kmh(bin_index: int, fft_size: int, max_speed_kmh: float) -> float:
    """Convert FFT bin index to velocity in km/h.

    Bins 0..N/2 = 0..+max_speed (outbound).
    Bins N/2..N = -max_speed..0 (inbound, aliased).
    """
    if bin_index <= fft_size // 2:
        return bin_index * max_speed_kmh / (fft_size // 2)
    else:
        return (bin_index - fft_size) * max_speed_kmh / (fft_size // 2)


def estimate_angle_from_phase(
    f1_complex: np.ndarray,
    f2_complex: np.ndarray,
) -> float:
    """Estimate angle from phase difference between two frequency channels.

    Uses cross-correlation phase to estimate the angle of arrival.
    The exact angle-to-phase mapping depends on K-LD7 antenna geometry
    (spacing, wavelength). This returns a proportional estimate that
    needs empirical calibration against known angles.

    Returns:
        Angle estimate in degrees (uncalibrated — proportional to phase diff)
    """
    # Cross-spectral phase
    cross = np.sum(f1_complex * np.conj(f2_complex))
    phase_rad = np.angle(cross)
    # Convert to degrees — scale factor TBD from calibration
    # For K-LD7 at 24 GHz with ~6mm antenna spacing, rough estimate:
    # angle ≈ arcsin(phase / pi) * (180/pi)
    # For now return raw phase in degrees as a proportional estimate
    return float(np.degrees(phase_rad))


def process_radc_frame(
    frame: dict,
    frame_index: int,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    cfar_threshold: float = 8.0,
    cfar_guard: int = 4,
    cfar_training: int = 16,
) -> list[RADCDetection]:
    """Process one RADC frame: parse → FFT → CFAR → physical units.

    Uses F1A channel as primary, F2A for angle estimation.
    """
    radc_raw = frame.get("radc")
    if radc_raw is None:
        return []

    if isinstance(radc_raw, bytes):
        channels = parse_radc_payload(radc_raw)
    else:
        channels = radc_raw

    f1a = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
    f2a = to_complex_iq(channels["f2a_i"], channels["f2a_q"])

    spectrum = compute_spectrum(f1a, fft_size=fft_size)
    cfar_hits = cfar_detect(
        spectrum,
        guard_cells=cfar_guard,
        training_cells=cfar_training,
        threshold_factor=cfar_threshold,
    )

    angle_deg = estimate_angle_from_phase(f1a, f2a)
    timestamp = float(frame["timestamp"])

    detections = []
    for hit in cfar_hits:
        velocity = bin_to_velocity_kmh(hit.bin_index, fft_size, max_speed_kmh)
        detections.append(
            RADCDetection(
                frame_index=frame_index,
                timestamp=timestamp,
                distance_m=0.0,  # RADC gives velocity, not range — set from FMCW chirp later
                velocity_kmh=velocity,
                angle_deg=angle_deg,
                magnitude=hit.magnitude,
                snr_db=hit.snr_db,
                bin_index=hit.bin_index,
            )
        )

    return detections


def compare_radc_vs_pdat(
    radc_detections: list[RADCDetection],
    pdat: list[dict],
) -> dict:
    """Compare our RADC FFT detections against the module's PDAT output.

    Returns a summary dict for logging / CSV export.
    """
    pdat_speeds = [abs(p.get("speed", 0)) for p in pdat if p]
    pdat_mags = [p.get("magnitude", 0) for p in pdat if p]
    radc_velocities = [abs(d.velocity_kmh) for d in radc_detections]
    radc_mags = [d.magnitude for d in radc_detections]

    return {
        "radc_count": len(radc_detections),
        "pdat_count": len(pdat),
        "radc_max_velocity_kmh": max(radc_velocities) if radc_velocities else 0.0,
        "pdat_max_speed_kmh": max(pdat_speeds) if pdat_speeds else 0.0,
        "radc_max_magnitude": max(radc_mags) if radc_mags else 0.0,
        "pdat_max_magnitude": max(pdat_mags) if pdat_mags else 0.0,
    }
