#!/usr/bin/env python3
"""Geometry-aware vertical launch selector v2 — per-bin-angle version.

Key change vs v1: instead of using `angle_peak_deg` (angle at the FFT peak,
which is usually clutter), we look up the angle at the OPS-predicted bin
(where the ball *should* be), searching a small window around that bin for
the local magnitude peak. This catches the ball signal even when clutter
dominates the overall spectrum.

Per frame:
  1. Decode RADC bytes -> F1A and F2A I/Q channels
  2. Complex FFT on each channel
  3. Compute per-bin angle array from phase difference
  4. Find the local magnitude peak within ±N bins of the OPS-predicted bin
  5. Read off the angle at that bin

Then geometric correction:
  α = arctan(K) + arcsin(K·d_initial / (v·t·sqrt(1+K²)))   (K = tan(β + mount))

Median across all good frames, gated by t-window and physical α range.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from openflight.kld7.radc import (  # noqa: E402
    RADC_PAYLOAD_BYTES,
    compute_fft_complex,
    expected_ball_bin_from_speed,
    parse_radc_payload,
    per_bin_angle_deg,
    to_complex_iq,
)


SESSION_GEOMETRY: dict[str, tuple[float, float, float]] = {
    "20260523_143732": (6.0, 18.0, 0.0),
    "20260523_144415": (5.0, 18.0, 0.0),
}

SESSION_LONG_NAMES: dict[str, str] = {
    "20260523_143732": "20260523_143732_18deg_7iron_8shots",
    "20260523_144415": "20260523_144415_18deg_7iron_5shots_cleaned",
}

SESSION_FILES: dict[str, tuple[str, str]] = {
    "20260523_143732": ("comparison_20260523_143732.csv", "20260523_143732"),
    "20260523_144415": ("comparison_20260523_144415.csv", "20260523_144415"),
}

MPH_TO_FTS = 1.46667
DEG = math.degrees
RAD = math.radians

ALPHA_BAND_DEG = (-5.0, 45.0)
FFT_SIZE = 2048


@dataclass
class ShotResult:
    session: str
    shot_number: int
    club: str
    tm_launch_v_deg: float
    estimated_alpha_deg: float | None
    error_deg: float | None
    impact_ts: float
    n_frames_total: int
    n_with_radc: int
    n_after_snr: int
    n_after_alpha_band: int


def _to_float(x: object) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_ts(s: str) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


def load_first_byte_times(csv_path: Path, long_session_name: str) -> dict[int, float]:
    out: dict[int, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("session") != long_session_name:
                continue
            shot_no = row.get("json_shot_no")
            ts = _parse_ts(row.get("first_byte_ts", ""))
            if shot_no is None or ts is None:
                continue
            try:
                out[int(shot_no)] = ts
            except ValueError:
                pass
    return out


def load_comparison(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("match_quality") != "good":
                continue
            shot = _to_float(row.get("shot_number_of"))
            tm_v = _to_float(row.get("launch_v_tm"))
            bs = _to_float(row.get("ball_speed_of"))
            if shot is None or tm_v is None or bs is None:
                continue
            out[int(shot)] = {
                "ball_speed_mph": bs,
                "tm_launch_v_deg": tm_v,
                "club": row.get("club", ""),
            }
    return out


def iter_vertical_buffers(jsonl_path: Path):
    with jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "kld7_buffer" or d.get("orientation") != "vertical":
                continue
            shot_n = d.get("shot_number")
            frames = d.get("frames") or []
            if shot_n is None or not frames:
                continue
            yield int(shot_n), frames


def angle_and_snr_at_ball_bin(
    radc_bytes: bytes,
    ball_speed_mph: float,
    search_window_bins: int = 60,
) -> tuple[float, float, int] | None:
    """Return (angle_deg, snr_db, bin) at the local peak nearest the OPS-predicted bin.

    Steps:
      - Decode RADC payload into F1A and F2A I/Q channels.
      - Compute complex FFT of each channel.
      - Find expected bin from OPS ball speed.
      - Within ±search_window_bins (handling FFT wrap), pick the bin with
        max combined magnitude.
      - Return per-bin-angle at that bin and an SNR estimate.

    Returns None on parse error.
    """
    try:
        parsed = parse_radc_payload(radc_bytes)
    except ValueError:
        return None

    iq1 = to_complex_iq(parsed["f1a_i"], parsed["f1a_q"])
    iq2 = to_complex_iq(parsed["f2a_i"], parsed["f2a_q"])
    fft1 = compute_fft_complex(iq1, fft_size=FFT_SIZE)
    fft2 = compute_fft_complex(iq2, fft_size=FFT_SIZE)
    mag = np.abs(fft1) + np.abs(fft2)
    angles = per_bin_angle_deg(fft1, fft2)

    expected_bin = expected_ball_bin_from_speed(
        ball_speed_mph, fft_size=FFT_SIZE,
    )

    # Build search-window index list with FFT wrap-around.
    window_idx = [
        (expected_bin + off) % FFT_SIZE
        for off in range(-search_window_bins, search_window_bins + 1)
    ]
    window_mag = mag[window_idx]
    local_peak_pos = int(np.argmax(window_mag))
    peak_bin = window_idx[local_peak_pos]
    peak_mag = float(window_mag[local_peak_pos])

    # Noise floor: median magnitude across whole spectrum (excluding DC-masked
    # bins, which are already zeroed by compute_fft_complex).
    noise = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
    snr_linear = peak_mag / noise if noise > 0 else 0.0
    snr_db = 10.0 * math.log10(max(snr_linear, 1e-9))

    return float(angles[peak_bin]), snr_db, peak_bin


def alpha_from_frame(
    bearing_deg: float,
    t_after_impact_s: float,
    ball_speed_mph: float,
    d_initial_ft: float,
    mount_deg: float,
    ball_above_radar_ft: float,
) -> float | None:
    if t_after_impact_s <= 0:
        return None
    v_fts = ball_speed_mph * MPH_TO_FTS
    if v_fts <= 0:
        return None
    K = math.tan(RAD(bearing_deg + mount_deg))
    rhs = (K * d_initial_ft + ball_above_radar_ft) / (v_fts * t_after_impact_s)
    denom = math.sqrt(1.0 + K * K)
    arg = rhs / denom
    if not -1.0 <= arg <= 1.0:
        return None
    return DEG(math.atan(K) + math.asin(arg))


def process_shot(
    frames: list[dict],
    impact_ts: float,
    ball_speed_mph: float,
    d_initial_ft: float,
    mount_deg: float,
    ball_above_radar_ft: float,
    snr_min_db: float,
    t_window: tuple[float, float],
    search_window_bins: int,
) -> tuple[float | None, dict]:
    n_total = len(frames)
    n_radc = 0
    n_snr = 0
    alphas: list[float] = []

    for raw_frame in frames:
        radc_b64 = raw_frame.get("radc_b64")
        if not isinstance(radc_b64, str):
            continue
        try:
            radc = base64.b64decode(radc_b64, validate=True)
        except ValueError:
            continue
        if len(radc) != RADC_PAYLOAD_BYTES:
            continue

        frame_ts_raw = raw_frame.get("timestamp")
        if frame_ts_raw is None:
            continue
        t = float(frame_ts_raw) - impact_ts
        if not (t_window[0] <= t <= t_window[1]):
            continue
        n_radc += 1

        res = angle_and_snr_at_ball_bin(radc, ball_speed_mph, search_window_bins)
        if res is None:
            continue
        beta, snr_db, _peak_bin = res
        if snr_db < snr_min_db:
            continue
        n_snr += 1

        alpha = alpha_from_frame(
            bearing_deg=beta,
            t_after_impact_s=t,
            ball_speed_mph=ball_speed_mph,
            d_initial_ft=d_initial_ft,
            mount_deg=mount_deg,
            ball_above_radar_ft=ball_above_radar_ft,
        )
        if alpha is None:
            continue
        if not (ALPHA_BAND_DEG[0] <= alpha <= ALPHA_BAND_DEG[1]):
            continue
        alphas.append(alpha)

    if not alphas:
        return None, {
            "n_frames_total": n_total,
            "n_with_radc": n_radc,
            "n_after_snr": n_snr,
            "n_after_alpha_band": 0,
        }

    sorted_alphas = sorted(alphas)
    m = len(sorted_alphas)
    median = (
        sorted_alphas[m // 2] if m % 2 == 1
        else 0.5 * (sorted_alphas[m // 2 - 1] + sorted_alphas[m // 2])
    )
    return median, {
        "n_frames_total": n_total,
        "n_with_radc": n_radc,
        "n_after_snr": n_snr,
        "n_after_alpha_band": len(alphas),
    }


def run_session(
    sessions_dir: Path, timing_csv: Path, session_id: str,
    snr_min_db: float, t_window: tuple[float, float],
    impact_offset_s: float, search_window_bins: int,
) -> list[ShotResult]:
    d_initial_ft, mount_deg, ball_above_radar_ft = SESSION_GEOMETRY[session_id]
    comparison_filename, jsonl_id = SESSION_FILES[session_id]
    long_name = SESSION_LONG_NAMES[session_id]

    first_byte_times = load_first_byte_times(timing_csv, long_name)
    targets = load_comparison(sessions_dir / comparison_filename)
    jsonl_path = sessions_dir / f"session_{jsonl_id}_range.jsonl"

    results: list[ShotResult] = []
    for shot_n, frames in iter_vertical_buffers(jsonl_path):
        tgt = targets.get(shot_n)
        if tgt is None:
            continue
        fb_ts = first_byte_times.get(shot_n)
        if fb_ts is None:
            continue
        impact_ts = fb_ts + impact_offset_s

        alpha, stats = process_shot(
            frames=frames, impact_ts=impact_ts,
            ball_speed_mph=tgt["ball_speed_mph"],
            d_initial_ft=d_initial_ft, mount_deg=mount_deg,
            ball_above_radar_ft=ball_above_radar_ft,
            snr_min_db=snr_min_db, t_window=t_window,
            search_window_bins=search_window_bins,
        )
        err = (alpha - tgt["tm_launch_v_deg"]) if alpha is not None else None
        results.append(ShotResult(
            session=session_id, shot_number=shot_n, club=tgt["club"],
            tm_launch_v_deg=tgt["tm_launch_v_deg"],
            estimated_alpha_deg=alpha, error_deg=err,
            impact_ts=impact_ts,
            n_frames_total=stats["n_frames_total"],
            n_with_radc=stats["n_with_radc"],
            n_after_snr=stats["n_after_snr"],
            n_after_alpha_band=stats["n_after_alpha_band"],
        ))
    return results


def summarize(rows: list[ShotResult]) -> dict[str, float]:
    detected = [r for r in rows if r.error_deg is not None]
    if not detected:
        return {"n": float(len(rows)), "n_detected": 0.0}
    errs = [r.error_deg for r in detected]
    abs_errs = [abs(e) for e in errs]
    n = len(detected)
    return {
        "n": float(len(rows)),
        "n_detected": float(n),
        "mae_deg": sum(abs_errs) / n,
        "bias_deg": sum(errs) / n,
        "rmse_deg": math.sqrt(sum(e * e for e in errs) / n),
        "retention_le_8": sum(1 for e in abs_errs if e <= 8.0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions-dir", type=Path,
                   default=Path("/Users/john.pacino/openflight_sessions"))
    p.add_argument("--timing-csv", type=Path,
                   default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/_geom_v2"))
    p.add_argument("--snr-min-db", type=float, default=8.0)
    p.add_argument("--t-window", type=float, nargs=2, default=[0.05, 3.0])
    p.add_argument("--impact-offset-s", type=float, default=-0.068)
    p.add_argument("--search-window-bins", type=int, default=30)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[ShotResult] = []
    for sid in SESSION_GEOMETRY.keys():
        print(f"Running {sid}...")
        rows = run_session(
            args.sessions_dir, args.timing_csv, sid,
            args.snr_min_db, tuple(args.t_window),
            args.impact_offset_s, args.search_window_bins,
        )
        all_rows.extend(rows)
        s = summarize(rows)
        if s["n_detected"]:
            print(f"  n={int(s['n'])}  detected={int(s['n_detected'])}  "
                  f"MAE={s['mae_deg']:.3f}  bias={s['bias_deg']:+.3f}  "
                  f"RMSE={s['rmse_deg']:.3f}  retention<=8={int(s['retention_le_8'])}")

    out = args.output_dir / "per_shot_geom_v2.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session", "shot_number", "club", "tm_launch_v_deg",
            "estimated_alpha_deg", "error_deg", "impact_ts",
            "n_frames_total", "n_with_radc", "n_after_snr", "n_after_alpha_band",
        ])
        for r in all_rows:
            w.writerow([
                r.session, r.shot_number, r.club, f"{r.tm_launch_v_deg:.2f}",
                "" if r.estimated_alpha_deg is None else f"{r.estimated_alpha_deg:.2f}",
                "" if r.error_deg is None else f"{r.error_deg:.2f}",
                f"{r.impact_ts:.3f}",
                r.n_frames_total, r.n_with_radc, r.n_after_snr, r.n_after_alpha_band,
            ])

    overall = summarize(all_rows)
    print("\n=== OVERALL ===")
    if overall["n_detected"]:
        print(f"  n={int(overall['n'])}  detected={int(overall['n_detected'])}  "
              f"MAE={overall['mae_deg']:.3f}  bias={overall['bias_deg']:+.3f}  "
              f"RMSE={overall['rmse_deg']:.3f}")
    print(f"\nWrote per-shot results to {out}")


if __name__ == "__main__":
    main()
