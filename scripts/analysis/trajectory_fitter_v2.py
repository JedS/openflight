#!/usr/bin/env python3
"""Trajectory-fit vertical launch estimator v2 — multi-bin bearing.

Differs from v1 by replacing the single-peak bearing read with a
**neighborhood-weighted bearing** at each frame:

  1. Find the local magnitude peak within ±search_window_bins of the
     OPS-predicted bin (same as v1).
  2. Look at bins within ±peak_neighborhood_bins of that peak; keep the
     ones whose magnitude is at least `peak_neighborhood_mag_frac` of the
     peak.
  3. Take a magnitude-weighted mean of those bins' per-bin angles.
  4. Report the angular spread (weighted std) within that neighborhood as
     a per-frame quality metric.

A clean ball signal produces a coherent peak — all bins in the
neighborhood report nearly the same angle (low spread). Clutter or
spectral leakage produces high spread.

Then in the trajectory fit:
  - Reject frames whose angle_std exceeds `angle_std_max`.
  - Weight each frame's contribution in the least-squares fit by
    1 / (angle_std + 0.5°).

Same single-parameter α grid search as v1 otherwise.
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


SESSION_GEOMETRY: dict[str, tuple[float, float, float, float]] = {
    "20260523_143732": (6.0, 18.0, 0.0, 12.0),
    "20260523_144415": (5.0, 18.0, 0.0, 10.0),
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
SAMPLE_RATE_HZ = 30_000
FIRST_BYTE_TRIGGER_DELAY_MS = 68.0
FFT_SIZE = 2048
ALPHA_SEARCH = np.arange(0.0, 35.1, 0.1)


@dataclass
class ShotResult:
    session: str
    shot_number: int
    club: str
    tm_launch_v_deg: float
    pi_impact_ts: float
    n_frames_used: int
    frames_t_ms: list[float]
    frames_beta_deg: list[float]
    frames_angle_std: list[float]
    estimated_alpha_deg: float | None
    error_deg: float | None
    fit_residual_deg: float | None


def _to_float(x: object) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_ts(s: str) -> float | None:
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


def load_first_byte_times(csv_path: Path, long_name: str) -> dict[int, float]:
    out: dict[int, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("session") != long_name:
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


def find_session_data(jsonl_path: Path):
    rbc: dict[int, dict] = {}
    kld7: dict[int, list[dict]] = {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            shot_n = d.get("shot_number")
            if shot_n is None:
                continue
            t = d.get("type")
            if t == "rolling_buffer_capture":
                rbc[int(shot_n)] = d
            elif t == "kld7_buffer" and d.get("orientation") == "vertical":
                frames = d.get("frames") or []
                if frames:
                    kld7[int(shot_n)] = frames
    return rbc, kld7


def pi_impact_ts_for_shot(rbc: dict, first_byte_ts: float,
                          extra_offset_ms: float = 0.0) -> float | None:
    ball_ts_ms = rbc.get("ball_timestamp_ms")
    trigger_off_ms = rbc.get("trigger_offset_ms")
    if ball_ts_ms is None or trigger_off_ms is None:
        return None
    trigger_delay_s = FIRST_BYTE_TRIGGER_DELAY_MS / 1000.0
    return (
        first_byte_ts - trigger_delay_s
        - (trigger_off_ms - ball_ts_ms) / 1000.0
        + extra_offset_ms / 1000.0
    )


def multi_bin_bearing(
    radc_bytes: bytes,
    ball_speed_mph: float,
    search_window_bins: int = 30,
    peak_neighborhood_bins: int = 4,
    peak_neighborhood_mag_frac: float = 0.5,
) -> tuple[float, float, float] | None:
    """Return (bearing_deg, snr_db, angle_std_deg).

    bearing is magnitude-weighted mean of per-bin angles in the peak
    neighborhood (bins within ±peak_neighborhood_bins of the local peak
    whose magnitude exceeds `peak_neighborhood_mag_frac` of the peak).
    angle_std is the magnitude-weighted std of those angles — a per-frame
    coherence indicator.
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

    expected_bin = expected_ball_bin_from_speed(ball_speed_mph, fft_size=FFT_SIZE)
    window_idx = [
        (expected_bin + off) % FFT_SIZE
        for off in range(-search_window_bins, search_window_bins + 1)
    ]
    window_mag = mag[window_idx]
    local_peak_pos = int(np.argmax(window_mag))
    peak_bin = window_idx[local_peak_pos]
    peak_mag = float(window_mag[local_peak_pos])
    if peak_mag <= 0:
        return None

    # Neighborhood around peak.
    nbhd_idx = [
        (peak_bin + off) % FFT_SIZE
        for off in range(-peak_neighborhood_bins, peak_neighborhood_bins + 1)
    ]
    nbhd_mag = mag[nbhd_idx]
    nbhd_ang = angles[nbhd_idx]

    # Keep bins above magnitude fraction of peak.
    mask = nbhd_mag >= peak_neighborhood_mag_frac * peak_mag
    if not np.any(mask):
        return None
    valid_mag = nbhd_mag[mask]
    valid_ang = nbhd_ang[mask]

    total_w = float(np.sum(valid_mag))
    weighted_mean = float(np.sum(valid_ang * valid_mag) / total_w)
    if len(valid_ang) > 1:
        weighted_var = float(np.sum(valid_mag * (valid_ang - weighted_mean) ** 2) / total_w)
        angle_std = math.sqrt(max(weighted_var, 0.0))
    else:
        angle_std = 0.0

    noise = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
    snr_db = 10.0 * math.log10(max(peak_mag / max(noise, 1e-9), 1e-9))

    return weighted_mean, snr_db, angle_std


def predicted_bearing_deg(
    alpha_deg: float, t_s: float, v_fts: float,
    d_initial_ft: float, mount_deg: float, ball_above_radar_ft: float,
) -> float:
    alpha_rad = RAD(alpha_deg)
    x = d_initial_ft + v_fts * math.cos(alpha_rad) * t_s
    y = ball_above_radar_ft + v_fts * math.sin(alpha_rad) * t_s
    if x <= 0:
        return float("inf")
    return DEG(math.atan2(y, x)) - mount_deg


def fit_alpha_weighted(
    observations: list[tuple[float, float, float]],
    # (t_s, beta_deg, angle_std_deg)
    ball_speed_mph: float, d_initial_ft: float, mount_deg: float,
    ball_above_radar_ft: float,
) -> tuple[float, float]:
    """Weighted 1D grid search for α."""
    v_fts = ball_speed_mph * MPH_TO_FTS
    best_alpha = float("nan")
    best_sse = float("inf")
    for alpha in ALPHA_SEARCH:
        sse = 0.0
        wsum = 0.0
        for t_s, beta_obs, std_deg in observations:
            beta_pred = predicted_bearing_deg(
                alpha, t_s, v_fts, d_initial_ft, mount_deg, ball_above_radar_ft,
            )
            # Weight = 1 / (std + 0.5°). Tighter coherence → more influence.
            w = 1.0 / (std_deg + 0.5)
            r = beta_obs - beta_pred
            sse += w * r * r
            wsum += w
        if sse < best_sse:
            best_sse = sse
            best_alpha = float(alpha)
    residual = math.sqrt(best_sse / max(wsum, 1e-9)) if observations else float("nan")
    return best_alpha, residual


def process_shot(
    rbc: dict, kld7_frames: list[dict], first_byte_ts: float, tm_v_deg: float,
    ball_speed_mph: float, club: str,
    d_initial_ft: float, mount_deg: float,
    ball_above_radar_ft: float, ball_to_net_ft: float,
    snr_min_db: float, angle_std_max_deg: float,
    t_post_ms: float, t_min_ms: float,
    search_window_bins: int, peak_neighborhood_bins: int,
    peak_neighborhood_mag_frac: float,
    impact_extra_offset_ms: float,
) -> ShotResult | None:
    impact_ts = pi_impact_ts_for_shot(rbc, first_byte_ts, impact_extra_offset_ms)
    if impact_ts is None:
        return None

    v_fts = ball_speed_mph * MPH_TO_FTS
    flight_s = ball_to_net_ft / (v_fts * math.cos(RAD(15.0)))

    t_lo = impact_ts
    t_hi = impact_ts + min(flight_s, t_post_ms / 1000.0)

    obs: list[tuple[float, float, float]] = []  # (t_s, beta, std)
    frames_t_ms: list[float] = []
    frames_beta: list[float] = []
    frames_std: list[float] = []

    for frame in kld7_frames:
        frame_ts = frame.get("timestamp")
        if frame_ts is None:
            continue
        if not (t_lo <= frame_ts <= t_hi):
            continue
        t_s = float(frame_ts) - impact_ts
        if t_s <= 0 or t_s * 1000.0 < t_min_ms:
            continue
        b64 = frame.get("radc_b64")
        if not isinstance(b64, str):
            continue
        try:
            radc = base64.b64decode(b64, validate=True)
        except ValueError:
            continue
        if len(radc) != RADC_PAYLOAD_BYTES:
            continue
        result = multi_bin_bearing(
            radc, ball_speed_mph,
            search_window_bins, peak_neighborhood_bins,
            peak_neighborhood_mag_frac,
        )
        if result is None:
            continue
        beta, snr_db, angle_std = result
        if snr_db < snr_min_db:
            continue
        if angle_std > angle_std_max_deg:
            continue
        obs.append((t_s, beta, angle_std))
        frames_t_ms.append(t_s * 1000)
        frames_beta.append(beta)
        frames_std.append(angle_std)

    if len(obs) < 2:
        return ShotResult(
            session="", shot_number=-1, club=club, tm_launch_v_deg=tm_v_deg,
            pi_impact_ts=impact_ts, n_frames_used=len(obs),
            frames_t_ms=frames_t_ms, frames_beta_deg=frames_beta,
            frames_angle_std=frames_std,
            estimated_alpha_deg=None, error_deg=None, fit_residual_deg=None,
        )

    alpha, fit_residual = fit_alpha_weighted(
        obs, ball_speed_mph, d_initial_ft, mount_deg, ball_above_radar_ft,
    )
    err = alpha - tm_v_deg
    return ShotResult(
        session="", shot_number=-1, club=club, tm_launch_v_deg=tm_v_deg,
        pi_impact_ts=impact_ts, n_frames_used=len(obs),
        frames_t_ms=frames_t_ms, frames_beta_deg=frames_beta,
        frames_angle_std=frames_std,
        estimated_alpha_deg=alpha, error_deg=err, fit_residual_deg=fit_residual,
    )


def run_session(
    sessions_dir: Path, timing_csv: Path, session_id: str,
    snr_min_db: float, angle_std_max_deg: float,
    t_post_ms: float, t_min_ms: float,
    search_window_bins: int, peak_neighborhood_bins: int,
    peak_neighborhood_mag_frac: float, impact_extra_offset_ms: float,
) -> list[ShotResult]:
    d_initial_ft, mount_deg, ball_above_radar_ft, ball_to_net_ft = SESSION_GEOMETRY[session_id]
    comparison_filename, jsonl_id = SESSION_FILES[session_id]
    long_name = SESSION_LONG_NAMES[session_id]

    first_byte_times = load_first_byte_times(timing_csv, long_name)
    targets = load_comparison(sessions_dir / comparison_filename)
    jsonl_path = sessions_dir / f"session_{jsonl_id}_range.jsonl"
    rbc_by_shot, kld7_by_shot = find_session_data(jsonl_path)

    rows: list[ShotResult] = []
    for shot_n in sorted(targets.keys()):
        tgt = targets[shot_n]
        rbc = rbc_by_shot.get(shot_n)
        if rbc is None:
            continue
        kld7_frames = kld7_by_shot.get(shot_n, [])
        if not kld7_frames:
            continue
        fb_ts = first_byte_times.get(shot_n)
        if fb_ts is None:
            continue
        res = process_shot(
            rbc=rbc, kld7_frames=kld7_frames, first_byte_ts=fb_ts,
            tm_v_deg=tgt["tm_launch_v_deg"],
            ball_speed_mph=tgt["ball_speed_mph"], club=tgt["club"],
            d_initial_ft=d_initial_ft, mount_deg=mount_deg,
            ball_above_radar_ft=ball_above_radar_ft, ball_to_net_ft=ball_to_net_ft,
            snr_min_db=snr_min_db, angle_std_max_deg=angle_std_max_deg,
            t_post_ms=t_post_ms, t_min_ms=t_min_ms,
            search_window_bins=search_window_bins,
            peak_neighborhood_bins=peak_neighborhood_bins,
            peak_neighborhood_mag_frac=peak_neighborhood_mag_frac,
            impact_extra_offset_ms=impact_extra_offset_ms,
        )
        if res is None:
            continue
        res.session = session_id
        res.shot_number = shot_n
        rows.append(res)
    return rows


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
        "retention_le_2": sum(1 for e in abs_errs if e <= 2.0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions-dir", type=Path,
                   default=Path("/Users/john.pacino/openflight_sessions"))
    p.add_argument("--timing-csv", type=Path,
                   default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/_trajfit_v2"))
    p.add_argument("--snr-min-db", type=float, default=2.0)
    p.add_argument("--angle-std-max-deg", type=float, default=10.0,
                   help="Reject frames whose per-frame neighborhood angle "
                        "spread exceeds this (default 10°).")
    p.add_argument("--t-post-ms", type=float, default=100.0)
    p.add_argument("--t-min-ms", type=float, default=15.0)
    p.add_argument("--search-window-bins", type=int, default=30)
    p.add_argument("--peak-neighborhood-bins", type=int, default=4)
    p.add_argument("--peak-neighborhood-mag-frac", type=float, default=0.5)
    p.add_argument("--impact-extra-offset-ms", type=float, default=-10.0)
    p.add_argument("--session-filter", nargs="*",
                   default=["20260523_143732", "20260523_144415"])
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions = [s for s in SESSION_GEOMETRY.keys() if s in args.session_filter]
    all_rows: list[ShotResult] = []
    for sid in sessions:
        print(f"Running {sid}...")
        rows = run_session(
            args.sessions_dir, args.timing_csv, sid,
            args.snr_min_db, args.angle_std_max_deg,
            args.t_post_ms, args.t_min_ms,
            args.search_window_bins, args.peak_neighborhood_bins,
            args.peak_neighborhood_mag_frac, args.impact_extra_offset_ms,
        )
        all_rows.extend(rows)
        s = summarize(rows)
        if s["n_detected"]:
            print(f"  n={int(s['n'])}  detected={int(s['n_detected'])}  "
                  f"MAE={s['mae_deg']:.3f}  bias={s['bias_deg']:+.3f}  "
                  f"RMSE={s['rmse_deg']:.3f}  "
                  f"retention<=8={int(s['retention_le_8'])}  "
                  f"retention<=2={int(s['retention_le_2'])}")

    out = args.output_dir / "per_shot_trajfit_v2.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session", "shot_number", "club", "tm_launch_v_deg",
            "estimated_alpha_deg", "error_deg", "fit_residual_deg",
            "n_frames_used", "frames_t_ms", "frames_beta_deg", "frames_angle_std",
        ])
        for r in all_rows:
            w.writerow([
                r.session, r.shot_number, r.club, f"{r.tm_launch_v_deg:.2f}",
                "" if r.estimated_alpha_deg is None else f"{r.estimated_alpha_deg:.2f}",
                "" if r.error_deg is None else f"{r.error_deg:.2f}",
                "" if r.fit_residual_deg is None else f"{r.fit_residual_deg:.3f}",
                r.n_frames_used,
                ";".join(f"{t:.1f}" for t in r.frames_t_ms),
                ";".join(f"{b:.2f}" for b in r.frames_beta_deg),
                ";".join(f"{s:.2f}" for s in r.frames_angle_std),
            ])

    overall = summarize(all_rows)
    print("\n=== OVERALL ===")
    if overall["n_detected"]:
        print(f"  n={int(overall['n'])}  detected={int(overall['n_detected'])}  "
              f"MAE={overall['mae_deg']:.3f}  bias={overall['bias_deg']:+.3f}  "
              f"RMSE={overall['rmse_deg']:.3f}  "
              f"retention<=8={int(overall['retention_le_8'])}  "
              f"retention<=2={int(overall['retention_le_2'])}")
    print(f"\nWrote per-shot results to {out}")


if __name__ == "__main__":
    main()
