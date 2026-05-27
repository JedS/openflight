#!/usr/bin/env python3
"""Trajectory-fit vertical launch estimator.

Pipeline per shot:
  1. Pull OPS-detected impact time (ball_timestamp_ms in the rolling
     buffer capture, converted to Pi clock).
  2. Find the 2-4 K-LD7 frames in [impact_ts, impact_ts + flight_window].
  3. For each frame, compute bearing at the local magnitude peak near the
     OPS-predicted bin (same as geom v2's per-bin lookup).
  4. Grid-search α ∈ [0°, 35°] minimizing sum of squared residuals against
     the geometric bearing trajectory model:

       β_pred(α; t) = arctan( v·sin(α)·t / (d_initial + v·cos(α)·t) ) - mount

  5. Report the α that best fits the (t, β) observations.
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


# Per-session geometry.
# (d_initial_ft, mount_deg, ball_above_radar_ft, ball_to_net_ft)
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
BUFFER_DURATION_MS = 4096 / SAMPLE_RATE_HZ * 1000.0  # ~136.5 ms
FIRST_BYTE_TRIGGER_DELAY_MS = 68.0
FFT_SIZE = 2048
ALPHA_SEARCH = np.arange(0.0, 35.1, 0.1)  # grid search resolution


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
    """Return (rolling_buffer_captures, kld7_vertical_buffers) keyed by shot."""
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
    """Derive Pi-clock impact timestamp from OPS rolling-buffer capture.

    `extra_offset_ms`: positive shifts impact later (and thus reduces the
    apparent t of each K-LD7 frame), useful for tuning out systematic
    timing biases.
    """
    ball_ts_ms = rbc.get("ball_timestamp_ms")
    trigger_off_ms = rbc.get("trigger_offset_ms")
    if ball_ts_ms is None or trigger_off_ms is None:
        return None
    trigger_delay_s = FIRST_BYTE_TRIGGER_DELAY_MS / 1000.0
    return (
        first_byte_ts
        - trigger_delay_s
        - (trigger_off_ms - ball_ts_ms) / 1000.0
        + extra_offset_ms / 1000.0
    )


def angle_at_ball_bin(radc_bytes: bytes, ball_speed_mph: float,
                      search_window_bins: int = 30) -> tuple[float, float] | None:
    """Return (bearing_deg, snr_db) at the local magnitude peak near the
    OPS-predicted bin. Returns None on parse error.
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
    local_peak = int(np.argmax(window_mag))
    peak_bin = window_idx[local_peak]
    peak_mag = float(window_mag[local_peak])
    noise = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
    snr_db = 10.0 * math.log10(max(peak_mag / max(noise, 1e-9), 1e-9))
    return float(angles[peak_bin]), snr_db


def predicted_bearing_deg(
    alpha_deg: float, t_s: float, v_fts: float,
    d_initial_ft: float, mount_deg: float, ball_above_radar_ft: float,
) -> float:
    """Geometric model: bearing the K-LD7 should observe at time t."""
    alpha_rad = RAD(alpha_deg)
    x = d_initial_ft + v_fts * math.cos(alpha_rad) * t_s
    y = ball_above_radar_ft + v_fts * math.sin(alpha_rad) * t_s
    if x <= 0:
        return float("inf")
    return DEG(math.atan2(y, x)) - mount_deg


def fit_alpha(
    observations: list[tuple[float, float]],  # (t_s, beta_deg)
    ball_speed_mph: float, d_initial_ft: float, mount_deg: float,
    ball_above_radar_ft: float,
) -> tuple[float, float]:
    """1D grid search for α minimizing SSE against observed bearings.

    Returns (alpha_deg, residual_rmse_deg).
    """
    v_fts = ball_speed_mph * MPH_TO_FTS
    best_alpha = float("nan")
    best_sse = float("inf")
    for alpha in ALPHA_SEARCH:
        sse = 0.0
        for t_s, beta_obs in observations:
            beta_pred = predicted_bearing_deg(
                alpha, t_s, v_fts, d_initial_ft, mount_deg, ball_above_radar_ft,
            )
            r = beta_obs - beta_pred
            sse += r * r
        if sse < best_sse:
            best_sse = sse
            best_alpha = float(alpha)
    n = len(observations)
    residual_rmse = math.sqrt(best_sse / n) if n > 0 else float("nan")
    return best_alpha, residual_rmse


def process_shot(
    rbc: dict, kld7_frames: list[dict], first_byte_ts: float, tm_v_deg: float,
    ball_speed_mph: float, club: str,
    d_initial_ft: float, mount_deg: float,
    ball_above_radar_ft: float, ball_to_net_ft: float,
    snr_min_db: float, t_pre_ms: float, t_post_ms: float,
    search_window_bins: int, impact_extra_offset_ms: float = 0.0,
    t_min_ms: float = 0.0,
) -> ShotResult | None:
    impact_ts = pi_impact_ts_for_shot(rbc, first_byte_ts, impact_extra_offset_ms)
    if impact_ts is None:
        return None

    v_fts = ball_speed_mph * MPH_TO_FTS
    # Conservative flight time estimate using α≈15° (cos drops by <2%).
    flight_s = ball_to_net_ft / (v_fts * math.cos(RAD(15.0)))

    t_lo = impact_ts - t_pre_ms / 1000.0
    t_hi = impact_ts + min(flight_s, t_post_ms / 1000.0)

    obs: list[tuple[float, float]] = []  # (t_s, beta_deg)
    frames_t_ms: list[float] = []
    frames_beta: list[float] = []
    for frame in kld7_frames:
        frame_ts = frame.get("timestamp")
        if frame_ts is None:
            continue
        if not (t_lo <= frame_ts <= t_hi):
            continue
        t_s = float(frame_ts) - impact_ts
        if t_s <= 0:
            # Skip pre-impact frames (no ball in flight yet).
            continue
        if t_s * 1000.0 < t_min_ms:
            # Skip frames too close to impact — ball still accelerating off
            # the face, geometry model not yet stable.
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
        result = angle_at_ball_bin(radc, ball_speed_mph, search_window_bins)
        if result is None:
            continue
        beta, snr_db = result
        if snr_db < snr_min_db:
            continue
        obs.append((t_s, beta))
        frames_t_ms.append(t_s * 1000)
        frames_beta.append(beta)

    if len(obs) < 2:
        return ShotResult(
            session="", shot_number=-1, club=club, tm_launch_v_deg=tm_v_deg,
            pi_impact_ts=impact_ts, n_frames_used=len(obs),
            frames_t_ms=frames_t_ms, frames_beta_deg=frames_beta,
            estimated_alpha_deg=None, error_deg=None, fit_residual_deg=None,
        )

    alpha, fit_residual = fit_alpha(
        obs, ball_speed_mph, d_initial_ft, mount_deg, ball_above_radar_ft,
    )
    err = alpha - tm_v_deg
    return ShotResult(
        session="", shot_number=-1, club=club, tm_launch_v_deg=tm_v_deg,
        pi_impact_ts=impact_ts, n_frames_used=len(obs),
        frames_t_ms=frames_t_ms, frames_beta_deg=frames_beta,
        estimated_alpha_deg=alpha, error_deg=err, fit_residual_deg=fit_residual,
    )


def run_session(
    sessions_dir: Path, timing_csv: Path, session_id: str,
    snr_min_db: float, t_pre_ms: float, t_post_ms: float,
    search_window_bins: int, impact_extra_offset_ms: float = 0.0,
    t_min_ms: float = 0.0,
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
            snr_min_db=snr_min_db, t_pre_ms=t_pre_ms, t_post_ms=t_post_ms,
            search_window_bins=search_window_bins,
            impact_extra_offset_ms=impact_extra_offset_ms,
            t_min_ms=t_min_ms,
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
        "p90_abs_deg": sorted(abs_errs)[int(math.ceil(0.9 * n) - 1)] if n else float("nan"),
        "retention_le_8": sum(1 for e in abs_errs if e <= 8.0),
        "retention_le_2": sum(1 for e in abs_errs if e <= 2.0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions-dir", type=Path,
                   default=Path("/Users/john.pacino/openflight_sessions"))
    p.add_argument("--timing-csv", type=Path,
                   default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/_trajfit"))
    p.add_argument("--snr-min-db", type=float, default=2.0)
    p.add_argument("--t-pre-ms", type=float, default=0.0,
                   help="Take frames up to this many ms BEFORE impact too (default 0 = post-impact only)")
    p.add_argument("--t-post-ms", type=float, default=100.0,
                   help="Maximum ms after impact to accept frames (default 100ms)")
    p.add_argument("--search-window-bins", type=int, default=30)
    p.add_argument("--session-filter", nargs="*",
                   default=["20260523_143732", "20260523_144415"])
    p.add_argument("--impact-extra-offset-ms", type=float, default=0.0,
                   help="Add this many ms to impact timing (positive = later). "
                        "Use to tune out systematic timing biases.")
    p.add_argument("--t-min-ms", type=float, default=15.0,
                   help="Reject frames with t < this many ms post-impact "
                        "(ball still accelerating off face).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions = [s for s in SESSION_GEOMETRY.keys() if s in args.session_filter]
    all_rows: list[ShotResult] = []
    for sid in sessions:
        print(f"Running {sid}...")
        rows = run_session(
            args.sessions_dir, args.timing_csv, sid,
            args.snr_min_db, args.t_pre_ms, args.t_post_ms,
            args.search_window_bins, args.impact_extra_offset_ms,
            args.t_min_ms,
        )
        all_rows.extend(rows)
        s = summarize(rows)
        if s["n_detected"]:
            print(f"  n={int(s['n'])}  detected={int(s['n_detected'])}  "
                  f"MAE={s['mae_deg']:.3f}  bias={s['bias_deg']:+.3f}  "
                  f"RMSE={s['rmse_deg']:.3f}  "
                  f"retention<=8={int(s['retention_le_8'])}  "
                  f"retention<=2={int(s['retention_le_2'])}")

    # Per-shot detail.
    out = args.output_dir / "per_shot_trajfit.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session", "shot_number", "club", "tm_launch_v_deg",
            "estimated_alpha_deg", "error_deg", "fit_residual_deg",
            "n_frames_used", "frames_t_ms", "frames_beta_deg",
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
