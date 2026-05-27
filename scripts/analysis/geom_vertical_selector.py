#!/usr/bin/env python3
"""Geometry-aware vertical launch selector.

Per frame, solves for launch angle α directly using the geometric model:

  Ball at impact:  (x=d_initial, y=0) relative to radar
  Ball at time t:  (x = d_initial + v·cos(α)·t,
                    y = v·sin(α)·t)
  Elevation:       η(t) = arctan(y(t) / x(t))
  K-LD7 bearing:   β(t) = η(t) - mount_deg

Solving for α given observed β at known (t, v, d_initial, mount):

  K = tan(β + mount)
  K·(d_initial + v·cos(α)·t) = v·sin(α)·t
  sin(α) - K·cos(α) = K·d_initial / (v·t)
  α = arctan(K) + arcsin(K·d_initial / (v·t·sqrt(1+K²)))

Then median across accepted frames (gated by OPS-bin match, time window,
and physical α range).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from openflight.kld7.radc import RADC_PAYLOAD_BYTES, radc_frame_diagnostics  # noqa: E402


# Per-session geometry. For sessions without measured geometry, skip them.
# (d_initial_ft, mount_deg, ball_above_radar_ft)
SESSION_GEOMETRY: dict[str, tuple[float, float, float]] = {
    "20260523_143732": (6.0, 18.0, 0.0),
    "20260523_144415": (5.0, 18.0, 0.0),
}

# Maps short session id -> long catalog name used in terminal timing CSV.
SESSION_LONG_NAMES: dict[str, str] = {
    "20260523_143732": "20260523_143732_18deg_7iron_8shots",
    "20260523_144415": "20260523_144415_18deg_7iron_5shots_cleaned",
}

# Maps short id -> (comparison_filename, jsonl_id).
SESSION_FILES: dict[str, tuple[str, str]] = {
    "20260523_143732": ("comparison_20260523_143732.csv", "20260523_143732"),
    "20260523_144415": ("comparison_20260523_144415.csv", "20260523_144415"),
}

MPH_TO_FTS = 1.46667
DEG = math.degrees
RAD = math.radians

# Plausible launch-angle band (irons + driver). Per-frame α estimates outside
# this range are clutter; reject before aggregation.
ALPHA_BAND_DEG = (-5.0, 45.0)


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
    n_after_ops_lock: int
    n_after_time_gate: int
    n_after_alpha_band: int


def _to_float(x: object) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_ts(s: str) -> float | None:
    """Parse 2026-05-22 13:58:13.310000 → float epoch seconds."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


def load_first_byte_times(csv_path: Path, long_session_name: str) -> dict[int, float]:
    """Return {json_shot_no -> first_byte_ts_epoch} for the named session."""
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


def _decode_frame(frame: dict) -> dict:
    out = dict(frame)
    radc_b64 = frame.get("radc_b64")
    if radc_b64 is not None and isinstance(radc_b64, str):
        try:
            radc = base64.b64decode(radc_b64, validate=True)
        except ValueError:
            return out
        if len(radc) == RADC_PAYLOAD_BYTES:
            out["radc"] = radc
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


def alpha_from_frame(
    bearing_deg: float,
    t_after_impact_s: float,
    ball_speed_mph: float,
    d_initial_ft: float,
    mount_deg: float,
    ball_above_radar_ft: float,
) -> float | None:
    """Solve closed-form for α given one frame's bearing observation.

    Handles the dh=0 case (ball at radar height) and the more general
    case (ball below/above radar). Returns None if the geometry has no
    real-valued solution (typically t too small or bearing physically
    impossible).
    """
    if t_after_impact_s <= 0:
        return None

    v_fts = ball_speed_mph * MPH_TO_FTS
    if v_fts <= 0:
        return None

    K = math.tan(RAD(bearing_deg + mount_deg))
    # General form:
    #   sin(α) - K·cos(α) = (K·d_initial + (-dh)) / (v·t)
    # where dh = h_radar - h_ball (positive if radar above ball).
    # ball_above_radar = -dh, so:
    #   sin(α) - K·cos(α) = (K·d_initial + ball_above_radar) / (v·t)
    rhs = (K * d_initial_ft + ball_above_radar_ft) / (v_fts * t_after_impact_s)
    denom = math.sqrt(1.0 + K * K)
    arg = rhs / denom
    if not -1.0 <= arg <= 1.0:
        return None
    phi = math.atan(K)
    return DEG(phi + math.asin(arg))


def process_shot(
    frames: list[dict],
    impact_ts: float,
    ball_speed_mph: float,
    d_initial_ft: float,
    mount_deg: float,
    ball_above_radar_ft: float,
    ops_bin_tol: int,
    t_window: tuple[float, float],
) -> tuple[float | None, dict]:
    """Return (alpha_estimate, stats) by aggregating per-frame α estimates."""
    n_total = len(frames)
    alpha_estimates: list[float] = []
    n_ops_lock = 0
    n_time_gate = 0

    for idx, raw_frame in enumerate(frames):
        frame = _decode_frame(raw_frame)
        frame_ts = frame.get("timestamp")
        if frame_ts is None:
            continue
        t = float(frame_ts) - impact_ts
        if not (t_window[0] <= t <= t_window[1]):
            continue
        n_time_gate += 1

        diag = radc_frame_diagnostics(
            frame, frame_index=idx,
            ops243_ball_speed_mph=ball_speed_mph, orientation="vertical",
        )
        if not diag.has_radc or not diag.valid_payload:
            continue
        if diag.angle_peak_deg is None:
            continue
        if diag.bin_error is None or diag.bin_error > ops_bin_tol:
            continue
        n_ops_lock += 1

        alpha = alpha_from_frame(
            bearing_deg=diag.angle_peak_deg,
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
        alpha_estimates.append(alpha)

    if not alpha_estimates:
        return None, {
            "n_frames_total": n_total,
            "n_after_ops_lock": n_ops_lock,
            "n_after_time_gate": n_time_gate,
            "n_after_alpha_band": 0,
        }

    sorted_alphas = sorted(alpha_estimates)
    m = len(sorted_alphas)
    median = (
        sorted_alphas[m // 2] if m % 2 == 1
        else 0.5 * (sorted_alphas[m // 2 - 1] + sorted_alphas[m // 2])
    )
    return median, {
        "n_frames_total": n_total,
        "n_after_ops_lock": n_ops_lock,
        "n_after_time_gate": n_time_gate,
        "n_after_alpha_band": len(alpha_estimates),
    }


def run_session(
    sessions_dir: Path,
    timing_csv: Path,
    session_id: str,
    ops_bin_tol: int,
    t_window: tuple[float, float],
    impact_offset_s: float,
) -> list[ShotResult]:
    geom = SESSION_GEOMETRY[session_id]
    d_initial_ft, mount_deg, ball_above_radar_ft = geom

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
            print(f"  shot {shot_n}: no first_byte_ts in timing CSV — skipping")
            continue
        impact_ts = fb_ts + impact_offset_s

        alpha, stats = process_shot(
            frames=frames,
            impact_ts=impact_ts,
            ball_speed_mph=tgt["ball_speed_mph"],
            d_initial_ft=d_initial_ft,
            mount_deg=mount_deg,
            ball_above_radar_ft=ball_above_radar_ft,
            ops_bin_tol=ops_bin_tol,
            t_window=t_window,
        )
        err = (alpha - tgt["tm_launch_v_deg"]) if alpha is not None else None
        results.append(ShotResult(
            session=session_id,
            shot_number=shot_n,
            club=tgt["club"],
            tm_launch_v_deg=tgt["tm_launch_v_deg"],
            estimated_alpha_deg=alpha,
            error_deg=err,
            impact_ts=impact_ts,
            n_frames_total=stats["n_frames_total"],
            n_after_ops_lock=stats["n_after_ops_lock"],
            n_after_time_gate=stats["n_after_time_gate"],
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
        "p90_abs_deg": sorted(abs_errs)[int(math.ceil(0.9 * n) - 1)] if n else float("nan"),
        "retention_le_8": sum(1 for e in abs_errs if e <= 8.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path("/Users/john.pacino/openflight_sessions"),
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/_geom"))
    parser.add_argument("--ops-bin-tol", type=int, default=25)
    parser.add_argument(
        "--t-window", type=float, nargs=2, default=[0.05, 3.0],
        help="Min and max time-after-impact (s) for frames to be used.",
    )
    parser.add_argument(
        "--impact-offset-s", type=float, default=-0.068,
        help="Offset added to first_byte_ts to estimate true impact. The first byte "
             "arrives ~68ms after the hardware trigger, so the default of -0.068 maps "
             "back to the trigger epoch.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[ShotResult] = []
    for sid in SESSION_GEOMETRY.keys():
        print(f"Running {sid}...")
        rows = run_session(
            args.sessions_dir, args.timing_csv, sid,
            args.ops_bin_tol, tuple(args.t_window), args.impact_offset_s,
        )
        all_rows.extend(rows)
        s = summarize(rows)
        if s["n_detected"]:
            print(f"  n={int(s['n'])}  detected={int(s['n_detected'])}  "
                  f"MAE={s['mae_deg']:.3f}  bias={s['bias_deg']:+.3f}  "
                  f"RMSE={s['rmse_deg']:.3f}  retention<=8={int(s['retention_le_8'])}")

    # Per-shot detail.
    out = args.output_dir / "per_shot_geom.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session", "shot_number", "club", "tm_launch_v_deg",
            "estimated_alpha_deg", "error_deg", "impact_ts",
            "n_frames_total", "n_after_ops_lock", "n_after_time_gate", "n_after_alpha_band",
        ])
        for r in all_rows:
            w.writerow([
                r.session, r.shot_number, r.club, f"{r.tm_launch_v_deg:.2f}",
                "" if r.estimated_alpha_deg is None else f"{r.estimated_alpha_deg:.2f}",
                "" if r.error_deg is None else f"{r.error_deg:.2f}",
                f"{r.impact_ts:.3f}",
                r.n_frames_total, r.n_after_ops_lock, r.n_after_time_gate, r.n_after_alpha_band,
            ])

    overall = summarize(all_rows)
    print("\n=== OVERALL ===")
    if overall["n_detected"]:
        print(f"  n={int(overall['n'])}  detected={int(overall['n_detected'])}  "
              f"MAE={overall['mae_deg']:.3f}  bias={overall['bias_deg']:+.3f}  "
              f"RMSE={overall['rmse_deg']:.3f}  P90={overall['p90_abs_deg']:.3f}")
    print(f"\nWrote per-shot results to {out}")


if __name__ == "__main__":
    main()
