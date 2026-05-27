#!/usr/bin/env python3
"""Tier 1 deterministic vertical launch selector — no training, runtime-only features.

Pipeline per shot:
  1. OPS-bin lock: keep frames where dominant Doppler bin is within ±N bins
     of the OPS-predicted bin for the measured ball speed.
  2. Mount-aware angle band: hard-reject raw angles outside a conservative
     mount-specific window.
  3. Trimmed mean: drop top/bottom T% of remaining frame angles, mean the rest.

Compares against TrackMan vertical launch from the comparison CSVs.

Usage:
    uv run --no-project --with numpy python scripts/analysis/tier1_vertical_selector.py \\
        --sessions-dir /Users/john.pacino/openflight_sessions \\
        --output-dir /tmp/_tier1
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from openflight.kld7.radc import RADC_PAYLOAD_BYTES, radc_frame_diagnostics  # noqa: E402


def _decode_frame(frame: dict) -> dict:
    """Return a copy of `frame` with `radc_b64` base64-decoded into `radc` bytes."""
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


# Session ID -> (mount_deg, comparison_filename, session_jsonl_id)
# Some sessions have multiple comparison files (one per club for multi-club sessions);
# the JSONL is the same. The "session_id" used as a key here is for catalog
# uniqueness; the jsonl path uses the third element.
SESSION_CATALOG: dict[str, tuple[float, str, str]] = {
    "20260522_135647":          (0.0,  "comparison_20260522_135647.csv",          "20260522_135647"),
    "20260522_141038":          (8.0,  "comparison_20260522_141038.csv",          "20260522_141038"),
    "20260522_141949":          (18.0, "comparison_20260522_141949.csv",          "20260522_141949"),
    "20260522_142538__7iron":   (0.0,  "comparison_20260522_142538_7iron.csv",    "20260522_142538"),
    "20260522_142538__8iron":   (0.0,  "comparison_20260522_142538_8iron.csv",    "20260522_142538"),
    "20260522_142538__9iron":   (0.0,  "comparison_20260522_142538_9iron.csv",    "20260522_142538"),
    "20260522_142538__driver":  (0.0,  "comparison_20260522_142538_driver.csv",   "20260522_142538"),
    "20260523_143732":          (18.0, "comparison_20260523_143732.csv",          "20260523_143732"),
    "20260523_144415":          (18.0, "comparison_20260523_144415.csv",          "20260523_144415"),
}


# Mount-aware hard-gate windows for vertical raw angle.
# Reject any frame whose raw angle falls outside [lo, hi].
# Windows are deliberately conservative — meant to exclude clutter spikes,
# not to constrain the launch angle.
MOUNT_BANDS: dict[int, tuple[float, float]] = {
    0: (-2.0, 25.0),
    8: (-8.0, 20.0),
    18: (-15.0, 12.0),
}

# Mount-aware center of the expected raw bearing for typical golf launches
# (~15-20° TM launch). Used as a tiebreaker when accepted angles split into
# two distinct clusters: pick the cluster whose mean is closest to this
# center. Derived from raw = TM_launch - (mount + 8.0) with TM_launch ≈ 18.
MOUNT_RAW_CENTERS: dict[int, float] = {
    0: 10.0,
    8: 2.0,
    18: -8.0,
}


@dataclass
class FrameResult:
    timestamp: float | None
    angle_raw_deg: float | None
    angle_centroid_deg: float | None
    snr_db: float
    bin_error: int | None
    peak_bin: int | None
    expected_bin: int | None
    rejected_reason: str | None


@dataclass
class ShotResult:
    session: str
    shot_number: int
    club: str
    mount_deg: float
    ball_speed_mph: float
    tm_launch_v_deg: float
    selected_angle_deg: float | None
    error_deg: float | None
    n_frames_total: int
    n_after_ops_lock: int
    n_after_band_gate: int
    n_after_trim: int
    rejection_reasons: dict[str, int]


def _to_float(x: object) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_comparison(path: Path) -> dict[int, dict]:
    """Map of-shot-number -> {ball_speed, tm_v, club} for good vertical rows."""
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
    """Yield (shot_number, frames_list) for each vertical kld7_buffer."""
    with jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "kld7_buffer":
                continue
            if d.get("orientation") != "vertical":
                continue
            shot_n = d.get("shot_number")
            frames = d.get("frames") or []
            if shot_n is None or not frames:
                continue
            yield int(shot_n), frames


def process_shot(
    frames: list[dict],
    ball_speed_mph: float,
    mount_deg: float,
    ops_bin_tol: int,
    trim_fraction: float,
) -> tuple[float | None, dict, list[FrameResult]]:
    """Apply Tier 1 filters and return (selected_angle, stats, per-frame details)."""
    rejection: dict[str, int] = {
        "no_radc": 0,
        "no_angle": 0,
        "ops_bin": 0,
        "angle_band": 0,
        "trimmed": 0,
    }
    accepted_angles: list[tuple[float, float]] = []  # (angle_deg, snr_db)
    frame_results: list[FrameResult] = []

    band_lo, band_hi = MOUNT_BANDS[int(round(mount_deg))]

    for idx, raw_frame in enumerate(frames):
        frame = _decode_frame(raw_frame)
        diag = radc_frame_diagnostics(
            frame,
            frame_index=idx,
            ops243_ball_speed_mph=ball_speed_mph,
            orientation="vertical",
        )
        angle = diag.angle_peak_deg

        if not diag.has_radc or not diag.valid_payload:
            rejection["no_radc"] += 1
            frame_results.append(FrameResult(
                diag.timestamp, None, None, diag.snr_db,
                diag.bin_error, diag.peak_bin, diag.expected_bin,
                "no_radc",
            ))
            continue

        if angle is None:
            rejection["no_angle"] += 1
            frame_results.append(FrameResult(
                diag.timestamp, None, None, diag.snr_db,
                diag.bin_error, diag.peak_bin, diag.expected_bin,
                "no_angle",
            ))
            continue

        # Filter 1: OPS-bin lock.
        if diag.bin_error is None or diag.bin_error > ops_bin_tol:
            rejection["ops_bin"] += 1
            frame_results.append(FrameResult(
                diag.timestamp, angle, diag.angle_centroid_deg, diag.snr_db,
                diag.bin_error, diag.peak_bin, diag.expected_bin,
                "ops_bin",
            ))
            continue

        # Filter 2: mount-aware angle band hard-gate.
        if not (band_lo <= angle <= band_hi):
            rejection["angle_band"] += 1
            frame_results.append(FrameResult(
                diag.timestamp, angle, diag.angle_centroid_deg, diag.snr_db,
                diag.bin_error, diag.peak_bin, diag.expected_bin,
                "angle_band",
            ))
            continue

        accepted_angles.append((angle, diag.snr_db))
        frame_results.append(FrameResult(
            diag.timestamp, angle, diag.angle_centroid_deg, diag.snr_db,
            diag.bin_error, diag.peak_bin, diag.expected_bin,
            None,
        ))

    n_after_lock_and_band = len(accepted_angles)

    # Filter 3: trimmed median of raw bearings.
    #
    # We explored cluster-aware variants (lower-cluster preference, mount-prior
    # tiebreaker, SNR-based tiebreaker) and all of them improved the bimodal
    # failure mode in the 14:44 session but hurt other sessions more than they
    # helped. The trimmed median is the best aggregator without multi-candidate
    # generation; the 14:44 failure mode (ball signal as minority cluster) is
    # a known residual that needs candidate-pool techniques to fully resolve.
    selected_raw: float | None = None
    n_after_trim = 0
    if accepted_angles:
        sorted_angles = sorted(a for a, _ in accepted_angles)
        n = len(sorted_angles)
        trim_count = int(math.floor(n * trim_fraction))
        kept = (
            sorted_angles[trim_count: n - trim_count]
            if (n - 2 * trim_count) > 0
            else sorted_angles
        )
        rejection["trimmed"] = n - len(kept)
        m = len(kept)
        selected_raw = (
            kept[m // 2]
            if m % 2 == 1
            else 0.5 * (kept[m // 2 - 1] + kept[m // 2])
        )
        n_after_trim = m

    # Convert raw bearing -> launch angle using effective offset (mount + boresight).
    # Matches the offline analysis convention: effective_offset = mount_deg + 8.0.
    selected_angle = (selected_raw + mount_deg + 8.0) if selected_raw is not None else None

    stats = {
        "n_frames_total": len(frames),
        "n_after_ops_lock": len(frames) - rejection["no_radc"] - rejection["no_angle"] - rejection["ops_bin"],
        "n_after_band_gate": n_after_lock_and_band,
        "n_after_trim": n_after_trim,
        "rejections": rejection,
        "raw_band_lo": band_lo,
        "raw_band_hi": band_hi,
        "mount_deg": mount_deg,
    }
    return selected_angle, stats, frame_results


def run_session(
    sessions_dir: Path,
    catalog_key: str,
    mount_deg: float,
    comparison_filename: str,
    jsonl_session_id: str,
    ops_bin_tol: int,
    trim_fraction: float,
) -> list[ShotResult]:
    jsonl = sessions_dir / f"session_{jsonl_session_id}_range.jsonl"
    comp = sessions_dir / comparison_filename
    if not jsonl.exists() or not comp.exists():
        print(f"  skip {catalog_key}: missing {jsonl.name} or {comp.name}")
        return []

    targets = load_comparison(comp)
    results: list[ShotResult] = []

    for shot_n, frames in iter_vertical_buffers(jsonl):
        tgt = targets.get(shot_n)
        if tgt is None:
            continue
        selected, stats, _frames = process_shot(
            frames,
            tgt["ball_speed_mph"],
            mount_deg,
            ops_bin_tol,
            trim_fraction,
        )
        err = (selected - tgt["tm_launch_v_deg"]) if selected is not None else None
        results.append(ShotResult(
            session=catalog_key,
            shot_number=shot_n,
            club=tgt["club"],
            mount_deg=mount_deg,
            ball_speed_mph=tgt["ball_speed_mph"],
            tm_launch_v_deg=tgt["tm_launch_v_deg"],
            selected_angle_deg=selected,
            error_deg=err,
            n_frames_total=stats["n_frames_total"],
            n_after_ops_lock=stats["n_after_ops_lock"],
            n_after_band_gate=stats["n_after_band_gate"],
            n_after_trim=stats["n_after_trim"],
            rejection_reasons=stats["rejections"],
        ))
    return results


def summarize(rows: list[ShotResult]) -> dict[str, float]:
    detected = [r for r in rows if r.error_deg is not None]
    if not detected:
        return {"n": 0, "n_detected": 0}
    errs = [r.error_deg for r in detected]
    abs_errs = [abs(e) for e in errs]
    n = len(detected)
    mae = sum(abs_errs) / n
    bias = sum(errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    abs_sorted = sorted(abs_errs)
    p90 = abs_sorted[int(math.ceil(0.9 * n) - 1)] if n else float("nan")
    retained = sum(1 for e in abs_errs if e <= 8.0)
    return {
        "n": float(len(rows)),
        "n_detected": float(n),
        "mae_deg": mae,
        "bias_deg": bias,
        "rmse_deg": rmse,
        "p90_abs_deg": p90,
        "retention_le_8": retained,
        "retention_pct": 100.0 * retained / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path,
                        default=Path("/Users/john.pacino/openflight_sessions"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/_tier1"))
    parser.add_argument("--ops-bin-tol", type=int, default=25,
                        help="Max bin-error from OPS-predicted bin (default 25)")
    parser.add_argument("--trim-fraction", type=float, default=0.25,
                        help="Top/bottom fraction trimmed from accepted angles (default 0.25)")
    parser.add_argument("--session-filter", nargs="*",
                        help="Only run these session IDs (default all in catalog)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions = list(SESSION_CATALOG.items())
    if args.session_filter:
        sessions = [s for s in sessions if s[0] in args.session_filter]

    all_rows: list[ShotResult] = []
    for catalog_key, (mount_deg, comp_name, jsonl_id) in sessions:
        print(f"Running {catalog_key} (mount={mount_deg}°)...")
        rows = run_session(
            args.sessions_dir, catalog_key, mount_deg, comp_name, jsonl_id,
            args.ops_bin_tol, args.trim_fraction,
        )
        all_rows.extend(rows)
        s = summarize(rows)
        if s.get("n_detected"):
            print(f"  n={int(s['n'])}  detected={int(s['n_detected'])}  "
                  f"MAE={s['mae_deg']:.3f}  bias={s['bias_deg']:+.3f}  "
                  f"RMSE={s['rmse_deg']:.3f}  retention={s['retention_pct']:.1f}%")

    # Write per-shot CSV.
    per_shot_path = args.output_dir / "per_shot_tier1.csv"
    with per_shot_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session", "shot_number", "club", "mount_deg", "ball_speed_mph",
            "tm_launch_v_deg", "selected_angle_deg", "error_deg",
            "n_frames_total", "n_after_ops_lock", "n_after_band_gate", "n_after_trim",
            "rej_no_radc", "rej_no_angle", "rej_ops_bin", "rej_angle_band", "rej_trimmed",
        ])
        for r in all_rows:
            w.writerow([
                r.session, r.shot_number, r.club, r.mount_deg, f"{r.ball_speed_mph:.2f}",
                f"{r.tm_launch_v_deg:.2f}",
                "" if r.selected_angle_deg is None else f"{r.selected_angle_deg:.2f}",
                "" if r.error_deg is None else f"{r.error_deg:.2f}",
                r.n_frames_total, r.n_after_ops_lock, r.n_after_band_gate, r.n_after_trim,
                r.rejection_reasons["no_radc"], r.rejection_reasons["no_angle"],
                r.rejection_reasons["ops_bin"], r.rejection_reasons["angle_band"],
                r.rejection_reasons["trimmed"],
            ])

    # Summaries.
    overall = summarize(all_rows)
    by_mount = {
        m: summarize([r for r in all_rows if int(round(r.mount_deg)) == m])
        for m in sorted({int(round(r.mount_deg)) for r in all_rows})
    }
    by_session = {
        s: summarize([r for r in all_rows if r.session == s])
        for s in sorted({r.session for r in all_rows})
    }

    print("\n=== OVERALL ===")
    if overall.get("n_detected"):
        print(f"  n={int(overall['n'])}  detected={int(overall['n_detected'])}  "
              f"MAE={overall['mae_deg']:.3f}  bias={overall['bias_deg']:+.3f}  "
              f"RMSE={overall['rmse_deg']:.3f}  P90={overall['p90_abs_deg']:.3f}  "
              f"retention={overall['retention_pct']:.1f}%")

    print("\n=== BY MOUNT ===")
    for m, s in by_mount.items():
        if s.get("n_detected"):
            print(f"  mount={m}°  n={int(s['n'])}  detected={int(s['n_detected'])}  "
                  f"MAE={s['mae_deg']:.3f}  bias={s['bias_deg']:+.3f}  "
                  f"P90={s['p90_abs_deg']:.3f}  retention={s['retention_pct']:.1f}%")

    print("\n=== BY SESSION ===")
    for sid, s in by_session.items():
        if s.get("n_detected"):
            print(f"  {sid}  n={int(s['n'])}  detected={int(s['n_detected'])}  "
                  f"MAE={s['mae_deg']:.3f}  bias={s['bias_deg']:+.3f}  "
                  f"P90={s['p90_abs_deg']:.3f}  retention={s['retention_pct']:.1f}%")

    print(f"\nWrote per-shot results to {per_shot_path}")


if __name__ == "__main__":
    main()
