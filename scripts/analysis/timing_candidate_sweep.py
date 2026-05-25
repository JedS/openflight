#!/usr/bin/env python3
"""Timing-aware candidate-pool rerun from raw kld7_buffer frames.

Builds per-shot candidate pools by re-running extract_launch_angle() over
multiple timing anchors and RADC parameters, then compares:
1) TM-oracle selector (analysis ceiling only)
2) Mount-aware TM-free pseudo selector

This is an offline analysis script and does not modify runtime behavior.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openflight.kld7.radc import extract_launch_angle

KNOWN_DISTANCE_FT = {
    "20260523_143732_18deg_7iron_8shots": 6.0,
    "20260523_144415_18deg_7iron_5shots_cleaned": 5.0,
}


@dataclass(frozen=True)
class SessionSpec:
    session_dir: str
    mount_deg: float
    session_path: Path
    comparison_csvs: list[Path]
    distance_ft: float
    distance_source: str


@dataclass(frozen=True)
class ShotLabel:
    session_dir: str
    comparison_file: str
    shot_number: int
    club: str
    mount_deg: float
    ops_ball_speed_mph: float
    tm_launch_v_deg: float
    distance_ft: float
    distance_source: str


@dataclass(frozen=True)
class ShotTiming:
    first_byte_ts: float | None
    accept_ts: float | None
    shot_log_ts: float | None
    callback_ms: float | None
    total_ms: float | None


def _to_float(value: str | None, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_dt(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    # Strict file uses "YYYY-MM-DD HH:MM:SS.ssssss"
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def _mae(values: list[float]) -> float:
    return float(sum(abs(v) for v in values) / len(values)) if values else float("nan")


def _rmse(values: list[float]) -> float:
    return float(math.sqrt(sum(v * v for v in values) / len(values))) if values else float("nan")


def _p90_abs(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = sorted(abs(v) for v in values)
    idx = int((len(arr) - 1) * 0.9)
    return float(arr[idx])


def discover_sessions(root: Path) -> list[SessionSpec]:
    specs: list[SessionSpec] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and re.match(r"^\d{8}_\d{6}_", p.name)):
        m = re.search(r"_(\d+)deg_", d.name)
        if not m:
            continue
        mount = float(m.group(1))
        session_files = sorted(d.glob("session_*.jsonl"))
        if not session_files:
            continue
        comparison_files = sorted(d.glob("comparison_*.csv"))
        if not comparison_files:
            continue
        distance_ft = KNOWN_DISTANCE_FT.get(d.name, 6.0)
        distance_source = "known" if d.name in KNOWN_DISTANCE_FT else "assumed_6ft"
        specs.append(
            SessionSpec(
                session_dir=d.name,
                mount_deg=mount,
                session_path=session_files[0],
                comparison_csvs=comparison_files,
                distance_ft=distance_ft,
                distance_source=distance_source,
            )
        )
    return specs


def load_labels(specs: list[SessionSpec]) -> list[ShotLabel]:
    out: list[ShotLabel] = []
    for spec in specs:
        for comp in spec.comparison_csvs:
            with comp.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("match_quality") != "good":
                        continue
                    tm = _to_float(row.get("launch_v_tm"))
                    if tm is None:
                        continue
                    shot = int(float(row["shot_number_of"]))
                    out.append(
                        ShotLabel(
                            session_dir=spec.session_dir,
                            comparison_file=comp.name,
                            shot_number=shot,
                            club=row.get("club", "unknown"),
                            mount_deg=spec.mount_deg,
                            ops_ball_speed_mph=float(row["ball_speed_of"]),
                            tm_launch_v_deg=tm,
                            distance_ft=spec.distance_ft,
                            distance_source=spec.distance_source,
                        )
                    )
    return out


def load_timing_map(path: Path) -> dict[tuple[str, int], ShotTiming]:
    out: dict[tuple[str, int], ShotTiming] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            session = row["session"]
            shot = int(float(row["json_shot_no"]))
            out[(session, shot)] = ShotTiming(
                first_byte_ts=_parse_dt(row.get("first_byte_ts")),
                accept_ts=_parse_dt(row.get("accept_ts")),
                shot_log_ts=_parse_dt(row.get("json_shot_ts")),
                callback_ms=_to_float(row.get("callback_ms")),
                total_ms=_to_float(row.get("total_ms")),
            )
    return out


def load_baseline_csv(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    out: dict[tuple[str, str, int], dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            tm = _to_float(row.get("trackman_launch_v_deg"))
            if tm is None:
                continue
            key = (row["session_dir"], row["comparison_file"], int(float(row["shot_number"])))
            out[key] = {
                "latest_pred_v_deg": float(row["latest_pred_v_deg"]),
                "fixed_pred_v_deg": float(row["fixed_pred_v_deg"]),
                "oracle_pred_v_deg": float(row["oracle_pred_v_deg"]),
                "latest_raw_angle_deg": float(row["latest_raw_angle_deg"]),
                "fixed_raw_angle_deg": float(row["fixed_raw_angle_deg"]),
                "candidate_count": float(row.get("candidate_count") or 0.0),
            }
    return out


def load_vertical_buffers(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") != "kld7_buffer":
                continue
            if rec.get("orientation") != "vertical":
                continue
            shot_no = int(rec.get("shot_number", 0))
            if shot_no <= 0:
                continue
            frames = rec.get("frames") or []
            decoded: list[dict[str, Any]] = []
            for fr in frames:
                if not fr.get("has_radc"):
                    continue
                b64 = fr.get("radc_b64")
                if not b64:
                    continue
                try:
                    payload = base64.b64decode(b64)
                except (ValueError, TypeError):
                    continue
                if len(payload) != 3072:
                    continue
                decoded.append(
                    {
                        "timestamp": float(fr["timestamp"]),
                        "radc": payload,
                    }
                )
            out[shot_no] = {
                "shot_timestamp": float(rec.get("shot_timestamp", 0.0)),
                "frames": decoded,
                "frame_count": int(rec.get("frame_count", len(frames))),
            }
    return out


def _mount_raw_prior_center(mount_deg: float, speed_mph: float) -> tuple[float, float]:
    m = int(round(mount_deg))
    if m >= 16:
        return (-8.0, 2.8)
    if m >= 6:
        return (1.0, 2.3)
    center = 9.0
    if speed_mph > 145.0:
        center -= 2.0
    elif speed_mph > 130.0:
        center -= 1.0
    return (center, 3.2)


def pseudo_score(candidate: dict[str, float], mount_deg: float, ops_speed_mph: float) -> float:
    pred = float(candidate["launch_angle_deg"])
    raw = float(candidate["raw_angle_deg"])
    conf = float(candidate.get("confidence", 0.0))
    snr = float(candidate.get("avg_snr_db", 0.0))
    frames = float(candidate.get("frame_count", 1.0))
    ball = float(candidate.get("ball_speed_mph", ops_speed_mph))
    speed_err = abs(ball - ops_speed_mph)

    center, sigma = _mount_raw_prior_center(mount_deg, ops_speed_mph)
    z = (raw - center) / max(sigma, 1e-6)
    score = -(z * z)
    score += 0.9 * (conf - 0.5)
    score += 0.5 * min(snr / 10.0, 1.0)
    score += 0.08 * min(frames, 3.0)
    score -= 0.04 * speed_err

    if mount_deg >= 16.0:
        if pred > 24.0:
            score -= 2.2
        if pred < 9.0:
            score -= 1.2
        if raw > -1.0:
            score -= 0.8
    elif mount_deg >= 6.0:
        if pred > 24.0:
            score -= 1.8
        if pred < 8.0:
            score -= 1.1
        if raw > 5.0:
            score -= 0.9
    else:
        if pred > 30.0:
            score -= 2.0
        if pred < 4.0:
            score -= 1.3
        if raw < 1.0:
            score -= 0.8
    return score


def build_impact_anchors(timing: ShotTiming | None, fallback_shot_ts: float) -> list[float]:
    anchors: list[float] = []
    if timing is not None:
        if timing.first_byte_ts is not None:
            anchors.extend(
                [
                    timing.first_byte_ts - 0.12,
                    timing.first_byte_ts - 0.06,
                    timing.first_byte_ts,
                    timing.first_byte_ts + 0.06,
                ]
            )
        if timing.accept_ts is not None:
            anchors.extend(
                [
                    timing.accept_ts - 4.7,
                ]
            )
    anchors.extend([fallback_shot_ts - 4.9])
    # Preserve order, remove dup-ish anchors.
    out: list[float] = []
    for a in anchors:
        if not out or abs(a - out[-1]) > 1e-4:
            out.append(a)
    return out


def build_candidates(
    frames: list[dict[str, Any]],
    anchors: list[float],
    ops_speed_mph: float,
    lateral_in: float,
    distance_ft: float,
) -> list[dict[str, Any]]:
    impact_thresholds = [2.2, 3.0]
    speed_tolerances = [8.0, 10.0]
    centroid_fracs = [0.45, 0.55]
    outlier_tols = [15, 25]

    seen: set[tuple[int, int, int, int]] = set()
    out: list[dict[str, Any]] = []

    for impact_ts in anchors:
        for thr in impact_thresholds:
            for tol in speed_tolerances:
                for frac in centroid_fracs:
                    for out_tol in outlier_tols:
                        shots = extract_launch_angle(
                            frames=frames,
                            fft_size=2048,
                            max_speed_kmh=100.0,
                            cfar_threshold=2.5,
                            impact_energy_threshold=thr,
                            angle_offset_deg=0.0,
                            ops243_ball_speed_mph=ops_speed_mph,
                            speed_tolerance_mph=tol,
                            orientation="vertical",
                            ops_bin_outlier_tol=out_tol,
                            ops_bin_outlier_penalty=10.0,
                            centroid_floor_frac=frac,
                            ball_lateral_offset_in=lateral_in,
                            ball_initial_range_in=distance_ft * 12.0,
                            impact_timestamp=impact_ts,
                        )
                        for s in shots:
                            key = (
                                int(round(float(s["launch_angle_deg"]) * 10)),
                                int(round(float(s["raw_angle_deg"]) * 10)),
                                int(round(float(s.get("confidence", 0.0)) * 100)),
                                int(float(s.get("frame_count", 0))),
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            rec = dict(s)
                            rec["sweep_impact_ts"] = impact_ts
                            rec["sweep_thr"] = thr
                            rec["sweep_tol"] = tol
                            rec["sweep_frac"] = frac
                            rec["sweep_out_tol"] = out_tol
                            out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize(rows: list[dict[str, Any]], pred_key: str, delta_key: str) -> dict[str, float]:
    deltas = [float(r[delta_key]) for r in rows if r.get(pred_key) not in (None, "")]
    retained = sum(1 for d in deltas if abs(d) <= 8.0)
    return {
        "n": float(len(deltas)),
        "mae": _mae(deltas),
        "bias": float(sum(deltas) / len(deltas)) if deltas else float("nan"),
        "rmse": _rmse(deltas),
        "p90_abs": _p90_abs(deltas),
        "retained_le8": float(retained),
        "ret_pct": (retained / len(deltas) * 100.0) if deltas else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path("~/Desktop/openflight_sessions").expanduser(),
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=Path(
            "~/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"
        ).expanduser(),
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "~/Desktop/openflight_sessions/_analysis_fixed_not_high_else_grid_oracle/per_shot_vertical_deltas_latest_fixed_oracle.csv"
        ).expanduser(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/Desktop/openflight_sessions/_analysis_timing_candidate_sweep").expanduser(),
    )
    parser.add_argument("--lateral-in", type=float, default=4.0)
    args = parser.parse_args()
    logging.getLogger("openflight.kld7.radc").setLevel(logging.ERROR)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_sessions(args.sessions_root)
    labels = load_labels(specs)
    timing = load_timing_map(args.timing_csv)
    baseline = load_baseline_csv(args.baseline_csv)

    buffers_by_session = {
        s.session_dir: load_vertical_buffers(s.session_path)
        for s in specs
    }

    per_shot_rows: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []

    for label in labels:
        bkey = (label.session_dir, label.comparison_file, label.shot_number)
        base = baseline.get(bkey)
        if base is None:
            continue

        shot_map = buffers_by_session.get(label.session_dir, {})
        buf = shot_map.get(label.shot_number)
        if not buf:
            continue
        frames: list[dict[str, Any]] = buf["frames"]
        if not frames:
            continue

        timing_row = timing.get((label.session_dir, label.shot_number))
        anchors = build_impact_anchors(timing_row, fallback_shot_ts=float(buf["shot_timestamp"]))
        candidates = build_candidates(
            frames=frames,
            anchors=anchors,
            ops_speed_mph=label.ops_ball_speed_mph,
            lateral_in=args.lateral_in,
            distance_ft=label.distance_ft,
        )

        for c in candidates:
            all_candidate_rows.append(
                {
                    "session": label.session_dir,
                    "comparison_file": label.comparison_file,
                    "shot_number": label.shot_number,
                    "mount_deg": label.mount_deg,
                    "distance_ft": label.distance_ft,
                    "distance_source": label.distance_source,
                    "ops_ball_speed_mph": label.ops_ball_speed_mph,
                    "tm_launch_v_deg": label.tm_launch_v_deg,
                    "cand_pred_v_deg": c.get("launch_angle_deg"),
                    "cand_raw_angle_deg": c.get("raw_angle_deg"),
                    "cand_confidence": c.get("confidence"),
                    "cand_frame_count": c.get("frame_count"),
                    "cand_avg_snr_db": c.get("avg_snr_db"),
                    "cand_ball_speed_mph": c.get("ball_speed_mph"),
                    "sweep_impact_ts": c.get("sweep_impact_ts"),
                    "sweep_thr": c.get("sweep_thr"),
                    "sweep_tol": c.get("sweep_tol"),
                    "sweep_frac": c.get("sweep_frac"),
                    "sweep_out_tol": c.get("sweep_out_tol"),
                }
            )

        # Add current production-like candidates for fair compare in same selector.
        pseudo_pool: list[dict[str, Any]] = list(candidates)
        pseudo_pool.append(
            {
                "launch_angle_deg": base["latest_pred_v_deg"],
                "raw_angle_deg": base["latest_raw_angle_deg"],
                "confidence": 0.50,
                "frame_count": 1,
                "avg_snr_db": 5.0,
                "ball_speed_mph": label.ops_ball_speed_mph,
                "candidate_src": "latest",
            }
        )
        pseudo_pool.append(
            {
                "launch_angle_deg": base["fixed_pred_v_deg"],
                "raw_angle_deg": base["fixed_raw_angle_deg"],
                "confidence": 0.50,
                "frame_count": 1,
                "avg_snr_db": 5.0,
                "ball_speed_mph": label.ops_ball_speed_mph,
                "candidate_src": "fixed",
            }
        )

        if pseudo_pool:
            pseudo_best = max(
                pseudo_pool,
                key=lambda c: pseudo_score(c, label.mount_deg, label.ops_ball_speed_mph),
            )
            oracle_best = min(
                pseudo_pool,
                key=lambda c: abs(float(c["launch_angle_deg"]) - label.tm_launch_v_deg),
            )
            pseudo_pred = float(pseudo_best["launch_angle_deg"])
            oracle_pred = float(oracle_best["launch_angle_deg"])
            pseudo_raw = float(pseudo_best["raw_angle_deg"])
            oracle_raw = float(oracle_best["raw_angle_deg"])
        else:
            pseudo_pred = float("nan")
            oracle_pred = float("nan")
            pseudo_raw = float("nan")
            oracle_raw = float("nan")

        latest_delta = base["latest_pred_v_deg"] - label.tm_launch_v_deg
        fixed_delta = base["fixed_pred_v_deg"] - label.tm_launch_v_deg
        prev_oracle_delta = base["oracle_pred_v_deg"] - label.tm_launch_v_deg
        pseudo_delta = pseudo_pred - label.tm_launch_v_deg
        sweep_oracle_delta = oracle_pred - label.tm_launch_v_deg

        per_shot_rows.append(
            {
                "session": label.session_dir,
                "comparison_file": label.comparison_file,
                "shot_number": label.shot_number,
                "club": label.club,
                "mount_deg": label.mount_deg,
                "distance_ft": label.distance_ft,
                "distance_source": label.distance_source,
                "ops_ball_speed_mph": round(label.ops_ball_speed_mph, 3),
                "tm_launch_v_deg": label.tm_launch_v_deg,
                "latest_pred_v_deg": base["latest_pred_v_deg"],
                "fixed_pred_v_deg": base["fixed_pred_v_deg"],
                "prev_oracle_pred_v_deg": base["oracle_pred_v_deg"],
                "pseudo_pred_v_deg": pseudo_pred,
                "sweep_oracle_pred_v_deg": oracle_pred,
                "latest_delta_deg": latest_delta,
                "fixed_delta_deg": fixed_delta,
                "prev_oracle_delta_deg": prev_oracle_delta,
                "pseudo_delta_deg": pseudo_delta,
                "sweep_oracle_delta_deg": sweep_oracle_delta,
                "latest_abs_err_deg": abs(latest_delta),
                "fixed_abs_err_deg": abs(fixed_delta),
                "prev_oracle_abs_err_deg": abs(prev_oracle_delta),
                "pseudo_abs_err_deg": abs(pseudo_delta),
                "sweep_oracle_abs_err_deg": abs(sweep_oracle_delta),
                "candidate_pool_size": len(candidates),
                "pseudo_selected_raw_deg": pseudo_raw,
                "sweep_oracle_selected_raw_deg": oracle_raw,
            }
        )

    write_csv(args.output_dir / "candidate_pool_rows.csv", all_candidate_rows)
    write_csv(args.output_dir / "per_shot_timing_candidate_sweep.csv", per_shot_rows)

    overall_rows: list[dict[str, Any]] = []
    strategies = [
        ("latest", "latest_pred_v_deg", "latest_delta_deg"),
        ("fixed", "fixed_pred_v_deg", "fixed_delta_deg"),
        ("prev_oracle", "prev_oracle_pred_v_deg", "prev_oracle_delta_deg"),
        ("pseudo", "pseudo_pred_v_deg", "pseudo_delta_deg"),
        ("sweep_oracle", "sweep_oracle_pred_v_deg", "sweep_oracle_delta_deg"),
    ]
    for name, pred_k, delta_k in strategies:
        s = summarize(per_shot_rows, pred_k, delta_k)
        overall_rows.append(
            {
                "strategy": name,
                "n": int(s["n"]),
                "mae_deg": s["mae"],
                "bias_deg": s["bias"],
                "rmse_deg": s["rmse"],
                "p90_abs_deg": s["p90_abs"],
                "retained_le8": int(s["retained_le8"]),
                "retention_pct": s["ret_pct"],
            }
        )
    write_csv(args.output_dir / "overall_summary_timing_candidate_sweep.csv", overall_rows)

    per_session_rows: list[dict[str, Any]] = []
    sessions = sorted({r["session"] for r in per_shot_rows})
    for session in sessions:
        chunk = [r for r in per_shot_rows if r["session"] == session]
        row: dict[str, Any] = {"session": session, "shots": len(chunk)}
        for name, pred_k, delta_k in strategies:
            s = summarize(chunk, pred_k, delta_k)
            row[f"{name}_mae_deg"] = s["mae"]
            row[f"{name}_retention_pct"] = s["ret_pct"]
        per_session_rows.append(row)
    write_csv(args.output_dir / "session_summary_timing_candidate_sweep.csv", per_session_rows)

    per_mount_rows: list[dict[str, Any]] = []
    mounts = sorted({float(r["mount_deg"]) for r in per_shot_rows})
    for mount in mounts:
        chunk = [r for r in per_shot_rows if float(r["mount_deg"]) == mount]
        row = {"mount_deg": mount, "shots": len(chunk)}
        for name, pred_k, delta_k in strategies:
            s = summarize(chunk, pred_k, delta_k)
            row[f"{name}_mae_deg"] = s["mae"]
            row[f"{name}_retention_pct"] = s["ret_pct"]
        per_mount_rows.append(row)
    write_csv(args.output_dir / "mount_summary_timing_candidate_sweep.csv", per_mount_rows)

    report = args.output_dir / "timing_candidate_sweep_report.md"
    with report.open("w") as f:
        f.write("# Timing Candidate Sweep Report\n\n")
        f.write("- Candidate pool source: raw `kld7_buffer` (vertical) per shot\n")
        f.write("- Distance assumption: known (5/6ft) where provided, else assumed 6ft\n")
        f.write("- Geometry: lateral offset 4in, distance by session\n")
        f.write("- Timing anchors: first-byte centered plus near-equivalent accept/shot-log back-shifts\n")
        f.write("- Oracle uses TM only for analysis ceiling, not production\n\n")

        f.write("## Overall\n\n")
        f.write("| strategy | n | MAE | Bias | RMSE | P90 abs | <=8deg | retention |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in overall_rows:
            f.write(
                f"| {r['strategy']} | {r['n']} | {r['mae_deg']:.3f} | {r['bias_deg']:.3f} | "
                f"{r['rmse_deg']:.3f} | {r['p90_abs_deg']:.3f} | {r['retained_le8']} | {r['retention_pct']:.1f}% |\n"
            )
        f.write("\n## By Session (MAE)\n\n")
        f.write("| session | shots | latest | fixed | prev_oracle | pseudo | sweep_oracle |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in per_session_rows:
            f.write(
                f"| {r['session']} | {r['shots']} | {r['latest_mae_deg']:.3f} | {r['fixed_mae_deg']:.3f} | "
                f"{r['prev_oracle_mae_deg']:.3f} | {r['pseudo_mae_deg']:.3f} | {r['sweep_oracle_mae_deg']:.3f} |\n"
            )
        f.write("\n## By Mount (MAE)\n\n")
        f.write("| mount | shots | latest | fixed | prev_oracle | pseudo | sweep_oracle |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in per_mount_rows:
            f.write(
                f"| {r['mount_deg']:.1f} | {r['shots']} | {r['latest_mae_deg']:.3f} | {r['fixed_mae_deg']:.3f} | "
                f"{r['prev_oracle_mae_deg']:.3f} | {r['pseudo_mae_deg']:.3f} | {r['sweep_oracle_mae_deg']:.3f} |\n"
            )

    print(f"Wrote: {args.output_dir}")
    print(f"- per_shot: {args.output_dir / 'per_shot_timing_candidate_sweep.csv'}")
    print(f"- overall:  {args.output_dir / 'overall_summary_timing_candidate_sweep.csv'}")
    print(f"- report:   {report}")


if __name__ == "__main__":
    main()
