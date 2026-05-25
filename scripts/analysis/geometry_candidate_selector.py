#!/usr/bin/env python3
"""Geometry/frame scoring on production-style K-LD7 candidates.

This experiment keeps ``extract_launch_angle(...)`` as the event/candidate
generator and adds per-frame consistency metrics to the resulting candidate
groups. It is intentionally offline-only.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from timing_candidate_sweep import (
    build_impact_anchors,
    discover_sessions,
    load_baseline_csv,
    load_labels,
    load_timing_map,
    load_vertical_buffers,
    summarize,
    write_csv,
)

from openflight.kld7.radc import extract_launch_angle


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = float(sum(weights))
    if total <= 0.0:
        return float(sum(values) / len(values)) if values else 0.0
    return float(sum(v * w for v, w in zip(values, weights)) / total)


def _weighted_std(values: list[float], weights: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _weighted_mean(values, weights)
    var = _weighted_mean([(v - mean) ** 2 for v in values], weights)
    return float(math.sqrt(max(var, 0.0)))


def _linear_residual(values: list[float], times: list[float], weights: list[float]) -> float:
    if len(values) < 3:
        return _weighted_std(values, weights)
    x = np.column_stack([np.ones(len(times)), np.array(times, dtype=float)])
    y = np.array(values, dtype=float)
    w = np.sqrt(np.maximum(np.array(weights, dtype=float), 1e-9))
    beta, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    pred = x @ beta
    return float(math.sqrt(_weighted_mean(list((y - pred) ** 2), weights)))


def frame_metrics(candidate: dict[str, Any]) -> dict[str, float]:
    per_frame = (candidate.get("aim_correction") or {}).get("per_frame") or []
    corrected: list[float] = []
    raw: list[float] = []
    times: list[float] = []
    for item in per_frame:
        t = item.get("t_after_impact_s")
        c = item.get("corrected_bearing_deg")
        r = item.get("raw_bearing_deg")
        if t is None or c is None or r is None:
            continue
        times.append(float(t))
        corrected.append(float(c))
        raw.append(float(r))
    weights = [1.0] * len(corrected)
    return {
        "per_frame_count": float(len(corrected)),
        "corrected_std_deg": _weighted_std(corrected, weights) if corrected else 99.0,
        "corrected_linear_resid_deg": _linear_residual(corrected, times, weights) if corrected else 99.0,
        "raw_std_deg": _weighted_std(raw, weights) if raw else 99.0,
        "t_span_s": float(max(times) - min(times)) if len(times) >= 2 else 0.0,
        "t_min_s": float(min(times)) if times else 0.0,
        "t_max_s": float(max(times)) if times else 0.0,
    }


def build_candidates_with_metrics(
    frames: list[dict[str, Any]],
    anchors: list[float],
    *,
    ops_speed_mph: float,
    distance_ft: float,
    radar_height_in: float,
    angle_offset_deg: float,
) -> list[dict[str, Any]]:
    # Slightly wider than timing_candidate_sweep: the v4/deep checks showed
    # useful candidates just outside the earlier lane/gating settings.
    impact_thresholds = [1.8, 2.2]
    speed_tolerances = [8.0, 12.0]
    centroid_fracs = [0.55]
    outlier_tols = [15, 35]

    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
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
                            angle_offset_deg=angle_offset_deg,
                            ops243_ball_speed_mph=ops_speed_mph,
                            speed_tolerance_mph=tol,
                            orientation="vertical",
                            ops_bin_outlier_tol=out_tol,
                            ops_bin_outlier_penalty=10.0,
                            centroid_floor_frac=frac,
                            ball_lateral_offset_in=radar_height_in,
                            ball_initial_range_in=distance_ft * 12.0,
                            impact_timestamp=impact_ts,
                        )
                        for shot in shots:
                            key = (
                                int(round(_to_float(shot.get("launch_angle_deg")) * 10)),
                                int(round(_to_float(shot.get("raw_angle_deg")) * 10)),
                                int(round(_to_float(shot.get("confidence")) * 100)),
                                int(_to_float(shot.get("frame_count"))),
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            rec = dict(shot)
                            rec.update(frame_metrics(shot))
                            rec["sweep_impact_ts"] = impact_ts
                            rec["sweep_thr"] = thr
                            rec["sweep_tol"] = tol
                            rec["sweep_frac"] = frac
                            rec["sweep_out_tol"] = out_tol
                            out.append(rec)
    return out


def bucket_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, float]]:
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        pred = round(_to_float(cand.get("launch_angle_deg")), 1)
        buckets[pred].append(cand)

    out: list[dict[str, float]] = []
    for pred, rows in buckets.items():
        n = float(len(rows))
        out.append(
            {
                "pred_v_deg": pred,
                "support": n,
                "confidence": sum(_to_float(r.get("confidence")) for r in rows) / n,
                "avg_snr_db": sum(_to_float(r.get("avg_snr_db")) for r in rows) / n,
                "frame_count": sum(_to_float(r.get("frame_count")) for r in rows) / n,
                "corrected_std_deg": sum(_to_float(r.get("corrected_std_deg"), 99.0) for r in rows) / n,
                "corrected_linear_resid_deg": sum(_to_float(r.get("corrected_linear_resid_deg"), 99.0) for r in rows) / n,
                "per_frame_count": sum(_to_float(r.get("per_frame_count")) for r in rows) / n,
                "t_span_s": sum(_to_float(r.get("t_span_s")) for r in rows) / n,
            }
        )
    return out


def _best_in_lane(
    buckets: list[dict[str, float]],
    lo: float,
    hi: float,
    *,
    use_geometry: bool,
) -> dict[str, float] | None:
    lane = [b for b in buckets if lo <= b["pred_v_deg"] <= hi]
    if not lane:
        return None

    def score(bucket: dict[str, float]) -> tuple[float, float, float, float]:
        geom_penalty = 0.0
        if use_geometry:
            geom_penalty = (
                0.25 * bucket["corrected_linear_resid_deg"]
                + 0.10 * bucket["corrected_std_deg"]
                - 0.15 * min(bucket["per_frame_count"], 6.0)
            )
        return (
            bucket["support"]
            + 0.50 * bucket["confidence"]
            + 0.03 * bucket["avg_snr_db"]
            - geom_penalty,
            bucket["support"],
            bucket["confidence"],
            -abs(bucket["pred_v_deg"] - 16.0),
        )

    return max(lane, key=score)


def predict_v4_like(
    buckets: list[dict[str, float]],
    fixed: float,
    latest: float,
    *,
    use_geometry: bool,
) -> tuple[float, str]:
    base = fixed if abs(fixed - 16.0) <= abs(latest - 16.0) else latest
    low = _best_in_lane(buckets, 9.0, 20.0, use_geometry=use_geometry)
    high = _best_in_lane(buckets, 18.0, 32.0, use_geometry=use_geometry)
    pred = base
    source = "base"

    if low and low["confidence"] >= 0.50 and base < 14.0 and low["pred_v_deg"] > base + 0.5:
        return low["pred_v_deg"], "low_lane"

    if base > 24.0:
        chosen: dict[str, float] | None = None
        source = "base"
        if low and low["confidence"] >= 0.50 and low["pred_v_deg"] < base - 0.5:
            chosen = low
            source = "low_lane"
        if high and high["confidence"] >= 0.50 and high["pred_v_deg"] < base - 0.5:
            if chosen is None or high["pred_v_deg"] > chosen["pred_v_deg"]:
                chosen = high
                source = "high_lane"
        if chosen is not None:
            pred = chosen["pred_v_deg"]
    return pred, source


def predict_geom_score(buckets: list[dict[str, float]]) -> tuple[float | None, str]:
    if not buckets:
        return None, "none"

    def score(bucket: dict[str, float]) -> float:
        pred = bucket["pred_v_deg"]
        plausibility = 0.0
        if pred < 4.0:
            plausibility += 2.0
        if pred > 32.0:
            plausibility += 2.0
        return (
            1.0 * bucket["support"]
            + 0.7 * bucket["confidence"]
            + 0.04 * bucket["avg_snr_db"]
            + 0.1 * min(bucket["per_frame_count"], 6.0)
            - 0.18 * bucket["corrected_linear_resid_deg"]
            - 0.08 * abs(pred - 16.0)
            - plausibility
        )

    best = max(buckets, key=score)
    return best["pred_v_deg"], "geom_score"


def _delta(pred: float | None, truth: float) -> float | None:
    if pred is None:
        return None
    return pred - truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", type=Path, default=Path("~/Desktop/openflight_sessions").expanduser())
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=Path("~/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv").expanduser(),
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("~/Desktop/openflight_sessions/_analysis_fixed_not_high_else_grid_oracle/per_shot_vertical_deltas_latest_fixed_oracle.csv").expanduser(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/Desktop/openflight_sessions/_analysis_geometry_candidate_selector").expanduser(),
    )
    parser.add_argument("--radar-height-in", type=float, default=4.0)
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

    per_shot: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for label in labels:
        base = baseline.get((label.session_dir, label.comparison_file, label.shot_number))
        if base is None:
            continue
        buf = buffers_by_session.get(label.session_dir, {}).get(label.shot_number)
        if not buf or not buf["frames"]:
            continue
        anchors = build_impact_anchors(
            timing.get((label.session_dir, label.shot_number)),
            fallback_shot_ts=float(buf["shot_timestamp"]),
        )
        candidates = build_candidates_with_metrics(
            buf["frames"],
            anchors,
            ops_speed_mph=label.ops_ball_speed_mph,
            distance_ft=label.distance_ft,
            radar_height_in=args.radar_height_in,
            angle_offset_deg=label.mount_deg + 8.0,
        )
        buckets = bucket_candidates(candidates)
        for bucket in buckets:
            bucket_rows.append(
                {
                    "session": label.session_dir,
                    "comparison_file": label.comparison_file,
                    "shot_number": label.shot_number,
                    "tm_launch_v_deg": label.tm_launch_v_deg,
                    **bucket,
                }
            )

        fixed = base["fixed_pred_v_deg"]
        latest = base["latest_pred_v_deg"]
        plain_pred, plain_source = predict_v4_like(buckets, fixed, latest, use_geometry=False)
        geom_lane_pred, geom_lane_source = predict_v4_like(buckets, fixed, latest, use_geometry=True)
        geom_score_pred, geom_score_source = predict_geom_score(buckets)
        oracle_pred = None
        if buckets:
            oracle_pred = min(buckets, key=lambda b: abs(b["pred_v_deg"] - label.tm_launch_v_deg))["pred_v_deg"]

        row: dict[str, Any] = {
            "session": label.session_dir,
            "comparison_file": label.comparison_file,
            "shot_number": label.shot_number,
            "club": label.club,
            "mount_deg": label.mount_deg,
            "distance_ft": label.distance_ft,
            "distance_source": label.distance_source,
            "ops_ball_speed_mph": label.ops_ball_speed_mph,
            "tm_launch_v_deg": label.tm_launch_v_deg,
            "latest_pred_v_deg": latest,
            "fixed_pred_v_deg": fixed,
            "prev_oracle_pred_v_deg": base["oracle_pred_v_deg"],
            "plain_lane_pred_v_deg": plain_pred,
            "geom_lane_pred_v_deg": geom_lane_pred,
            "geom_score_pred_v_deg": geom_score_pred if geom_score_pred is not None else "",
            "candidate_oracle_pred_v_deg": oracle_pred if oracle_pred is not None else "",
            "plain_lane_source": plain_source,
            "geom_lane_source": geom_lane_source,
            "geom_score_source": geom_score_source,
            "bucket_count": len(buckets),
            "candidate_count": len(candidates),
        }
        for name in ("latest", "fixed", "prev_oracle", "plain_lane", "geom_lane", "geom_score", "candidate_oracle"):
            pred = _to_float(row.get(f"{name}_pred_v_deg"), default=float("nan"))
            delta = _delta(pred, label.tm_launch_v_deg) if math.isfinite(pred) else None
            row[f"{name}_delta_deg"] = delta if delta is not None else ""
            row[f"{name}_abs_err_deg"] = abs(delta) if delta is not None else ""
        per_shot.append(row)

    write_csv(args.output_dir / "per_shot_geometry_candidate_selector.csv", per_shot)
    write_csv(args.output_dir / "bucket_geometry_candidate_selector.csv", bucket_rows)

    strategies = [
        ("latest", "latest_pred_v_deg", "latest_delta_deg"),
        ("fixed", "fixed_pred_v_deg", "fixed_delta_deg"),
        ("prev_oracle", "prev_oracle_pred_v_deg", "prev_oracle_delta_deg"),
        ("plain_lane", "plain_lane_pred_v_deg", "plain_lane_delta_deg"),
        ("geom_lane", "geom_lane_pred_v_deg", "geom_lane_delta_deg"),
        ("geom_score", "geom_score_pred_v_deg", "geom_score_delta_deg"),
        ("candidate_oracle", "candidate_oracle_pred_v_deg", "candidate_oracle_delta_deg"),
    ]
    overall: list[dict[str, Any]] = []
    for name, pred_key, delta_key in strategies:
        stats = summarize(per_shot, pred_key, delta_key)
        overall.append(
            {
                "strategy": name,
                "n": int(stats["n"]),
                "mae_deg": stats["mae"],
                "bias_deg": stats["bias"],
                "rmse_deg": stats["rmse"],
                "p90_abs_deg": stats["p90_abs"],
                "retained_le8": int(stats["retained_le8"]),
                "retention_pct": stats["ret_pct"],
            }
        )
    write_csv(args.output_dir / "overall_summary_geometry_candidate_selector.csv", overall)

    session_rows: list[dict[str, Any]] = []
    for session in sorted({r["session"] for r in per_shot}):
        rows = [r for r in per_shot if r["session"] == session]
        out: dict[str, Any] = {"session": session, "shots": len(rows)}
        for name, pred_key, delta_key in strategies:
            stats = summarize(rows, pred_key, delta_key)
            out[f"{name}_mae_deg"] = stats["mae"]
            out[f"{name}_retention_pct"] = stats["ret_pct"]
        session_rows.append(out)
    write_csv(args.output_dir / "session_summary_geometry_candidate_selector.csv", session_rows)

    report = args.output_dir / "geometry_candidate_selector_report.md"
    with report.open("w") as f:
        f.write("# Geometry Candidate Selector Report\n\n")
        f.write("- Candidate generation: production `extract_launch_angle(...)` groups\n")
        f.write("- Added metrics: support, confidence, SNR, corrected-bearing std/residual\n")
        f.write("- TM is only used for validation and oracle rows\n\n")
        f.write("## Overall\n\n")
        f.write("| strategy | n | MAE | Bias | RMSE | P90 abs | <=8deg | retention |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in overall:
            f.write(
                f"| {row['strategy']} | {row['n']} | {row['mae_deg']:.3f} | {row['bias_deg']:.3f} | "
                f"{row['rmse_deg']:.3f} | {row['p90_abs_deg']:.3f} | {row['retained_le8']} | {row['retention_pct']:.1f}% |\n"
            )
        f.write("\n## By Session MAE\n\n")
        f.write("| session | shots | fixed | plain_lane | geom_lane | geom_score | candidate_oracle |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in session_rows:
            f.write(
                f"| {row['session']} | {row['shots']} | {row['fixed_mae_deg']:.3f} | "
                f"{row['plain_lane_mae_deg']:.3f} | {row['geom_lane_mae_deg']:.3f} | "
                f"{row['geom_score_mae_deg']:.3f} | {row['candidate_oracle_mae_deg']:.3f} |\n"
            )
    print(f"Wrote {args.output_dir}")
    print(f"- {args.output_dir / 'overall_summary_geometry_candidate_selector.csv'}")
    print(f"- {report}")


if __name__ == "__main__":
    main()
