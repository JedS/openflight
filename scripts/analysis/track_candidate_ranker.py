#!/usr/bin/env python3
"""Track-style K-LD7 candidate ranker.

This is an offline analysis tool. It treats each K-LD7 launch candidate as
part of a small "track" through the candidate space: support across timing
sweeps, nearby support, frame consistency, confidence, SNR, and launch
plausibility. TrackMan is used only for labeling/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    return int(round(_to_float(value, float(default))))


def _mae(values: list[float]) -> float:
    return float(sum(abs(v) for v in values) / len(values)) if values else float("nan")


def _rmse(values: list[float]) -> float:
    return float(math.sqrt(sum(v * v for v in values) / len(values))) if values else float("nan")


def _p90_abs(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = sorted(abs(v) for v in values)
    return float(arr[int((len(arr) - 1) * 0.9)])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _shot_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        row["session"],
        row["comparison_file"],
        _to_int(row["shot_number"]),
    )


def _club_expected_launch(club: str) -> float:
    normalized = club.lower().replace(" ", "").replace("_", "-")
    if "driver" in normalized:
        return 12.0
    if "9" in normalized:
        return 19.0
    if "8" in normalized:
        return 18.0
    if "7" in normalized:
        return 16.5
    return 16.0


def _mount_center(mount_deg: float, club: str) -> float:
    expected = _club_expected_launch(club)
    if mount_deg >= 16.0:
        return max(15.5, min(19.0, expected + 0.5))
    if mount_deg >= 6.0:
        return max(14.0, min(19.5, expected))
    return expected


def _lane_penalty(pred: float, mount_deg: float, club: str) -> float:
    center = _mount_center(mount_deg, club)
    lo = max(3.0, center - 8.0)
    hi = center + 8.0
    if mount_deg >= 16.0:
        lo = max(lo, 8.0)
        hi = min(hi, 25.0)
    elif mount_deg >= 6.0:
        hi = min(hi, 27.0)
    else:
        hi = min(hi, 32.0)
    if pred < lo:
        return lo - pred
    if pred > hi:
        return pred - hi
    return 0.0


def _safe_ratio(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def load_shot_rows(per_shot_csv: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = {}
    for row in _read_csv(per_shot_csv):
        rows[_shot_key(row)] = row
    return rows


def load_candidate_rows(
    bucket_csv: Path,
    shot_rows: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    by_shot: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(bucket_csv):
        key = _shot_key(row)
        if key not in shot_rows:
            continue
        by_shot[key].append(
            {
                "source": "bucket",
                "pred_v_deg": round(_to_float(row["pred_v_deg"]), 1),
                "support": _to_float(row.get("support")),
                "confidence": _to_float(row.get("confidence")),
                "avg_snr_db": _to_float(row.get("avg_snr_db")),
                "frame_count": _to_float(row.get("frame_count")),
                "corrected_std_deg": _to_float(row.get("corrected_std_deg"), 99.0),
                "corrected_linear_resid_deg": _to_float(
                    row.get("corrected_linear_resid_deg"), 99.0
                ),
                "per_frame_count": _to_float(row.get("per_frame_count")),
                "t_span_s": _to_float(row.get("t_span_s")),
            }
        )

    for key, shot in shot_rows.items():
        pool = by_shot[key]
        existing = {round(_to_float(c["pred_v_deg"]), 1) for c in pool}
        for source, pred_key in (
            ("latest", "latest_pred_v_deg"),
            ("fixed", "fixed_pred_v_deg"),
        ):
            pred = round(_to_float(shot.get(pred_key)), 1)
            if pred in existing:
                continue
            pool.append(
                {
                    "source": source,
                    "pred_v_deg": pred,
                    "support": 0.0,
                    "confidence": 0.50,
                    "avg_snr_db": 5.0,
                    "frame_count": 1.0,
                    "corrected_std_deg": 99.0,
                    "corrected_linear_resid_deg": 99.0,
                    "per_frame_count": 0.0,
                    "t_span_s": 0.0,
                }
            )
    return by_shot


def enrich_candidates(
    shot_rows: dict[tuple[str, str, int], dict[str, Any]],
    by_shot: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, shot in shot_rows.items():
        pool = by_shot.get(key, [])
        if not pool:
            continue
        total_support = sum(_to_float(c["support"]) for c in pool)
        max_support = max((_to_float(c["support"]) for c in pool), default=0.0)
        max_conf = max((_to_float(c["confidence"]) for c in pool), default=0.0)
        preds = [_to_float(c["pred_v_deg"]) for c in pool]
        support_sorted = sorted(
            ((_to_float(c["support"]), _to_float(c["pred_v_deg"])) for c in pool),
            reverse=True,
        )
        rank_by_pred = {
            pred: idx + 1
            for idx, (_, pred) in enumerate(support_sorted)
        }

        session, comparison_file, shot_number = key
        mount = _to_float(shot.get("mount_deg"))
        club = shot.get("club", "unknown")
        center = _mount_center(mount, club)
        for cand in pool:
            pred = _to_float(cand["pred_v_deg"])
            support = _to_float(cand["support"])
            neighbor_05 = sum(
                _to_float(other["support"])
                for other in pool
                if abs(_to_float(other["pred_v_deg"]) - pred) <= 0.5
            )
            neighbor_10 = sum(
                _to_float(other["support"])
                for other in pool
                if abs(_to_float(other["pred_v_deg"]) - pred) <= 1.0
            )
            stronger = [
                abs(_to_float(other["pred_v_deg"]) - pred)
                for other in pool
                if _to_float(other["support"]) > support
            ]
            nearest_stronger = min(stronger) if stronger else 99.0
            out.append(
                {
                    "session": session,
                    "comparison_file": comparison_file,
                    "shot_number": shot_number,
                    "club": club,
                    "mount_deg": mount,
                    "distance_ft": _to_float(shot.get("distance_ft")),
                    "distance_source": shot.get("distance_source", ""),
                    "ops_ball_speed_mph": _to_float(shot.get("ops_ball_speed_mph")),
                    "tm_launch_v_deg": _to_float(shot.get("tm_launch_v_deg")),
                    "source": cand["source"],
                    "pred_v_deg": pred,
                    "candidate_count": len(pool),
                    "support": support,
                    "support_share": _safe_ratio(support, total_support),
                    "max_support_share": _safe_ratio(max_support, total_support),
                    "support_rank": rank_by_pred.get(pred, len(pool)),
                    "neighbor_support_05": neighbor_05,
                    "neighbor_support_10": neighbor_10,
                    "neighbor_share_05": _safe_ratio(neighbor_05, total_support),
                    "neighbor_share_10": _safe_ratio(neighbor_10, total_support),
                    "nearest_stronger_deg": nearest_stronger,
                    "confidence": _to_float(cand.get("confidence")),
                    "confidence_share": _safe_ratio(_to_float(cand.get("confidence")), max_conf),
                    "avg_snr_db": _to_float(cand.get("avg_snr_db")),
                    "frame_count": _to_float(cand.get("frame_count")),
                    "corrected_std_deg": _to_float(cand.get("corrected_std_deg"), 99.0),
                    "corrected_linear_resid_deg": _to_float(
                        cand.get("corrected_linear_resid_deg"), 99.0
                    ),
                    "per_frame_count": _to_float(cand.get("per_frame_count")),
                    "t_span_s": _to_float(cand.get("t_span_s")),
                    "expected_launch_deg": center,
                    "abs_from_expected": abs(pred - center),
                    "lane_penalty": _lane_penalty(pred, mount, club),
                    "is_high_pred": 1.0 if pred > 24.0 else 0.0,
                    "is_low_pred": 1.0 if pred < 8.0 else 0.0,
                    "is_fallback": 1.0 if cand["source"] != "bucket" else 0.0,
                    "has_nearby_pred": 1.0 if any(abs(other - pred) <= 1.0 and other != pred for other in preds) else 0.0,
                }
            )
    return out


FEATURE_COLUMNS = [
    "pred_v_deg",
    "support",
    "support_share",
    "max_support_share",
    "support_rank",
    "neighbor_support_05",
    "neighbor_support_10",
    "neighbor_share_05",
    "neighbor_share_10",
    "nearest_stronger_deg",
    "confidence",
    "confidence_share",
    "avg_snr_db",
    "frame_count",
    "corrected_std_deg",
    "corrected_linear_resid_deg",
    "per_frame_count",
    "t_span_s",
    "expected_launch_deg",
    "abs_from_expected",
    "lane_penalty",
    "is_high_pred",
    "is_low_pred",
    "is_fallback",
    "has_nearby_pred",
    "mount_deg",
    "distance_ft",
    "ops_ball_speed_mph",
    "candidate_count",
]


def feature_vector(row: dict[str, Any]) -> np.ndarray:
    values = [_to_float(row.get(name)) for name in FEATURE_COLUMNS]
    pred = _to_float(row["pred_v_deg"])
    mount = _to_float(row["mount_deg"])
    expected = _to_float(row["expected_launch_deg"])
    values.extend(
        [
            pred * pred / 100.0,
            pred * mount / 100.0,
            abs(pred - 16.0),
            abs(pred - expected) ** 2 / 25.0,
            math.log1p(max(_to_float(row["support"]), 0.0)),
            math.log1p(max(_to_float(row["neighbor_support_10"]), 0.0)),
        ]
    )
    return np.array(values, dtype=float)


def manual_track_score(row: dict[str, Any]) -> float:
    pred = _to_float(row["pred_v_deg"])
    mount = _to_float(row["mount_deg"])
    score = 0.0
    score += 2.2 * _to_float(row["support_share"])
    score += 1.8 * _to_float(row["neighbor_share_10"])
    score += 0.8 * _to_float(row["confidence"])
    score += 0.04 * min(_to_float(row["avg_snr_db"]), 25.0)
    score += 0.10 * min(_to_float(row["frame_count"]), 5.0)
    score += 0.08 * min(_to_float(row["per_frame_count"]), 5.0)
    score += 0.20 * _to_float(row["has_nearby_pred"])
    score -= 0.16 * _to_float(row["abs_from_expected"])
    score -= 0.40 * _to_float(row["lane_penalty"])
    score -= 0.015 * min(_to_float(row["corrected_linear_resid_deg"]), 30.0)
    score -= 0.020 * min(_to_float(row["corrected_std_deg"]), 30.0)
    score -= 0.18 * _to_float(row["is_fallback"])
    if mount >= 16.0 and pred > 24.0:
        score -= 1.7
    if mount >= 16.0 and pred < 10.0:
        score -= 1.1
    if mount < 6.0 and pred < 5.0:
        score -= 1.0
    return score


def summarize(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float]:
    deltas = [
        _to_float(row[pred_key]) - _to_float(row["tm_launch_v_deg"])
        for row in rows
        if row.get(pred_key) not in (None, "")
    ]
    retained = sum(1 for delta in deltas if abs(delta) <= 8.0)
    return {
        "n": float(len(deltas)),
        "mae_deg": _mae(deltas),
        "bias_deg": float(sum(deltas) / len(deltas)) if deltas else float("nan"),
        "rmse_deg": _rmse(deltas),
        "p90_abs_deg": _p90_abs(deltas),
        "retained_le8": float(retained),
        "retention_pct": retained / len(deltas) * 100.0 if deltas else 0.0,
    }


def _select_best(
    candidates: list[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    return max(candidates, key=lambda row: (_to_float(row[score_key]), -_to_float(row["support_rank"])))


def _select_oracle(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda row: abs(_to_float(row["pred_v_deg"]) - _to_float(row["tm_launch_v_deg"])),
    )


def _fit_pairwise_ranker(train_rows: list[dict[str, Any]], ridge_lambda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_shot: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_shot[_shot_key(row)].append(row)

    diffs: list[np.ndarray] = []
    for candidates in by_shot.values():
        if len(candidates) < 2:
            continue
        oracle = _select_oracle(candidates)
        oracle_err = abs(_to_float(oracle["pred_v_deg"]) - _to_float(oracle["tm_launch_v_deg"]))
        oracle_x = feature_vector(oracle)
        for row in candidates:
            if row is oracle:
                continue
            err = abs(_to_float(row["pred_v_deg"]) - _to_float(row["tm_launch_v_deg"]))
            if err <= oracle_err + 0.25:
                continue
            diff = oracle_x - feature_vector(row)
            diffs.append(diff)
            diffs.append(-diff)

    if not diffs:
        n = len(feature_vector(train_rows[0]))
        return np.zeros(n), np.zeros(n), np.ones(n)

    x_raw = np.vstack(diffs)
    y = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(diffs))])
    scale_rows = np.vstack([feature_vector(row) for row in train_rows])
    mean = np.mean(scale_rows, axis=0)
    std = np.std(scale_rows, axis=0)
    std[std < 1e-9] = 1.0
    x = x_raw / std
    xtx = x.T @ x
    reg = ridge_lambda * np.eye(x.shape[1])
    weights = np.linalg.solve(xtx + reg, x.T @ y)
    return weights, mean, std


def _ranker_score(row: dict[str, Any], weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    return float(((feature_vector(row) - mean) / std) @ weights)


def _fit_error_ranker(
    train_rows: list[dict[str, Any]],
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = np.vstack([feature_vector(row) for row in train_rows])
    mean = np.mean(x_raw, axis=0)
    std = np.std(x_raw, axis=0)
    std[std < 1e-9] = 1.0
    x = (x_raw - mean) / std
    abs_err = np.array(
        [
            abs(_to_float(row["pred_v_deg"]) - _to_float(row["tm_launch_v_deg"]))
            for row in train_rows
        ],
        dtype=float,
    )
    # Higher score should be better at selection time, so train against
    # negative absolute error rather than absolute error.
    y = -abs_err
    xtx = x.T @ x
    reg = ridge_lambda * np.eye(x.shape[1])
    weights = np.linalg.solve(xtx + reg, x.T @ y)
    return weights, mean, std


def run_loso_error_ranker(
    candidate_rows: list[dict[str, Any]],
    ridge_lambda: float,
) -> tuple[dict[tuple[str, str, int], tuple[float, str, float]], list[dict[str, Any]]]:
    sessions = sorted({row["session"] for row in candidate_rows})
    selected: dict[tuple[str, str, int], tuple[float, str, float]] = {}
    weight_rows: list[dict[str, Any]] = []
    feature_names = FEATURE_COLUMNS + [
        "pred_sq",
        "pred_x_mount",
        "abs_pred_16",
        "abs_expected_sq",
        "log_support",
        "log_neighbor10",
    ]
    for test_session in sessions:
        train = [row for row in candidate_rows if row["session"] != test_session]
        test = [row for row in candidate_rows if row["session"] == test_session]
        weights, mean, std = _fit_error_ranker(train, ridge_lambda)
        for feature, weight in sorted(
            zip(feature_names, weights),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:12]:
            weight_rows.append(
                {
                    "ranker": "error",
                    "heldout_session": test_session,
                    "feature": feature,
                    "weight": weight,
                }
            )

        by_shot: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in test:
            scored = dict(row)
            scored["learned_error_score"] = _ranker_score(row, weights, mean, std)
            by_shot[_shot_key(scored)].append(scored)
        for key, rows in by_shot.items():
            best = _select_best(rows, "learned_error_score")
            selected[key] = (
                _to_float(best["pred_v_deg"]),
                str(best["source"]),
                _to_float(best["learned_error_score"]),
            )
    return selected, weight_rows


def run_loso_pairwise_ranker(
    candidate_rows: list[dict[str, Any]],
    ridge_lambda: float,
) -> tuple[dict[tuple[str, str, int], tuple[float, str, float]], list[dict[str, Any]]]:
    sessions = sorted({row["session"] for row in candidate_rows})
    selected: dict[tuple[str, str, int], tuple[float, str, float]] = {}
    weight_rows: list[dict[str, Any]] = []
    for test_session in sessions:
        train = [row for row in candidate_rows if row["session"] != test_session]
        test = [row for row in candidate_rows if row["session"] == test_session]
        weights, mean, std = _fit_pairwise_ranker(train, ridge_lambda)
        for feature, weight in sorted(
            zip(FEATURE_COLUMNS + ["pred_sq", "pred_x_mount", "abs_pred_16", "abs_expected_sq", "log_support", "log_neighbor10"], weights),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:12]:
            weight_rows.append(
                {
                    "ranker": "pairwise",
                    "heldout_session": test_session,
                    "feature": feature,
                    "weight": weight,
                }
            )

        by_shot: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in test:
            scored = dict(row)
            scored["learned_rank_score"] = _ranker_score(row, weights, mean, std)
            by_shot[_shot_key(scored)].append(scored)
        for key, rows in by_shot.items():
            best = _select_best(rows, "learned_rank_score")
            selected[key] = (
                _to_float(best["pred_v_deg"]),
                str(best["source"]),
                _to_float(best["learned_rank_score"]),
            )
    return selected, weight_rows


def build_per_shot_output(
    shot_rows: dict[tuple[str, str, int], dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    learned_error_selected: dict[tuple[str, str, int], tuple[float, str, float]],
    learned_pairwise_selected: dict[tuple[str, str, int], tuple[float, str, float]],
) -> list[dict[str, Any]]:
    by_shot: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        scored = dict(row)
        scored["manual_track_score"] = manual_track_score(row)
        by_shot[_shot_key(row)].append(scored)

    out: list[dict[str, Any]] = []
    for key, shot in sorted(shot_rows.items()):
        candidates = by_shot.get(key)
        if not candidates:
            continue
        oracle = _select_oracle(candidates)
        manual = _select_best(candidates, "manual_track_score")
        learned_error_pred, learned_error_source, learned_error_score = learned_error_selected.get(
            key,
            (_to_float(shot["fixed_pred_v_deg"]), "fallback_missing", 0.0),
        )
        learned_pairwise_pred, learned_pairwise_source, learned_pairwise_score = (
            learned_pairwise_selected.get(
                key,
                (_to_float(shot["fixed_pred_v_deg"]), "fallback_missing", 0.0),
            )
        )
        tm = _to_float(shot["tm_launch_v_deg"])
        row = {
            "session": key[0],
            "comparison_file": key[1],
            "shot_number": key[2],
            "club": shot.get("club", ""),
            "mount_deg": _to_float(shot.get("mount_deg")),
            "distance_ft": _to_float(shot.get("distance_ft")),
            "distance_source": shot.get("distance_source", ""),
            "ops_ball_speed_mph": _to_float(shot.get("ops_ball_speed_mph")),
            "tm_launch_v_deg": tm,
            "latest_pred_v_deg": _to_float(shot.get("latest_pred_v_deg")),
            "fixed_pred_v_deg": _to_float(shot.get("fixed_pred_v_deg")),
            "prev_oracle_pred_v_deg": _to_float(shot.get("prev_oracle_pred_v_deg")),
            "geom_v5_pred_v_deg": _to_float(shot.get("geom_v5_pred_v_deg"), float("nan")),
            "manual_track_pred_v_deg": _to_float(manual["pred_v_deg"]),
            "manual_track_source": manual["source"],
            "manual_track_score": manual["manual_track_score"],
            "learned_error_pred_v_deg": learned_error_pred,
            "learned_error_source": learned_error_source,
            "learned_error_score": learned_error_score,
            "learned_pairwise_pred_v_deg": learned_pairwise_pred,
            "learned_pairwise_source": learned_pairwise_source,
            "learned_pairwise_score": learned_pairwise_score,
            "candidate_oracle_pred_v_deg": _to_float(oracle["pred_v_deg"]),
            "candidate_oracle_source": oracle["source"],
            "candidate_count": len(candidates),
            "bucket_count": sum(1 for c in candidates if c["source"] == "bucket"),
        }
        for name in (
            "latest",
            "fixed",
            "prev_oracle",
            "geom_v5",
            "manual_track",
            "learned_error",
            "learned_pairwise",
            "candidate_oracle",
        ):
            pred = _to_float(row.get(f"{name}_pred_v_deg"), float("nan"))
            if math.isfinite(pred):
                delta = pred - tm
                row[f"{name}_delta_deg"] = delta
                row[f"{name}_abs_err_deg"] = abs(delta)
            else:
                row[f"{name}_delta_deg"] = ""
                row[f"{name}_abs_err_deg"] = ""
        out.append(row)
    return out


def write_summaries(output_dir: Path, per_shot: list[dict[str, Any]], weight_rows: list[dict[str, Any]]) -> None:
    strategies = [
        "latest",
        "fixed",
        "prev_oracle",
        "geom_v5",
        "manual_track",
        "learned_error",
        "learned_pairwise",
        "candidate_oracle",
    ]
    overall = []
    for strategy in strategies:
        stats = summarize(per_shot, f"{strategy}_pred_v_deg")
        overall.append({"strategy": strategy, **stats})
    _write_csv(output_dir / "overall_summary_track_candidate_ranker.csv", overall)

    session_rows = []
    for session in sorted({row["session"] for row in per_shot}):
        rows = [row for row in per_shot if row["session"] == session]
        out: dict[str, Any] = {"session": session, "shots": len(rows)}
        for strategy in strategies:
            stats = summarize(rows, f"{strategy}_pred_v_deg")
            out[f"{strategy}_mae_deg"] = stats["mae_deg"]
            out[f"{strategy}_retention_pct"] = stats["retention_pct"]
        session_rows.append(out)
    _write_csv(output_dir / "session_summary_track_candidate_ranker.csv", session_rows)

    mount_rows = []
    for mount in sorted({_to_float(row["mount_deg"]) for row in per_shot}):
        rows = [row for row in per_shot if _to_float(row["mount_deg"]) == mount]
        out = {"mount_deg": mount, "shots": len(rows)}
        for strategy in strategies:
            stats = summarize(rows, f"{strategy}_pred_v_deg")
            out[f"{strategy}_mae_deg"] = stats["mae_deg"]
            out[f"{strategy}_retention_pct"] = stats["retention_pct"]
        mount_rows.append(out)
    _write_csv(output_dir / "mount_summary_track_candidate_ranker.csv", mount_rows)

    coverage_rows = []
    coverage_groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "ALL", per_shot),
    ]
    coverage_groups.extend(
        ("session", session, [row for row in per_shot if row["session"] == session])
        for session in sorted({row["session"] for row in per_shot})
    )
    coverage_groups.extend(
        ("mount", str(mount), [row for row in per_shot if _to_float(row["mount_deg"]) == mount])
        for mount in sorted({_to_float(row["mount_deg"]) for row in per_shot})
    )
    coverage_groups.extend(
        ("club", club, [row for row in per_shot if row["club"] == club])
        for club in sorted({str(row["club"]) for row in per_shot})
    )
    coverage_groups.extend(
        [
            ("subset", "non_driver", [row for row in per_shot if row["club"] != "driver"]),
            ("subset", "7iron_only", [row for row in per_shot if row["club"] == "7-iron"]),
            ("subset", "18deg_only", [row for row in per_shot if _to_float(row["mount_deg"]) == 18.0]),
        ]
    )
    for group_type, group, rows in coverage_groups:
        errors = [_to_float(row["candidate_oracle_abs_err_deg"]) for row in rows]
        if not errors:
            continue
        coverage_rows.append(
            {
                "group_type": group_type,
                "group": group,
                "shots": len(errors),
                "candidate_oracle_mae_deg": _mae(errors),
                "within_1deg": sum(1 for err in errors if err <= 1.0),
                "within_2deg": sum(1 for err in errors if err <= 2.0),
                "within_3deg": sum(1 for err in errors if err <= 3.0),
                "within_5deg": sum(1 for err in errors if err <= 5.0),
            }
        )
    _write_csv(output_dir / "coverage_summary_track_candidate_ranker.csv", coverage_rows)

    worst = sorted(
        per_shot,
        key=lambda row: _to_float(row["learned_error_abs_err_deg"]),
        reverse=True,
    )[:20]
    _write_csv(output_dir / "top20_worst_learned_error.csv", worst)
    _write_csv(output_dir / "learned_rank_top_weights.csv", weight_rows)

    report = output_dir / "track_candidate_ranker_report.md"
    with report.open("w") as f:
        f.write("# Track Candidate Ranker Report\n\n")
        f.write("Offline analysis only. TrackMan is used for labels/evaluation; runtime features only are used for scoring.\n\n")
        f.write("## Overall\n\n")
        f.write("| strategy | n | MAE | Bias | RMSE | P90 abs | <=8deg | retention |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in overall:
            f.write(
                f"| {row['strategy']} | {int(row['n'])} | {row['mae_deg']:.3f} | "
                f"{row['bias_deg']:.3f} | {row['rmse_deg']:.3f} | {row['p90_abs_deg']:.3f} | "
                f"{int(row['retained_le8'])} | {row['retention_pct']:.1f}% |\n"
            )
        f.write("\n## By Session MAE\n\n")
        f.write("| session | shots | fixed | geom_v5 | manual_track | learned_error | learned_pairwise | candidate_oracle |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in session_rows:
            f.write(
                f"| {row['session']} | {row['shots']} | {row['fixed_mae_deg']:.3f} | "
                f"{row['geom_v5_mae_deg']:.3f} | {row['manual_track_mae_deg']:.3f} | "
                f"{row['learned_error_mae_deg']:.3f} | {row['learned_pairwise_mae_deg']:.3f} | "
                f"{row['candidate_oracle_mae_deg']:.3f} |\n"
            )
        f.write("\n## Interpretation Notes\n\n")
        f.write("- `candidate_oracle` is the TM-selected ceiling for this candidate cache.\n")
        f.write("- `manual_track` is a hand-built TM-free score using support, nearby support, confidence, consistency, and plausibility.\n")
        f.write("- `learned_error` is leave-one-session-out: each held-out session is scored by a ridge model trained to predict negative candidate absolute error on the other sessions.\n")
        f.write("- `learned_pairwise` is leave-one-session-out: each held-out session is scored by a pairwise ranker trained on the other sessions.\n")
        f.write("- If `candidate_oracle` is high, the right answer is missing from this candidate pool or geometry assumptions are wrong.\n")
        f.write("\n## Candidate Coverage\n\n")
        f.write("| group | shots | oracle MAE | <=1deg | <=2deg | <=3deg | <=5deg |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in coverage_rows:
            if row["group_type"] not in {"overall", "subset", "mount"}:
                continue
            f.write(
                f"| {row['group_type']}:{row['group']} | {row['shots']} | "
                f"{row['candidate_oracle_mae_deg']:.3f} | {row['within_1deg']} | "
                f"{row['within_2deg']} | {row['within_3deg']} | {row['within_5deg']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("~/Desktop/openflight_sessions").expanduser()
    parser.add_argument(
        "--per-shot-csv",
        type=Path,
        default=root / "_analysis_geometry_candidate_selector_v5_cached/per_shot_geometry_candidate_selector_v5.csv",
    )
    parser.add_argument(
        "--bucket-csv",
        type=Path,
        default=root / "_analysis_geometry_candidate_selector/bucket_geometry_candidate_selector.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "_analysis_track_candidate_ranker_cached",
    )
    parser.add_argument("--error-ridge-lambda", type=float, default=1.0)
    parser.add_argument("--pairwise-ridge-lambda", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shot_rows = load_shot_rows(args.per_shot_csv)
    by_shot = load_candidate_rows(args.bucket_csv, shot_rows)
    candidates = enrich_candidates(shot_rows, by_shot)
    learned_error, error_weights = run_loso_error_ranker(
        candidates,
        ridge_lambda=args.error_ridge_lambda,
    )
    learned_pairwise, pairwise_weights = run_loso_pairwise_ranker(
        candidates,
        ridge_lambda=args.pairwise_ridge_lambda,
    )
    per_shot = build_per_shot_output(
        shot_rows,
        candidates,
        learned_error,
        learned_pairwise,
    )

    _write_csv(args.output_dir / "candidate_rows_track_candidate_ranker.csv", candidates)
    _write_csv(args.output_dir / "per_shot_track_candidate_ranker.csv", per_shot)
    write_summaries(args.output_dir, per_shot, error_weights + pairwise_weights)

    print(f"Wrote {args.output_dir}")
    print(f"- {args.output_dir / 'track_candidate_ranker_report.md'}")
    print(f"- {args.output_dir / 'per_shot_track_candidate_ranker.csv'}")


if __name__ == "__main__":
    main()
