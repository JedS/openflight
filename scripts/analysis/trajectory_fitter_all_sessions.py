#!/usr/bin/env python3
"""Run latest trajectory-fitter (multi-bin + corrected geometry) on all sessions.

Geometry assumptions:
- Radar center 4 inches above floor.
- Ball on floor → 4 inches BELOW radar center (ball_above_radar = -0.333 ft).
- Mount tilt per session (from session label).
- Net 10 feet past the ball (where unknown).
- d_initial configurable; for unknown sessions, runs with both 5 ft and 6 ft.

Pipeline:
- Impact timing from OPS rolling_buffer_capture (`ball_timestamp_ms`).
- Per-frame bearing from local-peak-near-OPS-bin (multi-bin neighborhood weighted mean).
- Frames filtered: snr ≥ 2, t ≥ 15 ms post-impact, t ≤ flight_time + 20 ms buffer.
- Trajectory fit: 1D grid search for α that best explains observed (t, β) trajectory.
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


# (session_id, mount_deg, comparison_filename, jsonl_id, long_name, d_known_ft_or_None, net_ft)
SESSIONS = [
    # For sessions with unknown net distance, use 15 ft to be generous
    # (we don't know if these old sessions had 10ft or larger nets).
    ("20260522_135647",         0.0, "comparison_20260522_135647.csv",         "20260522_135647", "20260522_135647_0deg_7iron_16shots",          None, 15.0),
    ("20260522_141038",         8.0, "comparison_20260522_141038.csv",         "20260522_141038", "20260522_141038_8deg_7iron_13shots",          None, 15.0),
    ("20260522_141949",        18.0, "comparison_20260522_141949.csv",         "20260522_141949", "20260522_141949_18deg_7iron_11shots",         None, 15.0),
    ("20260522_142538__7iron",  0.0, "comparison_20260522_142538_7iron.csv",   "20260522_142538", "20260522_142538_0deg_4club_68shots",          None, 15.0),
    ("20260522_142538__8iron",  0.0, "comparison_20260522_142538_8iron.csv",   "20260522_142538", "20260522_142538_0deg_4club_68shots",          None, 15.0),
    ("20260522_142538__9iron",  0.0, "comparison_20260522_142538_9iron.csv",   "20260522_142538", "20260522_142538_0deg_4club_68shots",          None, 15.0),
    ("20260522_142538__driver", 0.0, "comparison_20260522_142538_driver.csv",  "20260522_142538", "20260522_142538_0deg_4club_68shots",          None, 15.0),
    ("20260523_143732",        18.0, "comparison_20260523_143732.csv",         "20260523_143732", "20260523_143732_18deg_7iron_8shots",          6.0,  12.0),
    ("20260523_144415",        18.0, "comparison_20260523_144415.csv",         "20260523_144415", "20260523_144415_18deg_7iron_5shots_cleaned", 5.0,  10.0),
]

BALL_ABOVE_RADAR_FT = -4.0 / 12.0   # ball 4 inches below radar center
MPH_TO_FTS = 1.46667
FFT_SIZE = 2048
FIRST_BYTE_TRIGGER_DELAY_MS = 68.0
IMPACT_EXTRA_OFFSET_MS = -10.0  # empirical tuning
BORESIGHT_OFFSET_DEG = 8.0  # K-LD7 electrical boresight assumed +8° above mechanical mount axis


def _to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_ts(s):
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


def pi_impact_ts(rbc, first_byte_ts):
    ball_ts_ms = rbc.get("ball_timestamp_ms")
    trigger_off_ms = rbc.get("trigger_offset_ms")
    if ball_ts_ms is None or trigger_off_ms is None:
        return None
    return (
        first_byte_ts - FIRST_BYTE_TRIGGER_DELAY_MS / 1000.0
        - (trigger_off_ms - ball_ts_ms) / 1000.0
        + IMPACT_EXTRA_OFFSET_MS / 1000.0
    )


def multi_bin_bearing(radc_bytes, ball_speed_mph, search_win=30, nbhd=4, frac=0.5):
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
    expected = expected_ball_bin_from_speed(ball_speed_mph, fft_size=FFT_SIZE)
    win_idx = [(expected + o) % FFT_SIZE for o in range(-search_win, search_win + 1)]
    win_mag = mag[win_idx]
    peak_pos = int(np.argmax(win_mag))
    peak_bin = win_idx[peak_pos]
    peak_mag = float(win_mag[peak_pos])
    if peak_mag <= 0:
        return None
    nbhd_idx = [(peak_bin + o) % FFT_SIZE for o in range(-nbhd, nbhd + 1)]
    nbhd_mag = mag[nbhd_idx]
    nbhd_ang = angles[nbhd_idx]
    mask = nbhd_mag >= frac * peak_mag
    if not np.any(mask):
        return None
    valid_mag = nbhd_mag[mask]
    valid_ang = nbhd_ang[mask]
    total = float(np.sum(valid_mag))
    weighted_mean = float(np.sum(valid_ang * valid_mag) / total)
    noise = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
    snr_db = 10.0 * math.log10(max(peak_mag / max(noise, 1e-9), 1e-9))
    return weighted_mean, snr_db


def predict_beta(alpha, t_ms, v_mph, d, mount, ball_above):
    """Predicted K-LD7 bearing including assumed boresight offset.

    The radar's electrical boresight is treated as `mount + BORESIGHT_OFFSET_DEG`
    above horizontal. Equivalently: subtract that offset from the geometric
    bearing measured against the mechanical mount axis.
    """
    v_fts = v_mph * MPH_TO_FTS
    a = math.radians(alpha)
    t_s = t_ms / 1000.0
    x = d + v_fts * math.cos(a) * t_s
    y = ball_above + v_fts * math.sin(a) * t_s
    effective_mount = mount + BORESIGHT_OFFSET_DEG
    return math.degrees(math.atan2(y, x)) - effective_mount


def fit_alpha(frames, v_mph, d, mount, ball_above, alpha_range=(0.0, 35.0)):
    best_a, best_sse = float("nan"), float("inf")
    alphas = np.arange(alpha_range[0], alpha_range[1] + 0.01, 0.05)
    for alpha in alphas:
        sse = sum(
            (b - predict_beta(float(alpha), t, v_mph, d, mount, ball_above)) ** 2
            for t, b in frames
        )
        if sse < best_sse:
            best_sse, best_a = sse, float(alpha)
    return best_a, math.sqrt(best_sse / len(frames))


def process_shot(rbc, kld7_frames, fb_ts, ball_speed_mph,
                 d_initial, mount, ball_above, net_ft,
                 snr_min=2.0, t_min_ms=15.0):
    impact_ts = pi_impact_ts(rbc, fb_ts)
    if impact_ts is None:
        return None, None, "no_impact_ts", []

    v_fts = ball_speed_mph * MPH_TO_FTS
    if v_fts <= 0:
        return None, None, "no_speed", []

    flight_s = net_ft / (v_fts * math.cos(math.radians(15.0)))
    t_post_ms = flight_s * 1000.0

    obs = []
    for frame in kld7_frames:
        frame_ts = frame.get("timestamp")
        if frame_ts is None:
            continue
        t = float(frame_ts) - impact_ts
        if t <= 0 or t * 1000 < t_min_ms or t * 1000 > t_post_ms:
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
        res = multi_bin_bearing(radc, ball_speed_mph)
        if res is None:
            continue
        beta, snr = res
        if snr < snr_min:
            continue
        obs.append((t * 1000.0, beta))

    obs.sort()

    # Monotonicity filter: bearings should rise; reject frames that break
    # monotonicity by more than 2°. Keep the longest monotonic subsequence
    # via a simple greedy approach: drop the worst-violating frame.
    while len(obs) >= 2:
        violations = []
        for i in range(1, len(obs)):
            drop = obs[i - 1][1] - obs[i][1]
            if drop > 2.0:  # bearing went DOWN by more than 2°
                violations.append((drop, i, i - 1))
        if not violations:
            break
        violations.sort(reverse=True)
        worst = violations[0]
        # Drop the frame whose β is most out of trend
        i_curr = worst[1]
        i_prev = worst[2]
        # Determine which one to drop: the one farther from the median of all bearings
        med_b = sorted([b for _, b in obs])[len(obs) // 2]
        if abs(obs[i_curr][1] - med_b) > abs(obs[i_prev][1] - med_b):
            obs.pop(i_curr)
        else:
            obs.pop(i_prev)

    if len(obs) == 0:
        return None, None, "only_0_frames", obs
    if len(obs) == 1:
        # Closed-form single-frame solution (with boresight offset)
        t_ms, beta_obs = obs[0]
        v_fts = ball_speed_mph * MPH_TO_FTS
        t_s = t_ms / 1000.0
        effective_mount = mount + BORESIGHT_OFFSET_DEG
        K = math.tan(math.radians(beta_obs + effective_mount))
        rhs = (K * d_initial - ball_above) / (v_fts * t_s)
        denom = math.sqrt(1.0 + K * K)
        arg = rhs / denom
        if not -1.0 <= arg <= 1.0:
            return None, None, "single_frame_no_solution", obs
        alpha = math.degrees(math.atan(K) + math.asin(arg))
        return alpha, None, "single_frame", obs

    t_ms_list = [t for t, _ in obs]
    beta_list = [b for _, b in obs]
    obs_for_fit = list(zip(t_ms_list, beta_list))

    alpha, rmse = fit_alpha(obs_for_fit, ball_speed_mph, d_initial, mount, ball_above)
    return alpha, rmse, "ok", obs


def run_session(session_id, mount, comp_file, jsonl_id, long_name, d_initial, net_ft,
                sessions_dir, timing_csv):
    fb_times = load_first_byte_times(timing_csv, long_name)
    targets = load_comparison(sessions_dir / comp_file)
    rbc_map, kld7_map = find_session_data(sessions_dir / f"session_{jsonl_id}_range.jsonl")

    results = []
    for shot_n in sorted(targets.keys()):
        tgt = targets[shot_n]
        rbc = rbc_map.get(shot_n)
        kld7_frames = kld7_map.get(shot_n)
        fb_ts = fb_times.get(shot_n)
        if rbc is None or kld7_frames is None or fb_ts is None:
            results.append({
                "shot": shot_n, "tm": tgt["tm_launch_v_deg"], "club": tgt["club"],
                "alpha": None, "err": None, "rmse": None, "reason": "missing_data",
                "n_frames": 0,
            })
            continue
        alpha, rmse, reason, obs = process_shot(
            rbc, kld7_frames, fb_ts, tgt["ball_speed_mph"],
            d_initial, mount, BALL_ABOVE_RADAR_FT, net_ft,
        )
        err = alpha - tgt["tm_launch_v_deg"] if alpha is not None else None
        results.append({
            "shot": shot_n, "tm": tgt["tm_launch_v_deg"], "club": tgt["club"],
            "alpha": alpha, "err": err, "rmse": rmse, "reason": reason,
            "n_frames": len(obs),
        })
    return results


def summarize(results, label, high_conf_only=False):
    detected = [r for r in results if r["alpha"] is not None and r["err"] is not None]
    if high_conf_only:
        detected = [r for r in detected if r["rmse"] is not None and r["rmse"] <= 2.0]
    n_total = len(results)
    n_det = len(detected)
    if not detected:
        return {"label": label, "n_total": n_total, "n_det": 0,
                "mae": None, "bias": None, "rmse": None, "w1": 0, "w2": 0, "w3": 0}
    errs = [r["err"] for r in detected]
    abs_errs = [abs(e) for e in errs]
    return {
        "label": label,
        "n_total": n_total,
        "n_det": n_det,
        "mae": sum(abs_errs) / n_det,
        "bias": sum(errs) / n_det,
        "rmse": math.sqrt(sum(e * e for e in errs) / n_det),
        "w1": sum(1 for e in abs_errs if e <= 1),
        "w2": sum(1 for e in abs_errs if e <= 2),
        "w3": sum(1 for e in abs_errs if e <= 3),
    }


def fmt_signed(x):
    return f"{'+' if x >= 0 else '-'}{abs(x):.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions-dir", type=Path,
                   default=Path("/Users/john.pacino/openflight_sessions"))
    p.add_argument("--timing-csv", type=Path,
                   default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_trajectory_all_sessions"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    print()
    print("=" * 105)
    print("Trajectory fitter on all sessions (multi-bin bearing, ball -4in below radar, OPS impact timing)")
    print("=" * 105)

    for entry in SESSIONS:
        sid, mount, comp, jsonl_id, long_name, d_known, net_ft = entry
        if d_known is not None:
            results = run_session(sid, mount, comp, jsonl_id, long_name, d_known, net_ft,
                                  args.sessions_dir, args.timing_csv)
            all_results.append((sid, d_known, results))
        else:
            for d_try in [5.0, 6.0]:
                results = run_session(sid, mount, comp, jsonl_id, long_name, d_try, net_ft,
                                      args.sessions_dir, args.timing_csv)
                all_results.append((sid, d_try, results))

    print()
    print("--- ALL FITS (multi-frame + single-frame fallback) ---")
    print(f"{'Session':>30}  {'mount':>5}  {'d':>4}  {'N':>3}  {'det':>3}  {'MAE':>5}  {'bias':>6}  {'<=2':>4}  {'<=3':>4}")
    print("-" * 95)
    for (sid, d, results) in all_results:
        s = summarize(results, "")
        mount = next(e[1] for e in SESSIONS if e[0] == sid)
        if s["n_det"] == 0:
            print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {s['n_total']:>3}    0     —       —      —     —")
        else:
            print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {s['n_total']:>3}  {s['n_det']:>3}  {s['mae']:>5.2f}  {fmt_signed(s['bias']):>6}  {s['w2']:>2}/{s['n_det']}  {s['w3']:>2}/{s['n_det']}")

    print()
    print("--- MULTI-FRAME FITS ONLY (>= 2 frames) ---")
    print(f"{'Session':>30}  {'mount':>5}  {'d':>4}  {'N':>3}  {'det':>3}  {'MAE':>5}  {'bias':>6}  {'<=2':>4}  {'<=3':>4}")
    print("-" * 95)
    for (sid, d, results) in all_results:
        multi = [r for r in results if r.get("reason") == "ok"]
        if not multi:
            mount = next(e[1] for e in SESSIONS if e[0] == sid)
            print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {len(results):>3}    0     —       —      —     —")
            continue
        errs = [r["err"] for r in multi]
        abs_errs = [abs(e) for e in errs]
        n = len(errs)
        mae = sum(abs_errs)/n
        bias = sum(errs)/n
        w2 = sum(1 for e in abs_errs if e <= 2)
        w3 = sum(1 for e in abs_errs if e <= 3)
        mount = next(e[1] for e in SESSIONS if e[0] == sid)
        print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {len(results):>3}  {n:>3}  {mae:>5.2f}  {fmt_signed(bias):>6}  {w2:>2}/{n}  {w3:>2}/{n}")

    print()
    print("--- SINGLE-FRAME FITS ONLY ---")
    print(f"{'Session':>30}  {'mount':>5}  {'d':>4}  {'N':>3}  {'det':>3}  {'MAE':>5}  {'bias':>6}  {'<=2':>4}  {'<=3':>4}")
    print("-" * 95)
    for (sid, d, results) in all_results:
        single = [r for r in results if r.get("reason") == "single_frame"]
        if not single:
            mount = next(e[1] for e in SESSIONS if e[0] == sid)
            print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {len(results):>3}    0     —       —      —     —")
            continue
        errs = [r["err"] for r in single]
        abs_errs = [abs(e) for e in errs]
        n = len(errs)
        mae = sum(abs_errs)/n
        bias = sum(errs)/n
        w2 = sum(1 for e in abs_errs if e <= 2)
        w3 = sum(1 for e in abs_errs if e <= 3)
        mount = next(e[1] for e in SESSIONS if e[0] == sid)
        print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {len(results):>3}  {n:>3}  {mae:>5.2f}  {fmt_signed(bias):>6}  {w2:>2}/{n}  {w3:>2}/{n}")

    print()
    print("--- HIGH-CONFIDENCE MULTI-FRAME FITS (fit_RMSE <= 2.0°) ---")
    print(f"{'Session':>30}  {'mount':>5}  {'d':>4}  {'N':>3}  {'det':>3}  {'MAE':>5}  {'bias':>6}  {'<=1':>4}  {'<=2':>4}")
    print("-" * 95)
    for (sid, d, results) in all_results:
        s = summarize(results, "", high_conf_only=True)
        mount = next(e[1] for e in SESSIONS if e[0] == sid)
        if s["n_det"] == 0:
            print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {s['n_total']:>3}    0     —       —      —     —")
        else:
            print(f"{sid:>30}  {mount:>5.1f}  {d:>4.1f}  {s['n_total']:>3}  {s['n_det']:>3}  {s['mae']:>5.2f}  {fmt_signed(s['bias']):>6}  {s['w1']:>2}/{s['n_det']}  {s['w2']:>2}/{s['n_det']}")

    # Write per-shot CSV
    out_csv = args.output_dir / "per_shot_all_sessions.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "d_initial_ft", "mount_deg", "shot_number", "club",
                    "tm_launch_v_deg", "estimated_alpha_deg", "error_deg",
                    "fit_rmse_deg", "n_frames_used", "reason"])
        for (sid, d, results) in all_results:
            mount = next(e[1] for e in SESSIONS if e[0] == sid)
            for r in results:
                w.writerow([
                    sid, d, mount, r["shot"], r["club"],
                    f"{r['tm']:.2f}",
                    "" if r["alpha"] is None else f"{r['alpha']:.2f}",
                    "" if r["err"] is None else f"{r['err']:.2f}",
                    "" if r["rmse"] is None else f"{r['rmse']:.3f}",
                    r["n_frames"], r["reason"],
                ])

    print()
    print(f"Wrote per-shot results to {out_csv}")


if __name__ == "__main__":
    main()
