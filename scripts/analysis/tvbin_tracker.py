#!/usr/bin/env python3
"""Time-varying-bin trajectory tracker for K-LD7 vertical launch angle.

Improves on the fixed-bin trajectory fitter by predicting where the ball's
Doppler bin should be AT EACH FRAME TIME (the bin shifts during flight because
the radar measures radial velocity, which recovers from ~93% to ~99% of true
speed as the ball climbs away). For each candidate launch angle α it:

  1. Predicts the ball position (geometry) at each frame time t
  2. Computes radial velocity → expected Doppler bin
  3. Searches a tight window at that bin for a strong peak (the ball)
  4. Extracts the per-bin angle
  5. Scores how well the found angles fit α's predicted bearing trajectory

The α that finds the most frames with the lowest bearing residual wins. Uses
no training; everything is geometry + OPS impact timing.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
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

FFT_SIZE = 2048
MPH_TO_FTS = 1.46667
BALL_ABOVE_RADAR_FT = -4.0 / 12.0   # ball 4" below radar center
FIRST_BYTE_TRIGGER_DELAY_MS = 68.0
IMPACT_EXTRA_OFFSET_MS = -10.0
BIN_WIN = 8
SNR_MIN_DB = 8.0
T_MIN_MS = 0.0
T_MAX_MS = 110.0

# (catalog_key, mount_deg, comparison_file, jsonl_id, long_name, d_ft, net_ft)
SESSIONS = [
    ("20260522_135647",         0.0,  "comparison_20260522_135647.csv",        "20260522_135647", "20260522_135647_0deg_7iron_16shots",          6.0, 15.0),
    ("20260522_141038",         8.0,  "comparison_20260522_141038.csv",        "20260522_141038", "20260522_141038_8deg_7iron_13shots",          6.0, 15.0),
    ("20260522_141949",        18.0,  "comparison_20260522_141949.csv",        "20260522_141949", "20260522_141949_18deg_7iron_11shots",         6.0, 15.0),
    ("20260522_142538__7iron",  0.0,  "comparison_20260522_142538_7iron.csv",  "20260522_142538", "20260522_142538_0deg_4club_68shots",          6.0, 15.0),
    ("20260522_142538__8iron",  0.0,  "comparison_20260522_142538_8iron.csv",  "20260522_142538", "20260522_142538_0deg_4club_68shots",          6.0, 15.0),
    ("20260522_142538__9iron",  0.0,  "comparison_20260522_142538_9iron.csv",  "20260522_142538", "20260522_142538_0deg_4club_68shots",          6.0, 15.0),
    ("20260522_142538__driver", 0.0,  "comparison_20260522_142538_driver.csv", "20260522_142538", "20260522_142538_0deg_4club_68shots",          6.0, 15.0),
    ("20260523_143732",        18.0,  "comparison_20260523_143732.csv",        "20260523_143732", "20260523_143732_18deg_7iron_8shots",          6.0, 12.0),
    ("20260523_144415",        18.0,  "comparison_20260523_144415.csv",        "20260523_144415", "20260523_144415_18deg_7iron_5shots_cleaned", 5.0, 10.0),
]


def _to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_first_byte_times(csv_path, long_name):
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("session") != long_name:
                continue
            try:
                out[int(row["json_shot_no"])] = datetime.fromisoformat(row["first_byte_ts"]).timestamp()
            except (KeyError, ValueError, TypeError):
                pass
    return out


def load_comparison(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("match_quality") != "good":
                continue
            sn = _to_float(row.get("shot_number_of"))
            tm = _to_float(row.get("launch_v_tm"))
            bs = _to_float(row.get("ball_speed_of"))
            if sn is None or tm is None or bs is None:
                continue
            out[int(sn)] = (bs, tm)
    return out


def load_session(jsonl_path):
    rbc, kld7 = {}, {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sn = d.get("shot_number")
            if sn is None:
                continue
            if d.get("type") == "rolling_buffer_capture":
                rbc[int(sn)] = d
            elif d.get("type") == "kld7_buffer" and d.get("orientation") == "vertical":
                if d.get("frames"):
                    kld7[int(sn)] = d["frames"]
    return rbc, kld7


def impact_ts_for(rbc, fb_ts):
    bms = rbc.get("ball_timestamp_ms")
    tom = rbc.get("trigger_offset_ms")
    if bms is None or tom is None:
        return None
    return fb_ts - FIRST_BYTE_TRIGGER_DELAY_MS / 1000.0 - (tom - bms) / 1000.0 + IMPACT_EXTRA_OFFSET_MS / 1000.0


def decode_frames(frames, impact_ts):
    """Precompute (t_s, mag[], angles[], noise) per frame in the flight window."""
    out = []
    for raw in frames:
        ts = raw.get("timestamp")
        if ts is None:
            continue
        t = float(ts) - impact_ts
        if t <= 0 or t * 1000 < T_MIN_MS or t * 1000 > T_MAX_MS:
            continue
        b64 = raw.get("radc_b64")
        if not isinstance(b64, str):
            continue
        try:
            radc = base64.b64decode(b64, validate=True)
        except ValueError:
            continue
        if len(radc) != RADC_PAYLOAD_BYTES:
            continue
        p = parse_radc_payload(radc)
        F1 = compute_fft_complex(to_complex_iq(p["f1a_i"], p["f1a_q"]), fft_size=FFT_SIZE)
        F2 = compute_fft_complex(to_complex_iq(p["f2a_i"], p["f2a_q"]), fft_size=FFT_SIZE)
        mag = np.abs(F1) + np.abs(F2)
        ang = per_bin_angle_deg(F1, F2)
        noise = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
        out.append((t, mag, ang, noise))
    return out


def exp_bin_at_t(alpha, t_s, v_mph, d_ft):
    v = v_mph * MPH_TO_FTS
    a = math.radians(alpha)
    x = d_ft + v * math.cos(a) * t_s
    y = BALL_ABOVE_RADAR_FT + v * math.sin(a) * t_s
    r = math.hypot(x, y)
    if r <= 0:
        return None
    v_radial = (v * math.cos(a) * x + v * math.sin(a) * y) / r
    return expected_ball_bin_from_speed(v_radial / MPH_TO_FTS, fft_size=FFT_SIZE)


def pred_bearing(alpha, t_s, v_mph, d_ft, mount):
    v = v_mph * MPH_TO_FTS
    a = math.radians(alpha)
    x = d_ft + v * math.cos(a) * t_s
    y = BALL_ABOVE_RADAR_FT + v * math.sin(a) * t_s
    return math.degrees(math.atan2(y, x)) - mount


def fit_shot(loaded, v_mph, d_ft, mount):
    best = None  # (alpha, n_frames, rmse, score)
    for alpha in np.arange(0.0, 35.01, 0.5):
        residsq = []
        for t, mag, ang, noise in loaded:
            pb = exp_bin_at_t(float(alpha), t, v_mph, d_ft)
            if pb is None:
                continue
            idx = [(pb + o) % FFT_SIZE for o in range(-BIN_WIN, BIN_WIN + 1)]
            pk = idx[int(np.argmax(mag[idx]))]
            snr = 10 * math.log10(float(mag[pk]) / max(noise, 1e-9))
            if snr < SNR_MIN_DB:
                continue
            residsq.append((ang[pk] - pred_bearing(float(alpha), t, v_mph, d_ft, mount)) ** 2)
        if len(residsq) >= 2:
            rmse = math.sqrt(sum(residsq) / len(residsq))
            score = (len(residsq), -rmse)
            if best is None or score > best[3]:
                best = (float(alpha), len(residsq), rmse, score)
    return best


def summarize(errs, label, n_total):
    if not errs:
        return f"{label:>28}  n={n_total:>2}  det= 0   —"
    n = len(errs)
    mae = sum(abs(e) for e in errs) / n
    bias = sum(errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    w2 = sum(1 for e in errs if abs(e) <= 2)
    w3 = sum(1 for e in errs if abs(e) <= 3)
    sb = "+" if bias >= 0 else "-"
    return (f"{label:>28}  n={n_total:>2}  det={n:>2}  MAE={mae:5.2f}  "
            f"bias={sb}{abs(bias):.2f}  RMSE={rmse:4.2f}  <=2:{w2:>2}/{n}  <=3:{w3:>2}/{n}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions-dir", type=Path, default=Path("/Users/john.pacino/openflight_sessions"))
    p.add_argument("--timing-csv", type=Path,
                   default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_tvbin_tracker/per_shot_tvbin.csv"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("TIME-VARYING BIN TRACKER — all sessions (eff_mount = mount, no boresight fudge)")
    print("=" * 100)

    all_rows = []
    by_mount = {}
    for key, mount, comp, jsonl_id, long_name, d_ft, net_ft in SESSIONS:
        fb = load_first_byte_times(args.timing_csv, long_name)
        tgt = load_comparison(args.sessions_dir / comp)
        rbc_map, kld7_map = load_session(args.sessions_dir / f"session_{jsonl_id}_range.jsonl")
        errs = []
        for sn in sorted(tgt):
            bs, tm = tgt[sn]
            rbc = rbc_map.get(sn)
            frames = kld7_map.get(sn)
            fb_ts = fb.get(sn)
            if not rbc or not frames or fb_ts is None:
                all_rows.append((key, mount, sn, tm, None, None, None, 0)); continue
            it = impact_ts_for(rbc, fb_ts)
            if it is None:
                all_rows.append((key, mount, sn, tm, None, None, None, 0)); continue
            loaded = decode_frames(frames, it)
            b = fit_shot(loaded, bs, d_ft, mount)
            if b is None:
                all_rows.append((key, mount, sn, tm, None, None, None, 0)); continue
            a, nf, rmse, _ = b
            err = a - tm
            errs.append(err)
            by_mount.setdefault(mount, []).append(err)
            all_rows.append((key, mount, sn, tm, a, err, rmse, nf))
        print(summarize(errs, key, len(tgt)))

    print("-" * 100)
    print("\nBY MOUNT TILT (tests the multipath-vs-tilt hypothesis):")
    for m in sorted(by_mount):
        print(summarize(by_mount[m], f"mount={m}°", len(by_mount[m])))

    overall = [r[5] for r in all_rows if r[5] is not None]
    print("\n" + summarize(overall, "ALL SESSIONS", sum(len(load_comparison(args.sessions_dir / s[2])) for s in SESSIONS)))

    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "mount_deg", "shot", "tm_launch_v", "est_alpha", "err", "fit_rmse", "n_frames"])
        for key, mount, sn, tm, a, err, rmse, nf in all_rows:
            w.writerow([key, mount, sn, f"{tm:.2f}",
                        "" if a is None else f"{a:.2f}",
                        "" if err is None else f"{err:.2f}",
                        "" if rmse is None else f"{rmse:.3f}", nf])
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
