#!/usr/bin/env python3
"""Isolate how much the trajectory-evolution geometry actually contributes.

For the two confirmed-geometry 18 deg sessions, for each shot:
  - find the ball frames exactly as the time-varying bin tracker does
  - GEOM:  alpha from the full trajectory fit (bin prediction + bearing trajectory)
  - NAIVE: alpha = mean(measured bearing over found frames) + mount
           (i.e. pretend the bearing already IS the launch angle)
Both use the SAME frames, so the only difference is whether we apply the
bearing-trajectory geometry. If MAE is the same, the trajectory math is idle.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tvbin_tracker as T

SESSIONS = [s for s in T.SESSIONS if s[0] in ("20260523_143732", "20260523_144415")]
SDIR = Path("/Users/john.pacino/openflight_sessions")


def frames_at_alpha(loaded, alpha, v_mph, d_ft, mount):
    """Return [(t, measured_bearing)] for frames found at this alpha."""
    found = []
    for t, mag, ang, noise in loaded:
        pb = T.exp_bin_at_t(float(alpha), t, v_mph, d_ft)
        if pb is None:
            continue
        idx = [(pb + o) % T.FFT_SIZE for o in range(-T.BIN_WIN, T.BIN_WIN + 1)]
        pk = idx[int(np.argmax(mag[idx]))]
        snr = 10 * math.log10(float(mag[pk]) / max(noise, 1e-9))
        if snr < T.SNR_MIN_DB:
            continue
        found.append((t, float(ang[pk])))
    return found


def run():
    print("=" * 92)
    print("GEOM (trajectory fit)  vs  NAIVE (mean raw bearing + mount) — same frames")
    print("=" * 92)
    geom_errs, naive_errs = [], []
    for key, mount, comp, jid, long_name, d_ft, net_ft in SESSIONS:
        fb = T.load_first_byte_times(
            Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"),
            long_name)
        tgt = T.load_comparison(SDIR / comp)
        rbc_map, kld7_map = T.load_session(SDIR / f"session_{jid}_range.jsonl")
        print(f"\n{key}  (mount={mount}°, d={d_ft}ft)")
        print(f"  {'shot':>4} {'TM':>6} {'GEOM':>7} {'g_err':>6} {'NAIVE':>7} {'n_err':>6} {'nf':>3} {'meanBrg':>8}")
        for sn in sorted(tgt):
            bs, tm = tgt[sn]
            rbc, frames, fb_ts = rbc_map.get(sn), kld7_map.get(sn), fb.get(sn)
            if not rbc or not frames or fb_ts is None:
                continue
            it = T.impact_ts_for(rbc, fb_ts)
            if it is None:
                continue
            loaded = T.decode_frames(frames, it)
            b = T.fit_shot(loaded, bs, d_ft, mount)
            if b is None:
                continue
            a_geom, nf, rmse, _ = b
            found = frames_at_alpha(loaded, a_geom, bs, d_ft, mount)
            if len(found) < 2:
                continue
            mean_brg = sum(br for _, br in found) / len(found)
            a_naive = mean_brg + mount
            g_err, n_err = a_geom - tm, a_naive - tm
            geom_errs.append(g_err)
            naive_errs.append(n_err)
            print(f"  {sn:>4} {tm:>6.2f} {a_geom:>7.2f} {g_err:>6.2f} "
                  f"{a_naive:>7.2f} {n_err:>6.2f} {nf:>3} {mean_brg:>8.2f}")

    def stat(errs, label):
        n = len(errs)
        mae = sum(abs(e) for e in errs) / n
        bias = sum(errs) / n
        rmse = math.sqrt(sum(e * e for e in errs) / n)
        print(f"  {label:>6}  n={n}  MAE={mae:.2f}  bias={bias:+.2f}  RMSE={rmse:.2f}")

    print("\n" + "=" * 92)
    stat(geom_errs, "GEOM")
    stat(naive_errs, "NAIVE")


if __name__ == "__main__":
    run()
