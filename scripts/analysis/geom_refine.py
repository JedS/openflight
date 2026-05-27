#!/usr/bin/env python3
"""Close the GEOM scatter gap — fair version: all fit methods on the SAME shots
and the SAME detected bearings, so we isolate the FIT from the DETECTION.

Per shot we detect ball frames once (alpha-independent bin from OPS radial speed),
extract bearing two ways (single-peak, weighted +/-2 bins), then fit alpha by:
  LSQ   - smooth least-squares on single-peak bearings (fine 0.1 grid)
  LSQW  - same on weighted bearings
  CUR   - the current tracker's coupled grid-search (reference)
  CONST - mean_bearing + mount + 7.13 cheat constant (overfit floor reference)
Shots kept = those where detection finds >=2 frames (common set for all).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tvbin_tracker as T

SESSIONS = [s for s in T.SESSIONS if s[0] in ("20260523_143732", "20260523_144415")]
SDIR = Path("/Users/john.pacino/openflight_sessions")
TIMING = Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv")
REF_ALPHA = 15.0
WBINS = 2
CONST = 7.13


def detect(loaded, v_mph, d_ft, ref_alpha):
    """alpha-independent detection -> [(t, beta_single, beta_weighted)]."""
    found = []
    for t, mag, ang, noise in loaded:
        pb = T.exp_bin_at_t(ref_alpha, t, v_mph, d_ft)
        if pb is None:
            continue
        idx = [(pb + o) % T.FFT_SIZE for o in range(-T.BIN_WIN, T.BIN_WIN + 1)]
        pk = idx[int(np.argmax(mag[idx]))]
        snr = 10 * math.log10(float(mag[pk]) / max(noise, 1e-9))
        if snr < T.SNR_MIN_DB:
            continue
        wnb = [(pk + o) % T.FFT_SIZE for o in range(-WBINS, WBINS + 1)]
        w = np.array([mag[b] for b in wnb], dtype=float)
        angs = np.array([ang[b] for b in wnb], dtype=float)
        bw = float(np.sum(w * angs) / np.sum(w))
        found.append((t, float(ang[pk]), bw))
    return found


def fit_lsq(pairs, v_mph, d_ft, mount):
    best = None
    for alpha in np.arange(0.0, 35.001, 0.1):
        ss = sum((b - T.pred_bearing(float(alpha), t, v_mph, d_ft, mount)) ** 2 for t, b in pairs)
        if best is None or ss < best[1]:
            best = (float(alpha), ss)
    return best[0]


def run():
    rows = []
    for key, mount, comp, jid, long_name, d_ft, net_ft in SESSIONS:
        fb = T.load_first_byte_times(TIMING, long_name)
        tgt = T.load_comparison(SDIR / comp)
        rbc_map, kld7_map = T.load_session(SDIR / f"session_{jid}_range.jsonl")
        for sn in sorted(tgt):
            bs, tm = tgt[sn]
            rbc, frames, fb_ts = rbc_map.get(sn), kld7_map.get(sn), fb.get(sn)
            if not rbc or not frames or fb_ts is None:
                continue
            it = T.impact_ts_for(rbc, fb_ts)
            if it is None:
                continue
            loaded = T.decode_frames(frames, it)
            found = detect(loaded, bs, d_ft, REF_ALPHA)
            if len(found) < 2:
                continue
            cur = T.fit_shot(loaded, bs, d_ft, mount)
            cur_a = cur[0] if cur else None
            single = [(t, bs_) for t, bs_, _ in found]
            weighted = [(t, bw) for t, _, bw in found]
            a_lsq = fit_lsq(single, bs, d_ft, mount)
            a_lsqw = fit_lsq(weighted, bs, d_ft, mount)
            mean_brg = sum(b for _, b in single) / len(single)
            a_const = mean_brg + mount + CONST
            rows.append((key[-6:], sn, tm, len(found), cur_a, a_lsq, a_lsqw, a_const))

    hdr = f"{'sess':>6} {'sn':>3} {'TM':>6} {'nf':>2} | {'CUR':>6} {'LSQ':>6} {'LSQW':>6} {'CONST':>6}"
    print(hdr); print("-" * len(hdr))
    cols = {"CUR": [], "LSQ": [], "LSQW": [], "CONST": []}
    for sess, sn, tm, nf, cur_a, a_lsq, a_lsqw, a_const in rows:
        def e(a):
            return None if a is None else a - tm
        for name, a in (("CUR", cur_a), ("LSQ", a_lsq), ("LSQW", a_lsqw), ("CONST", a_const)):
            if a is not None:
                cols[name].append(a - tm)
        fmt = lambda a: "   —  " if a is None else f"{a:6.2f}"
        print(f"{sess:>6} {sn:>3} {tm:6.2f} {nf:>2} | {fmt(cur_a)} {fmt(a_lsq)} {fmt(a_lsqw)} {fmt(a_const)}")

    print("\n" + f"{'method':>6}  {'n':>3}  {'MAE':>5}  {'bias':>6}  {'RMSE':>5}")
    for name, errs in cols.items():
        n = len(errs)
        mae = sum(abs(x) for x in errs) / n
        bias = sum(errs) / n
        rmse = math.sqrt(sum(x * x for x in errs) / n)
        print(f"{name:>6}  {n:>3}  {mae:5.2f}  {bias:+6.2f}  {rmse:5.2f}")

    # Shrinkage sweep: MAP estimate = LSQ residual + lam*(alpha-alpha0)^2.
    # lam=0 is pure geometry; large lam -> pulls toward alpha0 (club-agnostic 15).
    # Tests the bias-variance story: does ANY honest shrinkage beat pure geometry?
    print("\nSHRINKAGE (MAP, alpha0=15, club-AGNOSTIC weak prior):")
    print(f"  {'lambda':>7}  {'MAE':>5}  {'bias':>6}  {'RMSE':>5}")

    def fit_map(pairs, v, d, mount, alpha0, lam):
        best = None
        for alpha in np.arange(0.0, 35.001, 0.1):
            ss = sum((b - T.pred_bearing(float(alpha), t, v, d, mount)) ** 2 for t, b in pairs)
            ss += lam * (float(alpha) - alpha0) ** 2
            if best is None or ss < best[1]:
                best = (float(alpha), ss)
        return best[0]

    # rebuild per-shot context for the MAP sweep
    ctx = []
    for key, mount, comp, jid, long_name, d_ft, net_ft in SESSIONS:
        fb = T.load_first_byte_times(TIMING, long_name)
        tgt = T.load_comparison(SDIR / comp)
        rbc_map, kld7_map = T.load_session(SDIR / f"session_{jid}_range.jsonl")
        for sn in sorted(tgt):
            bs, tm = tgt[sn]
            rbc, frames, fb_ts = rbc_map.get(sn), kld7_map.get(sn), fb.get(sn)
            if not rbc or not frames or fb_ts is None:
                continue
            it = T.impact_ts_for(rbc, fb_ts)
            if it is None:
                continue
            found = detect(T.decode_frames(frames, it), bs, d_ft, REF_ALPHA)
            if len(found) < 2:
                continue
            ctx.append((tm, bs, d_ft, mount, [(t, s) for t, s, _ in found]))

    for lam in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0):
        errs = [fit_map(pairs, bs, d, m, 15.0, lam) - tm for tm, bs, d, m, pairs in ctx]
        n = len(errs)
        mae = sum(abs(x) for x in errs) / n
        bias = sum(errs) / n
        rmse = math.sqrt(sum(x * x for x in errs) / n)
        print(f"  {lam:>7.1f}  {mae:5.2f}  {bias:+6.2f}  {rmse:5.2f}")


if __name__ == "__main__":
    run()
