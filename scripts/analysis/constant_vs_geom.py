#!/usr/bin/env python3
"""Could a fixed offset replace the trajectory geometry?

Two questions:
  A) In-sample: if we just remove the mean NAIVE bias as a constant, how close
     do we get vs the geometry? (best-case for the constant — it's fit to the
     same shots, so this is the floor of its error.)
  B) Is the geometry's correction actually CONSTANT across launch angle / range
     / speed? We sweep alpha, D, speed and see how much the "gap" the geometry
     applies (alpha - [mean_bearing + mount]) moves. If it moves a lot, a single
     hardcoded factor only works for the regime it was tuned in (e.g. 7-iron @ 5.5ft).
"""
from __future__ import annotations

import math

import tvbin_tracker as T

# NAIVE per-shot errors from geom_vs_naive.py (raw bearing + mount, vs TM)
NAIVE_ERRS = [-8.15, -8.82, -4.78, -7.64, -7.58, -8.10, -3.94, -6.81, -8.62, -6.88]


def part_a():
    n = len(NAIVE_ERRS)
    bias = sum(NAIVE_ERRS) / n
    resid = [e - bias for e in NAIVE_ERRS]  # after subtracting the best constant
    mae = sum(abs(r) for r in resid) / n
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    std = math.sqrt(sum((e - bias) ** 2 for e in NAIVE_ERRS) / n)
    print("A) CONSTANT-OFFSET, in-sample (subtract the mean bias from NAIVE)")
    print(f"   best constant = {-bias:+.2f}°   residual MAE={mae:.2f}  RMSE={rmse:.2f}  std={std:.2f}")
    print(f"   (compare: GEOM MAE=2.34, RMSE=2.64 — so the constant's FLOOR is ~{mae:.1f}°,")
    print(f"    but only because it's fit to these exact 10 shots.)")


def gap(alpha, v_mph, d_ft, mount, frame_times_ms):
    """The correction the geometry applies = alpha - (mean predicted bearing + mount)."""
    brgs = [T.pred_bearing(alpha, t / 1000.0, v_mph, d_ft, mount) for t in frame_times_ms]
    naive = sum(brgs) / len(brgs) + mount
    return alpha - naive


def part_b():
    mount = 18.0
    print("\nB) IS THE GEOMETRY'S CORRECTION CONSTANT?  (gap = alpha - [mean_bearing + mount])")
    print("   Frames assumed at t = 45 and 70 ms (typical 2-frame window).")
    ft = [45.0, 70.0]

    print("\n   vs LAUNCH ANGLE (v=100mph, D=5.5ft):")
    for a in (10, 15, 20, 25, 30, 35):
        print(f"     alpha={a:>2}°  ->  correction needed = {gap(float(a), 100.0, 5.5, mount, ft):+.2f}°")

    print("\n   vs RANGE D (alpha=17°, v=100mph):")
    for d in (4.0, 5.0, 5.5, 6.0, 7.0, 9.0):
        print(f"     D={d:>3}ft  ->  correction needed = {gap(17.0, 100.0, d, mount, ft):+.2f}°")

    print("\n   vs BALL SPEED (alpha=17°, D=5.5ft):")
    for v in (70, 90, 110, 130, 150):
        print(f"     v={v:>3}mph ->  correction needed = {gap(17.0, float(v), 5.5, mount, ft):+.2f}°")

    print("\n   vs FRAME TIMING (alpha=17°, v=100mph, D=5.5ft) — single frame at t:")
    for t in (30, 45, 60, 75, 90):
        print(f"     t={t:>2}ms  ->  correction needed = {gap(17.0, 100.0, 5.5, mount, [float(t)]):+.2f}°")


if __name__ == "__main__":
    part_a()
    part_b()
