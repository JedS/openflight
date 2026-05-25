# K-LD7 Vertical Candidate Ranker Analysis

## Summary

This analysis revisits the K-LD7 vertical launch angle replay using the May 22
and May 23 TrackMan comparison sessions.

The important finding is that the K-LD7 raw data often contains a good vertical
launch candidate, but candidate generation and selection must use the same
effective angle offset as the replay baseline:

```text
effective_offset = vertical_mount_angle + software_offset

0 deg mount  -> +8 deg
8 deg mount  -> +16 deg
18 deg mount -> +26 deg
```

Before applying that effective offset during candidate-pool generation, the
candidate oracle looked much worse and did not reproduce the strong May 23
18-degree results. After applying it, candidate coverage improved substantially.

## Terminology

### Candidate oracle

`candidate_oracle` is an offline diagnostic ceiling.

For each shot, it:

1. Generates all plausible K-LD7 launch candidates.
2. Looks at the TrackMan vertical launch angle.
3. Picks the K-LD7 candidate closest to TrackMan.
4. Reports the remaining error.

This **requires TrackMan data** and cannot run in production. Its purpose is to
answer one question:

> Did the K-LD7 candidate pool contain a good answer?

If candidate oracle is good, the sensor/candidate generation path likely found
the ball and the remaining problem is production candidate selection.

If candidate oracle is bad, selector changes alone cannot fix the shot because
the right answer was not present in the candidate pool.

### Learned error ranker

`learned_error` is a TrackMan-trained, leave-one-session-out candidate ranker.
It uses TrackMan only during offline training/evaluation. The features used for
selection are runtime-available radar features:

- predicted launch angle
- mount angle
- OPS ball speed
- support across candidate sweeps
- nearby candidate support
- confidence
- SNR
- frame count
- per-frame consistency metrics
- expected launch plausibility
- candidate fallback/source flags

The current implementation is analysis-only. It is not yet production code.

## What Changed In The Analysis

The analysis now generates geometry candidate buckets with:

```python
angle_offset_deg = mount_deg + 8.0
```

This matches the effective offset used by the existing replay/baseline data.
The earlier geometry candidate cache used `angle_offset_deg=0.0`, which meant
many good candidates were shifted into the wrong launch-angle lane.

Two offline scripts support this analysis:

- `scripts/analysis/geometry_candidate_selector.py`
- `scripts/analysis/track_candidate_ranker.py`

## Reproduction

Run from the repo root:

```bash
uv run --no-sync env PYTHONPATH=src:scripts/analysis \
  python scripts/analysis/geometry_candidate_selector.py \
  --output-dir /Users/john.pacino/Desktop/openflight_sessions/_analysis_geometry_candidate_selector_effective_offset
```

Then run the candidate ranker on that corrected bucket cache:

```bash
uv run --no-sync env PYTHONPATH=src:scripts/analysis \
  python scripts/analysis/track_candidate_ranker.py \
  --bucket-csv /Users/john.pacino/Desktop/openflight_sessions/_analysis_geometry_candidate_selector_effective_offset/bucket_geometry_candidate_selector.csv \
  --output-dir /Users/john.pacino/Desktop/openflight_sessions/_analysis_track_candidate_ranker_effective_offset
```

Lint check:

```bash
uv run --no-sync ruff check \
  scripts/analysis/geometry_candidate_selector.py \
  scripts/analysis/track_candidate_ranker.py
```

Result:

```text
All checks passed.
```

## Results

### Overall

| Strategy | Rows | MAE | Bias | RMSE | P90 abs | <=8 deg |
|---|---:|---:|---:|---:|---:|---:|
| latest | 120 | 5.562 | 0.557 | 7.795 | 12.400 | 97/120 |
| fixed | 120 | 5.316 | -0.591 | 7.301 | 11.600 | 98/120 |
| prev_oracle | 120 | 2.940 | -0.497 | 3.816 | 5.800 | 116/120 |
| geom_v5 | 120 | 3.594 | -0.861 | 4.582 | 7.200 | 111/120 |
| manual_track | 120 | 3.185 | -1.685 | 4.371 | 7.300 | 112/120 |
| learned_error | 120 | 2.463 | -0.623 | 3.584 | 5.900 | 114/120 |
| learned_pairwise | 120 | 2.831 | -0.912 | 4.294 | 7.000 | 112/120 |
| candidate_oracle | 120 | 1.162 | -0.333 | 2.016 | 3.500 | 120/120 |

### By mount angle

| Mount | Shots | fixed MAE | geom_v5 MAE | learned_error MAE | candidate_oracle MAE |
|---:|---:|---:|---:|---:|---:|
| 0 deg | 83 | 5.447 | 3.412 | 2.688 | 1.148 |
| 8 deg | 13 | 6.762 | 4.708 | 2.946 | 2.423 |
| 18 deg | 24 | 4.079 | 3.621 | 1.425 | 0.529 |

### By session

| Session | Shots | fixed MAE | geom_v5 MAE | learned_error MAE | candidate_oracle MAE |
|---|---:|---:|---:|---:|---:|
| `20260522_135647_0deg_7iron_16shots` | 16 | 3.875 | 2.975 | 1.869 | 1.275 |
| `20260522_141038_8deg_7iron_13shots` | 13 | 6.762 | 4.708 | 2.946 | 2.423 |
| `20260522_141949_18deg_7iron_11shots` | 11 | 4.845 | 4.845 | 1.327 | 0.718 |
| `20260522_142538_0deg_4club_68shots` | 67 | 5.822 | 3.516 | 2.884 | 1.118 |
| `20260523_143732_18deg_7iron_8shots` | 8 | 3.513 | 2.938 | 1.513 | 0.175 |
| `20260523_144415_18deg_7iron_5shots_cleaned` | 5 | 3.300 | 2.020 | 1.500 | 0.680 |

### Candidate coverage

Candidate coverage is the most important diagnostic in this run.

| Group | Shots | Oracle MAE | <=1 deg | <=2 deg | <=3 deg | <=5 deg |
|---|---:|---:|---:|---:|---:|---:|
| all shots | 120 | 1.162 | 83 | 96 | 104 | 114 |
| 0 deg mount | 83 | 1.148 | 58 | 66 | 72 | 80 |
| 8 deg mount | 13 | 2.423 | 5 | 7 | 8 | 10 |
| 18 deg mount | 24 | 0.529 | 20 | 23 | 24 | 24 |
| non-driver | 96 | 1.032 | 69 | 80 | 85 | 92 |
| 7-iron only | 64 | 1.222 | 42 | 51 | 55 | 60 |

## Key Finding

With effective-offset candidate generation, the K-LD7 candidate pool is much
stronger than previous geometry runs suggested.

The earlier hard failures in the May 22 18-degree session now have good
candidates:

| Shot | TrackMan | Previous fixed/latest style value | Candidate oracle |
|---:|---:|---:|---:|
| 8 | 18.8 deg | 29.5 deg | 19.7 deg |
| 10 | 18.4 deg | 28.7 deg | 17.0 deg |

That means those shots are no longer best explained as missing sensor data.
They are candidate selection failures.

## Interpretation

The MIMO-style takeaway still applies even though K-LD7 is not a true MIMO
array: this should be treated as a track/candidate-selection problem, not a
single strongest-angle problem.

Good candidates tend to be supported by multiple pieces of evidence:

- plausible launch angle for mount and club
- repeated or nearby support across timing/candidate sweeps
- frame-level consistency
- reasonable confidence/SNR
- agreement with OPS speed constraints

The current `learned_error` ranker is not production-ready, but it proves that a
runtime-feature selector can get much closer to the TrackMan-selected oracle
without using TrackMan at inference time.

## Recommendation

Use this PR to document and preserve the offline analysis path, but do not ship
the candidate oracle or TM-trained weights as production behavior yet.

Recommended next implementation step:

1. Keep effective-offset candidate generation in the offline replay tools.
2. Turn the best runtime-only features into a deterministic selector that can be
   reviewed and tested.
3. Validate the selector with leave-one-session-out tests before moving it into
   `openflight.kld7`.
4. Add regression tests around the May 22 18-degree hard shots so candidates
   near `18-20 deg` are not hidden by the high-angle `28-30 deg` alternatives.

The practical goal should be to make production selection approach the
`learned_error` result first, then continue toward the candidate-oracle ceiling
as more geometry and camera-derived setup measurements become available.
