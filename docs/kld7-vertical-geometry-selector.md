# K-LD7 Vertical Geometry Selector

This document describes the geometry-based K-LD7 vertical launch-angle selector.
It is intended to make the production behavior understandable without relying on
session-specific analysis or TrackMan comparisons.

## Goal

The K-LD7 raw RADC path measures target bearing from receive-channel phase
difference. A bearing is not the same thing as a launch angle: it is the angle
from the radar to the ball's current position at a specific time after impact.

The geometry selector estimates vertical launch angle by combining:

- K-LD7 per-frame bearing
- OPS243 ball speed
- impact timestamp
- radar-to-ball distance
- radar mount tilt
- radar/ball height relationship
- K-LD7 boresight or bearing offset

The selector prefers a two-frame trajectory fit. If only one strong in-flight
frame is available, it can produce a low-confidence single-frame geometry
fallback.

## Required Inputs

The geometry selector runs through `extract_launch_angle(...)` in
`src/openflight/kld7/radc.py`.

Required for geometry mode:

- `frames`: K-LD7 frames containing raw `radc` payloads.
- `ops243_ball_speed_mph`: OPS243 measured ball speed.
- `shot_timestamp` or `impact_timestamp`: impact time in epoch seconds.
- `mount_deg`: physical vertical radar mount tilt in degrees.
- `distance_ft`: ball-to-radar-front distance in feet.
- `orientation="vertical"`.
- `vertical_estimator="geometry"`.

For live range sessions, the kiosk shortcut is:

```bash
scripts/start-kiosk.sh --kld7-geometry
```

This enables the vertical K-LD7 and forwards the current field defaults:
`--kld7-vertical-estimator geometry`, `--kld7-mount-tilt 10`,
`--kld7-ball-distance 5`, and `--kld7-angle-offset 2.5`. The individual
`--kld7-*` flags can still be passed to override the preset.

Important optional/configuration inputs:

- `angle_offset_deg`: bearing/boresight correction applied to raw K-LD7 angle
  before geometry fitting.
- `speed_tolerance_mph`: OPS-speed search half-width for candidate bins.
- `spectrum_source`: spectrum used to choose the Doppler bin. `sum12` combines
  the F1A and F2A receive-channel magnitudes before angle extraction.
- `ops_bin_outlier_tol`: maximum bin distance used by the OPS-near peak search.
- `ops_anchored_peak_min_snr`: SNR required for a primary OPS-bin anchor.
- `centroid_floor_frac`: controls centroid angle extraction around the selected
  spectral peak.

## Coordinate Terms

`mount_deg` and `angle_offset_deg` correct different things.

`mount_deg` is the mechanical direction of the radar relative to the room and
ball flight. It is used by the trajectory model when predicting what bearing
the radar should see for a given launch angle.

`angle_offset_deg` is a bearing correction applied to the raw phase-derived
K-LD7 angle:

```text
corrected_bearing = raw_kld7_angle + angle_offset_deg
```

This offset should be treated as a boresight/electrical-zero calibration term,
not as a launch-angle fudge. It accounts for the possibility that raw RADC phase
angle zero does not exactly equal the radar's mechanical zero.

## Frame Discovery

The selector starts by deriving a Doppler bin range from OPS ball speed. Golf
ball speeds can alias at the K-LD7 speed setting, so the expected bin may appear
in the wrapped portion of the FFT spectrum.

For each RADC frame:

1. Parse the RADC payload into F1A/F2A/F1B channels.
2. FFT the receive channels.
3. Build the configured spectrum (`f1a`, `f2a`, or `sum12`).
4. Search for a peak near the OPS-expected Doppler bin.
5. Compute SNR against the frame noise floor.
6. Compute per-bin angle from F1A/F2A phase difference.
7. Compute the selected peak angle, normally as a magnitude-weighted centroid
   around the spectral peak.

Before per-frame candidate selection, the code finds impact-energy frames and
groups nearby frames into shot events. For each group, it examines the impact
frames plus a small surrounding window.

## Candidate Gates

For vertical geometry, per-frame candidates are passed through a rule stack:

- Time gate: candidate must be roughly `20-100 ms` after impact.
- OPS-bin gate: candidate bin must be within `50` bins of the expected OPS bin.
- Strong-anchor requirement: at least one non-weak OPS-bin candidate is required.
- Weak adjacent candidates may be retained only as context around a strong
  anchor.

There are two SNR concepts:

- Strong anchor: must meet `ops_anchored_peak_min_snr`.
- Weak adjacent frame: may be kept at `SNR >= 3.0`, but only if it is adjacent
  to a strong anchor and passes the trajectory sanity checks.

Weak frames are never allowed to start a geometry fit by themselves.

## Anchor Selection

After primary gates pass, the selector chooses one strong anchor frame.

Anchor sorting prefers:

1. smallest OPS-bin error
2. highest SNR
3. highest phase coherence

The anchor is the most trusted frame for the shot.

## Two-Frame Geometry

The selector tries to pair the anchor with an adjacent frame:

- previous gated frame, if present
- otherwise next gated frame, if present

The pair must pass a rising-bearing rule:

```text
later_bearing > earlier_bearing
```

For a ball launched upward, the vertical bearing should increase over early
flight. If the adjacent frame moves the bearing downward, it is rejected as
likely clutter, multipath, club contamination, or timing-adjacent noise.

The adjacent frame SNR threshold is:

- weak adjacent: `>= 3.0`
- normal adjacent: `>= max(2.0, 50% of anchor SNR)`

If a valid pair is found, the selected per-frame data is fit with
`fit_launch_angle_geometric(...)`.

The fit searches launch angle from `0-45 deg` in `0.1 deg` steps and minimizes
weighted bearing residual:

```text
predicted_bearing(alpha, t, speed, distance, mount)
```

The trajectory model treats the ball as moving in a straight line over the short
in-flight window:

```text
x = distance_ft + speed_ft_s * cos(alpha) * t
y = ball_above_radar_ft + speed_ft_s * sin(alpha) * t

bearing = atan2(y, x) - mount_deg
```

The current default vertical origin assumes the ball starts about `4 in` below
the radar center:

```text
ball_above_radar_ft = -4 / 12
```

Gravity is ignored because the modeled time window is very short.

If a weak adjacent frame participates in the two-frame fit, the fit is accepted
only when the geometry RMSE is small. A high-RMSE weak pair is rejected and the
selector can fall back to the single-frame path.

## Single-Frame Geometry Fallback

If a two-frame geometry fit is unavailable, the selector can attempt a
single-frame geometry fallback.

This path is intentionally narrow:

- geometry mode must be enabled
- vertical orientation is required
- ball speed, impact timing, mount tilt, and distance must all be present
- exactly one strong non-weak frame must remain after rule selection
- weak adjacent frames cannot create a single-frame result
- the solved angle must stay inside the physical vertical bounds

The single-frame solver uses the same predicted-bearing model as the two-frame
fit, but with one bearing observation. It grid-searches `0-45 deg` and accepts
the angle only if the predicted bearing can match the observed bearing within
the configured residual limit.

The result is marked separately:

```text
estimator = "geometry_single_frame"
```

Single-frame geometry is lower confidence because bearing, timing, and setup
errors cannot be averaged across multiple frames. The confidence is capped below
the strict-accept threshold so downstream selection treats it as a soft candidate.

## Fallbacks And Bounds

If geometry cannot run, the function can still return the legacy naive estimate:

```text
naive_angle = weighted_average_raw_bearing + angle_offset_deg
```

For vertical orientation, final output is bounded to:

```text
0 deg <= launch_angle <= 45 deg
```

Angles outside that range are rejected.

## Confidence

Confidence is estimator-dependent.

Two-frame geometry:

- blends average SNR with fit RMSE
- lower RMSE increases confidence

Single-frame geometry:

- uses SNR and one-frame residual
- confidence is capped below strict radar acceptance

Naive:

- follows legacy SNR/frame-count confidence behavior

The estimator name is included in the result payload so callers can distinguish
between true trajectory geometry, single-frame fallback, and legacy naive output.

## Logging And Diagnostics

The selector logs:

- rule-stack pass/fail reasons for candidate frames
- anchor frame choice
- pair-rule failures
- weak adjacent candidate retention
- geometry RMSE rejection
- single-frame geometry fallback details
- OPS-bin penalty warnings when selected bins are far from the OPS-expected bin

Useful result fields include:

- `launch_angle_deg`
- `raw_angle_deg`
- `angle_offset_deg`
- `estimator`
- `geom_fit_rmse_deg`
- `geom_single_frame_resid_deg`
- `weak_adjacent_frame_used`
- `ball_speed_mph`
- `confidence`
- `frame_count`
- `avg_snr_db`
- `impact_frames`

## Calibration Notes

The geometry selector depends on physical setup measurements. The most important
ones are:

- impact timestamp
- radar-to-ball distance
- radar mount tilt
- radar center height relative to ball height
- boresight/bearing offset

The boresight offset should be calibrated with a known reflector position when
possible. For a reflector at known height and distance:

```text
true_bearing = atan2(reflector_height - radar_height, distance) - mount_deg
offset_needed = true_bearing - measured_raw_angle
```

If the offset is stable across several known reflector positions, it is a real
boresight/electrical-zero correction. If it changes substantially by height,
floor position, or distance, the setup may be affected by multipath or alignment
error.

## Current Limitations

- Very early frames can contain club or multipath contamination even when bin and
  SNR look plausible.
- A one-frame result is underconstrained and must remain low confidence.
- Incorrect impact timing can move the solved launch angle materially.
- Incorrect radar distance, mount tilt, or height assumptions bias the geometry.
- Ground multipath can bias the measured bearing before the selector sees it.

## Follow-Up TODOs

- Revisit single-frame anchor choice when no clean rising pair exists. The
  current anchor ranking prefers smallest OPS-bin error before SNR/coherence.
  TrackMan review of the 2026-05-30 7-iron session showed cases where that can
  select a later, lower-SNR frame over an earlier strong in-flight frame with
  comparable bin error. Candidate rule: when no adjacent rising pair survives,
  prefer an earlier strong frame if it is before the screen-time boundary, has a
  comparable bin error, and has materially better SNR/coherence.
- Tighten early-assisted pair acceptance. Shot review showed cases where an
  early frame alone matched TrackMan, but pairing it with the next frame pushed
  launch high even though RMSE stayed below the high-RMSE fallback threshold.
  Candidate rule: if the paired result diverges sharply from the strongest
  single-frame candidate or requires a large bearing jump, fall back to the
  strongest/earliest single frame and mark it as `geometry_single_frame`.
