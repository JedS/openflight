# Trackman Test Process

This note is a handoff for future agents working on OpenFlight accuracy against
Trackman. Treat Trackman as the source of truth for metric comparison.

## Goal

Use Trackman sessions to improve OpenFlight accuracy without hiding failures.
For every test pass, preserve enough raw data and diagnostics to answer:

- Did OpenFlight detect the same shot Trackman saw?
- Are ball speed, club speed, launch angles, spin, and carry biased?
- If spin or K-LD7 angles are missing, did the system log why?
- Are rejected values truly bad signal, or are guardrails too strict?

## Before A Session

1. Pull latest `main` on the Pi and restart OpenFlight.
2. Confirm the selected club in the UI matches the club being hit.
   - This matters for launch-angle fallbacks, spin expectations, and club-aware
     spin rail filtering.
3. Confirm OPS243 rolling buffer mode is active and sound trigger re-arms after
   each shot.
4. Confirm K-LD7 orientation and udev symlinks:
   - horizontal: `/dev/kld7_horizontal`
   - vertical: `/dev/kld7_vertical`
5. If testing raw K-LD7 ADC, run the RADC capture script and save a `.pkl`.
6. Make sure Trackman export includes shot number/order, club, ball speed, club
   speed, launch angle, launch direction, spin rate, and carry.

## Files To Collect

Always collect:

- OpenFlight JSONL session log: `session_logs/session_<timestamp>_range.jsonl`
- Trackman normalized CSV export
- Any generated comparison CSV/plots

Collect when debugging K-LD7 angles:

- K-LD7 raw ADC `.pkl` from `scripts/analysis/capture_kld7_radc.py`
- Any `diagnose_kld7_raw_adc.py` output directories

Common local paths:

- Current workspace: `/Users/colemanrollins/conductor/workspaces/openflight/<workspace>`
- Main parent checkout: `/Users/colemanrollins/code/openflight`

The Pi or parent checkout may contain session logs that are not present in the
Conductor workspace. Check both before assuming a file is missing.

## Raw K-LD7 ADC Capture

Use this when investigating horizontal or vertical launch-angle misses:

```bash
uv run --no-project \
  --with pyserial --with numpy --with scipy \
  python scripts/analysis/capture_kld7_radc.py \
  --orientation horizontal \
  --duration 90
```

Notes:

- Default capture is RADC-only. Add `--include-targets` only when PDAT/TDAT are
  needed; they reduce available serial bandwidth.
- The script should leave OPS243 rolling buffer armed after each trigger and on
  shutdown.
- Store the `.pkl` next to session logs or copy it into a shared location.
- If a capture has many short/invalid RADC payloads, note the K-LD7 frame rate
  and USB/serial contention before tuning angle logic.

Analyze a `.pkl`:

```bash
uv run --no-project \
  --with numpy --with scipy \
  python scripts/analysis/diagnose_kld7_raw_adc.py \
  session_logs/kld7_radc_<timestamp>.pkl \
  --output .context/raw_adc_diag_<timestamp>
```

Useful outputs:

- `radc_summary.json`
- `radc_frame_diagnostics.csv`
- `shot_summaries.csv`
- per-shot `shot_##_frame_diagnostics.csv`

## Trackman Comparison

Generate an OpenFlight vs Trackman comparison:

```bash
PYTHONPATH=scripts/analysis uv run --no-project \
  --with numpy \
  python scripts/analysis/compare_trackman.py \
  --openflight session_logs/session_<timestamp>_range.jsonl \
  --trackman session_logs/<trackman_export>.csv \
  --output session_logs/comparison_<timestamp>.csv
```

The comparison script reports per-club bias and writes row-level deltas for:

- ball speed
- club speed
- smash
- vertical launch
- horizontal launch
- spin
- carry

It also includes OpenFlight spin diagnostics when present:

- `spin_candidate_of`
- `spin_confidence_of`
- `spin_quality_of`
- `spin_snr_of`
- `spin_rejection_of`

Use these columns to determine whether OpenFlight had no spin signal, rejected a
candidate, or accepted a low-confidence value.

## Reading Session Logs

Important JSONL rows:

- `shot_detected`: user-facing shot values and per-shot diagnostics
- `rolling_buffer_capture`: raw OPS243 I/Q plus detailed speed/spin processing
- `kld7_buffer`: buffered K-LD7 frames and selected angle diagnostics
- `trigger_diagnostic`: trigger acceptance, latency, and speed timeline details

Spin diagnostics to inspect:

- `spin_rpm`: accepted user-facing spin, or `null`
- `spin_candidate_rpm`: candidate RPM even when rejected
- `spin_snr`: envelope peak SNR
- `spin_quality`: processor quality label for accepted spin
- `spin_rejection_reason`: why spin was withheld
- `spin_at_lower_rail` / `spin_at_upper_rail`: boundary artifacts

Launch-angle diagnostics to inspect:

- `launch_angle_vertical`
- `launch_angle_horizontal`
- `launch_angle_confidence`
- `angle_source`
- `club_angle_deg`
- `club_path_deg`
- `spin_axis_deg`

## Interpreting Spin

Current spin handling is intentionally conservative:

- Upper-rail candidates near 12000 RPM are usually rejected as filter-edge noise.
- Lower-rail candidates around 3300-3500 RPM are capped to low confidence.
- For high-spin clubs such as 7-iron, PW, GW, SW, and LW, implausibly low
  lower-rail candidates are withheld and logged with a plausibility reason.
- Rejected spin should still log candidate RPM, SNR, peak frequency, seam cycles,
  rail flags, and rejection reason.

Do not loosen spin guardrails just to increase read rate. First confirm from
Trackman and raw I/Q whether the accepted candidates would be accurate. A useful
spin improvement should increase matched, accurate readings without reintroducing
rail artifacts.

## Interpreting K-LD7 Angles

Current angle handling:

- Live shots should always emit some vertical and horizontal launch angle.
- Radar/camera measurements win when plausible.
- Vertical fallback uses club/speed/smash/spin estimates.
- Horizontal fallback is neutral `0.0`.
- K-LD7 RADC extraction is filtered by shot timestamp so stale frames do not
  dominate the result.
- Weak wall/edge candidates are retried with low-energy settings rather than
  blindly reported.

When debugging K-LD7 misses, prefer shot-window RADC analysis over whole-buffer
analysis. Whole-buffer replays can select stale frames that live processing now
ignores.

## After A Session

1. Copy OpenFlight JSONL, Trackman CSV, and any `.pkl` into `session_logs/`.
2. Run `compare_trackman.py`.
3. For K-LD7 angle misses, run `diagnose_kld7_raw_adc.py`.
4. Summarize per-club bias and detection rate:
   - ball speed bias/stddev
   - vertical launch bias/RMSE
   - horizontal launch bias/RMSE
   - spin read rate and spin delta where accepted
   - rejected spin reasons by count
   - K-LD7 valid/invalid RADC frame counts
5. Only tune live processing after separating:
   - pairing errors
   - hardware/throughput problems
   - stale-buffer artifacts
   - real DSP/gating problems

## Commands For Validation

Run focused checks after changing launch angle, spin, logging, or comparison
scripts:

```bash
ruff check \
  src/openflight/rolling_buffer \
  src/openflight/server.py \
  src/openflight/session_logger.py \
  scripts/analysis \
  tests/test_rolling_buffer.py \
  tests/test_server.py \
  tests/test_session_logger.py \
  tests/test_compare_trackman.py

PYTHONPATH=src:scripts/analysis uv run --no-project \
  --with pytest --with pyserial --with flask --with flask-socketio \
  --with flask-cors --with numpy --with scipy \
  python -m pytest \
  tests/test_rolling_buffer.py \
  tests/test_session_logger.py \
  tests/test_server.py \
  tests/test_compare_trackman.py
```

For the full suite, include optional analysis dependencies:

```bash
PYTHONPATH=src:scripts/analysis uv run --no-project \
  --with pytest --with pyserial --with flask --with flask-socketio \
  --with flask-cors --with numpy --with scipy --with matplotlib \
  python -m pytest tests
```

## Related Docs

- `docs/rolling_buffer_spin_detection.md`
- `docs/kld7-session-review.md`
- `docs/kld7-troubleshooting.md`
- `docs/kld7-ball-detection-theory.md`
- `docs/observability.md`
