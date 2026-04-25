# Camera-Based Spin Measurement — Design

**Date:** 2026-04-25
**Status:** Draft

## Goal

Add a global-shutter camera as a second spin sensor to OpenFlight, providing both spin **rate** and **spin axis** with materially better accuracy than Doppler-only extraction from the OPS243-A. Designed for outdoor use: variable lighting, sun glare, range-of-conditions.

The OPS243-A stays the source of truth for ball/club speed. The K-LD7 pair stays the source for launch angles. The camera adds **spin axis** (which Doppler cannot resolve from a single channel) and improves spin rate accuracy. When camera confidence is high, camera spin replaces Doppler spin in the Shot; otherwise Doppler stays as fallback.

## Why a camera

Single-channel Doppler extracts spin rate from seam-modulation sidebands but cannot resolve the spin axis — the axis requires spatial diversity (multi-antenna phased array, two cameras, or one camera with markers). Outdoor conditions favor camera over multi-static radar:

- Direct sunlight gives ideal contrast for marker tracking.
- 24 GHz radar arrays are DIY-hostile and expensive.
- Trajectory back-fit (observe ball curve, infer spin) is wind-corrupted outdoors.

Reference accuracy targets:

- **Spin rate:** ±50–100 RPM
- **Spin axis:** ±2–3°
- **Latency:** result available within ~1.5 s of impact (post-shot, not blocking the initial shot emission)

## Architecture

### CameraSpin Module (`src/openflight/camera_spin/`)

**`service.py`** — Main integration class, mirrors `KLD7Tracker` pattern:

```python
CameraSpinService(
    device="/dev/video0",        # Picamera2 device or path
    intrinsics_path="...",       # Pre-calibrated camera matrix + distortion
    extrinsics_path="...",       # Camera-to-ball pose
    burst_ms=150,                # Capture window post-impact
    target_fps=400,              # Achieved via ROI crop
    roi=(320, 240),              # Centered on expected ball-at-rest
)
```

- Subscribes to the existing `SoundTrigger` callback (no new wiring)
- On trigger: spawns a capture thread, then a solve thread; non-blocking from the shot pipeline
- `get_spin_for_shot(impact_timestamp) -> SpinResult | None`:
  - Correlates by `impact_timestamp` against the capture timestamp
  - Returns rpm, axis_deg, confidence, supporting metadata
  - Returns None if capture failed, ball not detected, or solve timed out

**`capture.py`** — Camera I/O:

- Uses `picamera2` for IMX296 access via CSI on Pi 5
- Burst capture into a numpy ring buffer in shared memory (no disk writes on hot path)
- ROI crop applied at sensor level (not post-capture) to hit 400+ fps
- Auto-exposure on the ball ROI; fast shutter (~200 µs) to freeze ball motion

**`detect.py`** — Per-frame detection:

- `find_ball(frame) -> (cx, cy, r) | None`: Hough circle on first frame to localize ball
- `find_markers(frame, ball_roi) -> list[(x, y)]`: threshold + blob detection on dark dots
- `track_markers(frames, initial_markers) -> list[list[(x, y)]]`: KLT optical flow or template match across the burst

**`solve.py`** — Rotation fit:

- Inputs: marker tracks (2D image positions per frame), camera intrinsics, ball pose
- Project 3D marker positions on the ball sphere → fit `R(t)` (rotation matrix per frame) via least squares
- Extract spin axis (Rodrigues vector direction) and angular velocity (rad/s → RPM) from `R(t)` trajectory
- Returns `SpinResult` with confidence based on fit residuals + frame count + marker count

**`types.py`** — Data types:

```python
@dataclass
class SpinResult:
    spin_rpm: float
    spin_axis_deg: float          # Same convention as Shot.spin_axis_deg
    confidence: str               # "high" | "medium" | "low"
    n_frames: int                 # Frames with successful tracking
    n_markers: int                # Markers tracked across all frames
    fit_residual_px: float        # Median reprojection error
    capture_timestamp: float      # First-frame monotonic_ns
    solve_ms: float               # Wall-clock solve time

@dataclass
class CameraConfig:
    device: str
    burst_ms: int
    target_fps: int
    roi: tuple[int, int]
    intrinsics_path: str
    extrinsics_path: str
```

### Server Integration

**CLI flags** (server.py + start-kiosk.sh):

| Flag | Default | Description |
|------|---------|-------------|
| `--camera-spin` | off | Enable camera spin sensor |
| `--camera-spin-device` | `/dev/video0` | Picamera2 device |
| `--camera-spin-intrinsics` | `~/.openflight/camera_intrinsics.npz` | Calibration file |
| `--camera-spin-extrinsics` | `~/.openflight/camera_extrinsics.npz` | Pose file |

**Startup:**

1. If `--camera-spin` present, create `CameraSpinService`, load calibration, start
2. Non-fatal on failure — warn and continue with Doppler-only spin (like K-LD7)

**Shot fusion** (in `on_shot_detected()`, after Doppler/K-LD7 attachment):

```python
if camera_spin_service:
    camera_spin = camera_spin_service.get_spin_for_shot(shot.impact_timestamp)
    if camera_spin and camera_spin.confidence == "high":
        shot.spin_rpm = camera_spin.spin_rpm
        shot.spin_axis_deg = camera_spin.spin_axis_deg
        shot.spin_source = "camera"
        shot.spin_confidence = 0.95   # camera "high" maps above the Doppler "high" threshold
```

**Spin source priority:**

1. Camera (high confidence) — replaces Doppler
2. Doppler (high confidence) — current behavior
3. Club-typical fallback — `ballistics.resolve_launch` substitutes per-club spin

Camera trumps Doppler when confident. Doppler stays as fallback. `ballistics.resolve_launch` is unchanged — it consumes `Shot.spin_rpm` + `Shot.spin_axis_deg` regardless of source.

**Async result delivery:**

The camera result lands ~1 s after Doppler. The server emits two events:

1. `shot` — fires immediately with Doppler/K-LD7 data
2. `shot_updated` — fires when camera spin lands, UI updates spin number + axis in place

### UI Changes

**Current:** Spin card shows `spin_rpm` + quality badge.

**Changes:**

- Spin card adds `spin_axis_deg` line (e.g. "+1.4° fade")
- Add `spin_source` badge: "📷 camera" / "📡 doppler" / "≈ club typical"
- Spin card shows brief loading state (~1 s) when camera result is pending
- Animate the spin numbers when `shot_updated` arrives

## Data Flow

```
SEN-14262 (sound trigger)
    │
    ├─► OPS243 HOST_INT ──► RollingBufferMonitor
    │                            │
    │                            ▼
    │                       Shot (Doppler spin, fallback)
    │                            │
    ├─► KLD7Tracker ─────────────┤ (existing)
    │                            │
    └─► CameraSpinService        │
            │                    │
            ▼                    │
        burst capture            │
        (150 ms, ~75 frames)     │
            │                    │
            ▼                    │
        CV solve                 │
        (~700 ms)                │
            │                    │
            ▼                    │
        SpinResult ──────────────┤
                                 │
                                 ▼
                         on_shot_detected()
                                 │
                                 ├── fuse spin (camera > Doppler > club_typical)
                                 ├── ballistics.resolve_launch + simulate
                                 ├── session_logger.log_shot(...)
                                 └── socketio.emit("shot")
                                          │
                                          ▼ (~1 s later)
                                 socketio.emit("shot_updated", camera spin)
                                          │
                                          ▼
                                      React UI
```

## Hardware

### Camera

| Part | Spec | Notes |
|------|------|-------|
| Pi Global Shutter Camera | Sony IMX296, 1456×1088 mono | ~$50, CSI direct to Pi 5 |
| Lens | 6 mm CS-mount, f/1.4 or wider | Wider FOV captures ball longer |
| Lens hood + UV filter | — | Cuts sun glare, protects optics |
| IR illuminator (optional) | 850 nm, 5–10 W | For dawn/dusk/shaded conditions |
| IR pass filter (optional) | 850 nm pass | Required if using IR illumination |

**Why global shutter:** rolling shutter smears a 4000-RPM ball into rotational artifacts that destroy the rotation fit. This is non-negotiable.

**Why mono:** higher sensitivity per pixel; spin tracking doesn't need color.

### Mount

- Behind the ball, 0.5–1.0 m back, slightly elevated (~0.3 m above ball)
- Aimed downrange, ball centered in frame at the tee position
- Shares enclosure with the OPS243-A — same trigger plane, easier alignment
- Rigid mount is critical for calibration stability (any flex invalidates extrinsics)

### Trigger

**Phase 1: software trigger.** Existing `SoundTrigger` Python callback fires the camera burst.

- Latency: ~1–2 ms (Python + USB/CSI overhead)
- At 400 fps that's <1 frame of jitter — acceptable
- No new wiring

**Phase 4 (if needed): hardware trigger.** Tap SEN-14262 `GATE` into the GS Camera external trigger (XTR pin) via a 3.3 V level shifter.

- Latency: ~µs
- Only needed if software latency consistently misses the first useful frame in testing

## Capture Pipeline

### Sensor configuration

- **ROI crop:** 320×240 centered on expected ball-at-rest position
  - Full sensor: 1456×1088 → ~120 fps max
  - Cropped to 320×240 → **400–500 fps** sustained
- **Shutter:** 200 µs (motion blur < 1 px at expected ball ranges)
- **Exposure:** auto-exposure on the ball ROI only — outdoor lighting changes shot-to-shot
- **Pixel format:** 8-bit mono

### Burst window

- Duration: ~150 ms post-impact
- Frame count: 60–75 frames at 400 fps
- Buffer: numpy ring buffer in shared memory; never written to disk on the hot path

### Spin observation

At 1500 RPM (typical iron) the ball completes 1 rotation in 40 ms — the 150 ms window captures ~3 full rotations. At 8000 RPM (wedge) it's ~20 rotations. Either is sufficient for a robust rotation fit.

## CV Pipeline

```
frames ──► ball detect (frame 1, Hough circle)
            │
            ▼
        ROI mask + intensity threshold
            │
            ▼
        marker detect (3+ dots applied to ball)
            │
            ▼
        track marker positions across frames (KLT optical flow / template match)
            │
            ▼
        solve 3D rotation: marker_3d_t = R(t) · marker_3d_0
            │   (Rodrigues axis-angle, least squares fit)
            ▼
        SpinResult { rpm, axis_deg, fit_residual, n_frames, confidence }
```

### Marker approach (Phase 1)

Apply 3+ small black dots to the ball pre-shot. Marker positions in 3D ball-frame coordinates are known once the ball is detected (radius known, dot positions parameterized on the sphere surface).

For each subsequent frame:

```
project(R(t) · marker_3d_0)  →  observed 2D marker positions in image
```

Least-squares fit `R(t)` using camera intrinsics. The trajectory of `R(t)` over the burst gives:

- **Spin axis:** principal rotation axis (Rodrigues vector direction)
- **Spin rate:** angular velocity magnitude (rad/s → RPM)

Integration over the ~150 ms window averages out frame-level noise.

### Dimple-pattern correlation (future)

No marker prep, but fragile in low contrast and requires a calibrated dimple template. Out of scope for Phase 1; revisit if marker-prep friction becomes a real complaint.

## Confidence Model

`SpinResult.confidence` is `"high"` / `"medium"` / `"low"` based on:

| Signal | High | Medium | Low |
|--------|------|--------|-----|
| Ball detected in % frames | ≥ 80% | ≥ 50% | < 50% |
| Markers tracked ≥ frames | 3+ × 20+ | 2+ × 10+ | otherwise |
| Fit residual (median px) | < 2.0 | < 4.0 | otherwise |
| Rate stability (CV%) | < 5% | < 15% | otherwise |

All four must hit "high" tier for high confidence. Single-tier drops set the result to medium; two or more drops sets it to low.

Confidence drives fusion: only `"high"` overrides Doppler.

## Calibration

### Camera intrinsics (one-time)

Standard OpenCV chessboard calibration:

```bash
uv run python scripts/camera-spin/calibrate_intrinsics.py
```

Saves `~/.openflight/camera_intrinsics.npz` (camera matrix + distortion coefficients). Run once per camera + lens combination.

### Camera extrinsics (per install)

The camera-to-ball pose must be known to project markers correctly:

1. Place a calibration ball at the tee position
2. Capture a single frame
3. `solvePnP` with a known marker pattern → `R_cam_world`, `t_cam_world`

Saves `~/.openflight/camera_extrinsics.npz`. Re-run if the camera or tee position moves.

### Validation

Ship 5 known-spin reference shots in `tests/data/` (frames + ground-truth spin from a commercial monitor). Solver must reproduce ±100 RPM / ±5° on those shots in CI.

## Synchronization

- Both OPS243 buffer dump and camera burst trigger off the same `SoundTrigger` callback
- Camera worker timestamps the **first frame** with `time.monotonic_ns()`
- Server correlates by `impact_timestamp` — same pattern as K-LD7 today (`server.py:1096–1113`)
- Camera result arrives ~1 s after Doppler; server emits a second `shot_updated` event

## Outdoor Robustness

| Condition | Mitigation |
|-----------|-----------|
| Direct sun glare | Lens hood + UV filter; auto-exposure on ball ROI |
| Dawn / dusk | Optional 850 nm IR illuminator + IR pass filter |
| Overcast | Auto-exposure handles; baseline case |
| Rain / fog | Hydrophobic coating; physical hood; degrades gracefully (low confidence → Doppler fallback) |
| Wind blowing dust | Same as rain — confidence drops, Doppler takes over |
| Dirty lens | UI alert when ball-detection rate drops below threshold; user wipes lens |

The fusion model means **no shot ever fails** because the camera failed — it just falls back to Doppler or club-typical.

## Performance

Pi 5 has 4× Cortex-A76 cores. Capture and solve run on separate threads:

- **Capture thread:** picamera2 burst into ring buffer (~150 ms wall time)
- **Solve thread:** CV pipeline (~500–800 ms wall time)
- **Total post-shot latency:** ~1 s before camera spin available

Doppler shot emission is **not blocked** — it ships immediately with the existing data, then `shot_updated` fires when camera result lands.

Memory: each burst is 75 frames × 320×240 × 1 byte = ~5.7 MB. Held in memory during solve, then released.

## Session Logging

New JSONL entry types:

```json
{
  "type": "camera_spin",
  "timestamp": 1234567890.123,
  "impact_timestamp": 1234567890.100,
  "spin_rpm": 2750,
  "spin_axis_deg": 1.4,
  "confidence": "high",
  "n_frames": 67,
  "n_markers_tracked": 3,
  "fit_residual_px": 0.8,
  "solve_ms": 720
}
```

Optional raw frame dump (off by default, enabled for debugging):

```json
{
  "type": "camera_frames",
  "timestamp": 1234567890.123,
  "frames_path": "session_20260425/shot_042_frames.npz"
}
```

## Known Limitations

1. **Marker prep required (Phase 1).** Apply 3+ dots to the ball before play. Friction; dimple-pattern tracking is a future enhancement.
2. **Calibration sensitive to mount flex.** Any movement of the camera invalidates extrinsics — re-calibrate after physical changes.
3. **Rain / heavy fog degrades silently.** Confidence drops to low → Doppler fallback. UI shows source so user can interpret.
4. **Spin axis only.** Camera does not measure dynamic loft, face angle, or strike location — those need a different camera position (face-on or side view), out of scope.
5. **Post-shot latency ~1 s.** Spin number "lands late" in the UI. Mitigated by pending state + animation, but distinct from immediate fields.

## Module Responsibilities (Clean Separation)

| Measurement | Module | Notes |
|-------------|--------|-------|
| Ball speed | OPS243-A | Primary measurement |
| Club speed | OPS243-A | From pre-impact readings |
| Spin rate | **Camera (high conf)** → OPS243-A → club typical | Camera replaces Doppler when confident |
| Spin axis | **Camera (high conf)** → 0° (assumed pure backspin) | Doppler cannot measure axis |
| Vertical angle | K-LD7 (vertical mount) | Existing |
| Horizontal angle | K-LD7 (horizontal mount) | Existing |
| Trajectory | `ballistics.simulate` | Consumes spin from above |
