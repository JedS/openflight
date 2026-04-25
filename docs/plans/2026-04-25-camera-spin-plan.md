# Camera-Based Spin Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a global-shutter camera as a second spin sensor. Capture a 150 ms burst at ~400 fps after each impact, track marker dots on the ball, fit a 3D rotation, and emit spin rate + axis. Fuse with the existing Doppler spin (camera replaces Doppler when confident; otherwise Doppler stays).

**Architecture:** Trigger off the existing `SoundTrigger`. Capture and solve run on background threads — they never block the initial shot emission. When the solve completes (~1 s post-impact), the server emits a `shot_updated` event and the UI updates the spin display in place.

**Tech Stack:** Python (picamera2, OpenCV, numpy), React/TypeScript, Flask-SocketIO

**Spec:** `docs/plans/2026-04-25-camera-spin-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/openflight/camera_spin/__init__.py` | Create | Package exports |
| `src/openflight/camera_spin/types.py` | Create | `SpinResult`, `CameraConfig` dataclasses |
| `src/openflight/camera_spin/capture.py` | Create | picamera2 burst capture into ring buffer |
| `src/openflight/camera_spin/detect.py` | Create | Ball + marker detection (Hough, blob, threshold) |
| `src/openflight/camera_spin/track.py` | Create | Cross-frame marker tracking (KLT / template match) |
| `src/openflight/camera_spin/solve.py` | Create | 3D rotation fit → `SpinResult` |
| `src/openflight/camera_spin/calibration.py` | Create | Load/save intrinsics + extrinsics |
| `src/openflight/camera_spin/service.py` | Create | `CameraSpinService`: trigger handling, threading, fusion API |
| `scripts/camera-spin/calibrate_intrinsics.py` | Create | Standalone chessboard calibration tool |
| `scripts/camera-spin/calibrate_extrinsics.py` | Create | Standalone PnP pose-estimation tool |
| `scripts/camera-spin/bench_capture.py` | Create | Phase 1 bench tool (capture + offline solve, no integration) |
| `src/openflight/launch_monitor.py` | Modify | Add `Shot.spin_source`, `Shot.spin_axis_confidence` |
| `src/openflight/server.py` | Modify | CLI flags, `init_camera_spin()`, fusion in `on_shot_detected()`, `shot_updated` emit |
| `src/openflight/session_logger.py` | Modify | Log `camera_spin` and optional `camera_frames` entries |
| `scripts/start-kiosk.sh` | Modify | Pass `--camera-spin` flags |
| `ui/src/types/shot.ts` | Modify | Add `spin_source`, `spin_axis_deg` fields |
| `ui/src/components/ShotDisplay.tsx` | Modify | Spin card: add axis line, source badge, pending state |
| `ui/src/hooks/useShotSocket.ts` | Modify | Handle `shot_updated` event |
| `tests/test_camera_spin.py` | Create | Unit tests for types, detect, track, solve, service |
| `tests/test_server.py` | Modify | Test `shot_to_dict` with camera spin fields and source |
| `pyproject.toml` | Modify | Add `picamera2`, `opencv-python` as optional `[camera-spin]` extras |

---

### Task 1: SpinResult and CameraConfig Types

**Files:**
- Create: `src/openflight/camera_spin/__init__.py`
- Create: `src/openflight/camera_spin/types.py`
- Test: `tests/test_camera_spin.py`

- [ ] **Step 1: Create the camera_spin package**

```bash
mkdir -p src/openflight/camera_spin
mkdir -p scripts/camera-spin
```

- [ ] **Step 2: Write types.py**

Create `src/openflight/camera_spin/types.py` with `SpinResult` and `CameraConfig` dataclasses per the design doc. Use string literal type for `confidence` (`Literal["high", "medium", "low"]`).

- [ ] **Step 3: Write __init__.py**

Create `src/openflight/camera_spin/__init__.py` exporting `SpinResult`, `CameraConfig`.

- [ ] **Step 4: Write basic type tests**

Create `tests/test_camera_spin.py` with `TestCameraSpinTypes` class covering field defaults, valid confidence literals, and serialization round-trip.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_camera_spin.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/openflight/camera_spin/ tests/test_camera_spin.py
git commit -m "feat(camera_spin): add SpinResult and CameraConfig types"
```

---

### Task 2: Calibration Load/Save

**Files:**
- Create: `src/openflight/camera_spin/calibration.py`
- Create: `scripts/camera-spin/calibrate_intrinsics.py`
- Create: `scripts/camera-spin/calibrate_extrinsics.py`
- Test: `tests/test_camera_spin.py`

- [ ] **Step 1: Write failing tests for calibration I/O**

Add `TestCalibration` to `tests/test_camera_spin.py` covering:
- `load_intrinsics(path) -> (camera_matrix, dist_coeffs)` round-trip
- `load_extrinsics(path) -> (R_cam_world, t_cam_world)` round-trip
- Missing file → raises `FileNotFoundError` with helpful message

- [ ] **Step 2: Implement calibration.py**

Create `src/openflight/camera_spin/calibration.py` with `load_intrinsics`, `save_intrinsics`, `load_extrinsics`, `save_extrinsics`. Use `numpy.savez_compressed` / `numpy.load` for `.npz` files.

- [ ] **Step 3: Implement standalone calibration scripts**

`scripts/camera-spin/calibrate_intrinsics.py`: chessboard capture loop using OpenCV (works headless or with preview), runs `cv2.calibrateCamera`, writes to `~/.openflight/camera_intrinsics.npz`.

`scripts/camera-spin/calibrate_extrinsics.py`: capture single frame of a marker pattern at the tee position, runs `cv2.solvePnP`, writes to `~/.openflight/camera_extrinsics.npz`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_camera_spin.py::TestCalibration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openflight/camera_spin/calibration.py scripts/camera-spin/ tests/test_camera_spin.py
git commit -m "feat(camera_spin): add calibration I/O and standalone calibration scripts"
```

---

### Task 3: Ball and Marker Detection

**Files:**
- Create: `src/openflight/camera_spin/detect.py`
- Test: `tests/test_camera_spin.py`
- Add: `tests/data/camera_spin/synthetic_ball.png`, `synthetic_ball_with_dots.png`

- [ ] **Step 1: Generate synthetic test images**

Add a fixture or helper that generates a synthetic ball image (white circle on black) and a version with 3 dark dots at known positions on the sphere. Save to `tests/data/camera_spin/`.

- [ ] **Step 2: Write failing tests for detection**

Add `TestBallDetection` and `TestMarkerDetection` to `tests/test_camera_spin.py`:
- `find_ball` returns `(cx, cy, r)` close to ground truth on synthetic image
- `find_ball` returns `None` on a frame with no ball
- `find_markers` returns 3 positions close to ground truth dots
- `find_markers` filters out detections outside the ball ROI

- [ ] **Step 3: Implement detect.py**

```python
def find_ball(frame: np.ndarray) -> Optional[tuple[int, int, int]]:
    """Hough circle on grayscale frame. Returns (cx, cy, r) or None."""

def find_markers(frame: np.ndarray, ball_roi: tuple[int, int, int]) -> list[tuple[float, float]]:
    """Threshold + blob detection inside ball_roi. Returns list of (x, y)."""
```

Use `cv2.HoughCircles` for ball, `cv2.SimpleBlobDetector` for markers. Tune parameters against synthetic fixtures first.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_camera_spin.py::TestBallDetection tests/test_camera_spin.py::TestMarkerDetection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openflight/camera_spin/detect.py tests/test_camera_spin.py tests/data/camera_spin/
git commit -m "feat(camera_spin): add ball and marker detection"
```

---

### Task 4: Cross-Frame Marker Tracking

**Files:**
- Create: `src/openflight/camera_spin/track.py`
- Test: `tests/test_camera_spin.py`
- Add: `tests/data/camera_spin/synthetic_burst/` (synthetic rotating-ball frame sequence)

- [ ] **Step 1: Generate synthetic rotating-ball burst**

Helper that renders a sequence of frames showing a ball with 3 dots rotating about a known axis at a known rate. ~30 frames covering ~1 full rotation. Used as ground truth for both this task and Task 5.

- [ ] **Step 2: Write failing tests for tracking**

Add `TestMarkerTracking`:
- `track_markers` on synthetic burst returns 3 trajectories with consistent length (= n_frames)
- Each trajectory's positions match ground truth ±1 px
- Returns shorter trajectories if some markers go out of frame mid-burst (graceful degradation)

- [ ] **Step 3: Implement track.py**

```python
def track_markers(
    frames: list[np.ndarray],
    initial_markers: list[tuple[float, float]],
) -> list[list[Optional[tuple[float, float]]]]:
    """KLT optical flow across frames. Returns per-marker trajectory; None for lost frames."""
```

Use `cv2.calcOpticalFlowPyrLK`. Forward-backward error check to drop unreliable tracks.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_camera_spin.py::TestMarkerTracking -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openflight/camera_spin/track.py tests/test_camera_spin.py tests/data/camera_spin/synthetic_burst/
git commit -m "feat(camera_spin): add cross-frame marker tracking via optical flow"
```

---

### Task 5: 3D Rotation Solve

**Files:**
- Create: `src/openflight/camera_spin/solve.py`
- Test: `tests/test_camera_spin.py`

- [ ] **Step 1: Write failing tests for rotation solve**

Add `TestRotationSolve`:
- Synthetic burst with known axis + rate → solve recovers rate ±50 RPM, axis ±2°
- Confidence is `"high"` on the clean synthetic case
- Confidence drops to `"medium"` or `"low"` when injecting noise (jitter marker positions)
- Returns `None` when fewer than 2 markers tracked

- [ ] **Step 2: Implement solve.py**

```python
def solve_spin(
    marker_tracks: list[list[Optional[tuple[float, float]]]],
    ball_radius_m: float,
    intrinsics: tuple[np.ndarray, np.ndarray],
    extrinsics: tuple[np.ndarray, np.ndarray],
    frame_dt_s: float,
) -> Optional[SpinResult]:
    """Fit R(t) per frame, derive spin axis + rate from R(t) trajectory."""
```

Algorithm:
1. Compute initial 3D marker positions on sphere from frame-1 image positions (back-project to sphere surface)
2. For each subsequent frame, fit `R(t)` via least-squares (OpenCV `solvePnP` on the marker correspondences)
3. Convert `R(t)` to axis-angle (Rodrigues); compute `omega(t)` from finite difference
4. Average `omega` over the burst; principal direction = spin axis, magnitude = spin rate
5. Compute confidence from fit residuals + frame count + rate stability

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_camera_spin.py::TestRotationSolve -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/openflight/camera_spin/solve.py tests/test_camera_spin.py
git commit -m "feat(camera_spin): add 3D rotation solve from marker tracks"
```

---

### Task 6: Bench Capture Tool (Phase 1 deliverable)

**Files:**
- Create: `scripts/camera-spin/bench_capture.py`

- [ ] **Step 1: Write bench tool**

Standalone script that:
1. Connects to camera via picamera2
2. Captures a 150 ms burst on keypress (no trigger integration)
3. Saves frames to `bench_capture_<timestamp>.npz`
4. Optionally runs the full detect + track + solve pipeline offline and prints `SpinResult`

This validates the full CV pipeline on real shots before any server integration. Smallest possible step that proves the camera path works end-to-end.

- [ ] **Step 2: Manual validation**

Run on Pi with camera connected:
```bash
uv run python scripts/camera-spin/bench_capture.py --solve
```

Hit 5 shots with marker dots applied. Verify the script reports plausible spin numbers (±200 RPM of expected per-club values).

- [ ] **Step 3: Commit**

```bash
git add scripts/camera-spin/bench_capture.py
git commit -m "feat(camera_spin): add bench capture tool for offline validation"
```

---

### Task 7: Live Capture Service

**Files:**
- Create: `src/openflight/camera_spin/capture.py`
- Create: `src/openflight/camera_spin/service.py`
- Modify: `src/openflight/camera_spin/__init__.py`
- Test: `tests/test_camera_spin.py`

- [ ] **Step 1: Write failing tests for service**

Add `TestCameraSpinService` covering:
- Service can be constructed without a camera (`device=None`) for unit-test mode
- `_on_trigger()` spawns a worker thread per trigger
- `get_spin_for_shot(impact_timestamp)` blocks up to N seconds, returns `SpinResult` or `None`
- Concurrent triggers don't race (lock or queue)

Use mocks for picamera2 — no hardware needed for unit tests.

- [ ] **Step 2: Implement capture.py**

```python
class BurstCapture:
    """picamera2 burst into a numpy ring buffer. ROI cropped at sensor level."""
    def __init__(self, config: CameraConfig): ...
    def start(self) -> None: ...
    def capture_burst(self) -> tuple[np.ndarray, float]:
        """Returns (frames_array, first_frame_monotonic_ns)."""
    def stop(self) -> None: ...
```

- [ ] **Step 3: Implement service.py**

```python
class CameraSpinService:
    def __init__(self, config: CameraConfig, sound_trigger): ...
    def start(self) -> None:
        """Subscribes to sound_trigger callback, starts BurstCapture."""
    def stop(self) -> None: ...
    def get_spin_for_shot(self, impact_timestamp: float, timeout_s: float = 2.0) -> Optional[SpinResult]: ...

    def _on_trigger(self, trigger_timestamp: float) -> None:
        """Triggered by sound. Spawns capture+solve worker."""
```

Worker thread: capture burst → run detect + track + solve → store `SpinResult` in a dict keyed by `capture_timestamp`. `get_spin_for_shot` correlates by timestamp ±100 ms.

- [ ] **Step 4: Update __init__.py exports**

Export `CameraSpinService`, `BurstCapture`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_camera_spin.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/openflight/camera_spin/capture.py src/openflight/camera_spin/service.py src/openflight/camera_spin/__init__.py tests/test_camera_spin.py
git commit -m "feat(camera_spin): add BurstCapture and CameraSpinService"
```

---

### Task 8: Shot Fields and Server Fusion

**Files:**
- Modify: `src/openflight/launch_monitor.py`
- Modify: `src/openflight/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write failing test for spin_source in shot_to_dict**

Add to `tests/test_server.py`:

```python
def test_spin_source_field(self):
    shot = Shot(
        ball_speed_mph=150.0,
        timestamp=datetime.now(),
        spin_rpm=2750,
        spin_axis_deg=1.4,
        spin_source="camera",
    )
    result = shot_to_dict(shot)
    assert result["spin_source"] == "camera"
    assert result["spin_axis_deg"] == 1.4
```

- [ ] **Step 2: Add fields to Shot**

In `src/openflight/launch_monitor.py`, add to `Shot`:

```python
spin_source: Optional[str] = None  # "camera", "doppler", "club_typical", or None
```

(`spin_axis_deg` already exists.)

- [ ] **Step 3: Add CLI flags to server.py**

```python
parser.add_argument("--camera-spin", action="store_true", help="Enable camera spin sensor")
parser.add_argument("--camera-spin-device", default="/dev/video0")
parser.add_argument("--camera-spin-intrinsics", default="~/.openflight/camera_intrinsics.npz")
parser.add_argument("--camera-spin-extrinsics", default="~/.openflight/camera_extrinsics.npz")
```

- [ ] **Step 4: Add init_camera_spin() and global**

```python
camera_spin_service = None

def init_camera_spin(args, sound_trigger) -> bool:
    """Initialize camera spin service. Non-fatal on failure."""
```

Wire into `main()` after sound trigger is set up. Add to `finally` cleanup.

- [ ] **Step 5: Add fusion in on_shot_detected()**

After existing Doppler/K-LD7 attachment, before `socketio.emit("shot")`:

```python
if camera_spin_service and shot.mode != "mock":
    camera_spin = camera_spin_service.get_spin_for_shot(
        shot.impact_timestamp, timeout_s=2.0
    )
    if camera_spin and camera_spin.confidence == "high":
        shot.spin_rpm = camera_spin.spin_rpm
        shot.spin_axis_deg = camera_spin.spin_axis_deg
        shot.spin_confidence = 0.95
        shot.spin_source = "camera"
    elif shot.spin_quality == "high":
        shot.spin_source = "doppler"
```

(For `spin_source = "club_typical"`, that gets set in `ballistics.resolve_launch` via existing logic — no change there.)

- [ ] **Step 6: Update shot_to_dict**

Add `"spin_source": shot.spin_source` and ensure `"spin_axis_deg"` is included.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/openflight/launch_monitor.py src/openflight/server.py tests/test_server.py
git commit -m "feat(camera_spin): server fusion — camera spin replaces Doppler when confident"
```

---

### Task 9: Async shot_updated Event

**Files:**
- Modify: `src/openflight/server.py`
- Modify: `ui/src/hooks/useShotSocket.ts`
- Modify: `ui/src/types/shot.ts`

The fusion in Task 8 uses a blocking `get_spin_for_shot(timeout=2s)`. For lower perceived latency, switch to non-blocking emit + delayed update.

- [ ] **Step 1: Make initial emit non-blocking**

In `on_shot_detected()`, emit `shot` with current data immediately. Spawn a background thread to await `get_spin_for_shot()` and emit `shot_updated` when it lands.

```python
def _await_camera_spin(shot_id: str, impact_timestamp: float):
    camera_spin = camera_spin_service.get_spin_for_shot(impact_timestamp, timeout_s=3.0)
    if camera_spin and camera_spin.confidence == "high":
        socketio.emit("shot_updated", {
            "shot_id": shot_id,
            "spin_rpm": camera_spin.spin_rpm,
            "spin_axis_deg": camera_spin.spin_axis_deg,
            "spin_source": "camera",
        })

if camera_spin_service:
    threading.Thread(target=_await_camera_spin, args=(shot.id, shot.impact_timestamp), daemon=True).start()
```

- [ ] **Step 2: Add shot_id to Shot**

Generate a UUID per shot at creation time. Include in `shot_to_dict`.

- [ ] **Step 3: Handle shot_updated in UI**

In `ui/src/hooks/useShotSocket.ts`, register a `shot_updated` listener that merges updated fields into the matching shot in state by `shot_id`.

- [ ] **Step 4: Update Shot type**

In `ui/src/types/shot.ts`, add `shot_id`, `spin_source`, `spin_axis_deg` if not already present.

- [ ] **Step 5: Commit**

```bash
git add src/openflight/server.py ui/src/hooks/useShotSocket.ts ui/src/types/shot.ts
git commit -m "feat(camera_spin): async shot_updated event for delayed camera spin"
```

---

### Task 10: UI — Spin Card Updates

**Files:**
- Modify: `ui/src/components/ShotDisplay.tsx`

- [ ] **Step 1: Add spin axis line to spin card**

Show `spin_axis_deg` below `spin_rpm` with shape label:

```tsx
{shot.spin_axis_deg !== null && (
  <div className="spin-axis">
    {shot.spin_axis_deg > 0 ? '+' : ''}{shot.spin_axis_deg.toFixed(1)}°
    {' '}
    <span className="shape-label">
      {Math.abs(shot.spin_axis_deg) < 1 ? 'straight'
        : shot.spin_axis_deg > 0 ? 'fade' : 'draw'}
    </span>
  </div>
)}
```

- [ ] **Step 2: Add spin source badge**

```tsx
<span className="spin-source-badge">
  {shot.spin_source === 'camera' ? '📷' :
   shot.spin_source === 'doppler' ? '📡' :
   shot.spin_source === 'club_typical' ? '≈' : ''}
  {' '}{shot.spin_source ?? '—'}
</span>
```

- [ ] **Step 3: Show pending state for ~1s when camera enabled**

If camera spin is enabled and shot just landed without `spin_source === "camera"`, show a "..." or skeleton on the spin numbers. Replace when `shot_updated` arrives.

- [ ] **Step 4: Build UI**

Run: `cd ui && npm run build`
Expected: no TypeScript errors

- [ ] **Step 5: Commit**

```bash
git add ui/src/components/ShotDisplay.tsx
git commit -m "feat(ui): spin card shows axis, source badge, and pending state"
```

---

### Task 11: Session Logging

**Files:**
- Modify: `src/openflight/session_logger.py`

- [ ] **Step 1: Add camera_spin entry**

Add a `log_camera_spin(spin_result: SpinResult, impact_timestamp: float)` method that writes a JSONL entry per the design doc schema.

- [ ] **Step 2: Call from on_shot_detected (or _await_camera_spin)**

When a `SpinResult` lands, log it regardless of confidence (we want the medium/low entries for analysis).

- [ ] **Step 3: Add optional camera_frames entry**

Behind a `--camera-spin-log-frames` CLI flag (off by default). Saves the raw burst to `session_<id>/shot_<n>_frames.npz` and writes a pointer entry.

- [ ] **Step 4: Commit**

```bash
git add src/openflight/session_logger.py src/openflight/server.py
git commit -m "feat(camera_spin): log camera_spin and optional camera_frames to session JSONL"
```

---

### Task 12: pyproject.toml Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional [camera-spin] extra**

```toml
[project.optional-dependencies]
camera-spin = [
    "picamera2>=0.3.0",
    "opencv-python>=4.8",
]
```

- [ ] **Step 2: Add to the standard Pi install group**

If there's a `[pi]` or `[ui]` extra used for the kiosk install, include `camera-spin` so it ships by default on Pi.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(camera_spin): add picamera2 and opencv as optional camera-spin extras"
```

---

### Task 13: start-kiosk.sh Flag Pass-Through

**Files:**
- Modify: `scripts/start-kiosk.sh`

- [ ] **Step 1: Add CAMERA_SPIN variables and arg parsing**

Match the pattern used for `--kld7`. Variables:

```bash
CAMERA_SPIN=false
CAMERA_SPIN_DEVICE=""
CAMERA_SPIN_INTRINSICS=""
CAMERA_SPIN_EXTRINSICS=""
```

In the `while` arg loop:

```bash
        --camera-spin)
            CAMERA_SPIN=true
            shift
            ;;
        --camera-spin-device)
            CAMERA_SPIN_DEVICE="$2"
            shift 2
            ;;
        # ... and so on for intrinsics + extrinsics
```

In the command-building section:

```bash
if [ "$CAMERA_SPIN" = true ]; then
    SERVER_CMD="$SERVER_CMD --camera-spin"
fi
# ... etc
```

- [ ] **Step 2: Commit**

```bash
git add scripts/start-kiosk.sh
git commit -m "feat(camera_spin): add --camera-spin flags to start-kiosk.sh"
```

---

### Task 14: Validation Test Fixtures

**Files:**
- Add: `tests/data/camera_spin/reference_shots/` (5 known-spin bursts)
- Add: `tests/data/camera_spin/reference_shots/ground_truth.json`
- Test: `tests/test_camera_spin.py`

- [ ] **Step 1: Capture 5 reference shots**

Use `bench_capture.py` to record 5 shots with simultaneously-recorded ground truth from a commercial monitor (TrackMan, GCQuad, or similar). Save bursts as `.npz` files plus a `ground_truth.json` mapping each shot to its expected `spin_rpm` and `spin_axis_deg`.

- [ ] **Step 2: Write validation test**

Add `TestCameraSpinValidation` that runs the full pipeline on each reference shot and asserts:
- `spin_rpm` within ±100 RPM of ground truth
- `spin_axis_deg` within ±5° of ground truth
- Confidence is `"high"` for at least 4 of 5 shots

This test is marked `@pytest.mark.slow` and excluded from the default test run; run explicitly in CI nightly.

- [ ] **Step 3: Commit**

```bash
git add tests/data/camera_spin/reference_shots/ tests/test_camera_spin.py
git commit -m "test(camera_spin): add validation against 5 reference shots with ground truth"
```

---

### Task 15: Documentation Updates

**Files:**
- Modify: `docs/raspberry-pi-setup.md`
- Modify: `docs/PARTS.md`
- Modify: `CLAUDE.md`
- Create: `docs/camera-spin-calibration.md`

- [ ] **Step 1: Add camera section to raspberry-pi-setup.md**

Mirror the K-LD7 section structure:
- Wiring (CSI cable to Pi 5 CAM port)
- Calibration steps (intrinsics once, extrinsics per install)
- Marker dot application instructions
- Outdoor mounting tips

- [ ] **Step 2: Add camera to PARTS.md**

```markdown
## Camera (Spin Measurement)

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| Pi Global Shutter Camera | Sony IMX296, 1456×1088 mono | [Adafruit](...) | $50 |
| 6mm CS-mount lens | f/1.4 or wider | [Amazon](...) | $15 |
```

- [ ] **Step 3: Update CLAUDE.md architecture diagram**

Add `CameraSpinService` to the data flow ASCII art and the Key Modules section.

- [ ] **Step 4: Create camera-spin-calibration.md**

Detailed walkthrough of intrinsics + extrinsics calibration with photos / sample output.

- [ ] **Step 5: Commit**

```bash
git add docs/ CLAUDE.md
git commit -m "docs(camera_spin): add camera setup, parts, and calibration guides"
```

---

## Phasing

These tasks group naturally into the four phases from the design doc:

- **Phase 1 — Bench prove:** Tasks 1–6. CV pipeline works on real shots, no server integration.
- **Phase 2 — Live capture, log only:** Tasks 7, 11, 12, 13. Service runs end-to-end, logs alongside Doppler. No UI changes; no spin replacement.
- **Phase 3 — Fusion:** Tasks 8, 9, 10. Camera spin replaces Doppler when confident; UI shows source badge and pending state.
- **Phase 4 — Polish:** Tasks 14, 15. Validation tests in CI; user-facing docs.

Each phase is independently shippable. Stop after Phase 2 if accuracy doesn't justify the UI complexity.

---

## Verification

```bash
# All camera_spin unit tests
uv run pytest tests/test_camera_spin.py -v

# Server tests (shot_to_dict, fusion)
uv run pytest tests/test_server.py -v

# Validation against reference shots (slow, runs nightly)
uv run pytest tests/test_camera_spin.py::TestCameraSpinValidation -v -m slow

# UI builds cleanly
cd ui && npm run build

# Manual test on Pi with camera connected:
scripts/start-kiosk.sh --camera-spin

# Manual test without camera (should work exactly as before):
scripts/start-kiosk.sh
```
