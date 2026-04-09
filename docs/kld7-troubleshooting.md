# K-LD7 Troubleshooting Guide

Common issues with the K-LD7 angle radar and how to resolve them.

## Dual Radar Setup

When running two K-LD7s (vertical + horizontal), each needs:
- **Its own FTDI adapter and USB port**
- **A different base frequency** to avoid RF interference (set via `base_freq` parameter: 0=Low, 2=High)
- **A stable device name** via udev rules (see [setup guide](raspberry-pi-setup.md#stable-device-names-udev-rules)) — USB enumeration order can swap `/dev/ttyUSB0` and `/dev/ttyUSB1` after reboot

## Connection Issues

### "Wrong length reply" on startup

**Symptom:** Server fails to start with:
```
[KLD7] Connect attempt 1/5 failed: Wrong length reply
```

**Cause:** The K-LD7 was left at 3Mbaud from a prior session that crashed without sending GBYE. The kld7 library tries to INIT at 115200 baud, but the K-LD7 is still at 3Mbaud and can't parse the packet.

**Automatic recovery:** The server sends a binary GBYE packet at 3Mbaud between retry attempts to reset the K-LD7 to its idle state. This usually succeeds on attempt 2. If it doesn't recover after 5 attempts, the server exits.

**Manual recovery:** Power cycle the K-LD7 (unplug USB, wait 3 seconds, replug).

### "No K-LD7 EVAL board detected"

**Symptom:** Auto-detection can't find the K-LD7 serial port.

**Cause:** The K-LD7 uses an FTDI USB-to-serial adapter which shows up as `/dev/ttyUSB*`. If multiple USB-serial devices are connected, auto-detection may pick the wrong one.

**Fix:** Specify the port explicitly:
```bash
scripts/start-kiosk.sh --kld7 --kld7-port /dev/ttyUSB0
```

To find the correct port:
```bash
ls /dev/ttyUSB*
# or
python3 -m serial.tools.list_ports -v
```

## RADC Streaming Issues

### No RADC frames in buffer

**Symptom:** K-LD7 connects successfully but shots show `angle_source: estimated` instead of `radar`. The K-LD7 buffer log shows `NO ANGLE` or zero RADC frames.

**Check the startup logs for:**
```
[KLD7] Stream started: RADC only (3Mbaud)
[KLD7] First RADC frame received (3072 bytes)    ← must see this
[KLD7] Stream health: 50 RADC frames             ← confirms sustained streaming
```

**If "First RADC frame" never appears:**
- The K-LD7 isn't sending RADC data. Check that the connection is at 3Mbaud (the `Connected at X baud` log line).
- RADC frames are 3072 bytes each at ~18 FPS = ~55 KB/s. This requires 3Mbaud. At 115200 baud, RADC can't be transmitted in real time and the library silently drops frames.

**If "Stream crashed" appears:**
- Check the traceback. Common cause: the K-LD7 FTDI adapter disconnected or had a USB glitch. The stream thread exits and won't restart — you need to restart the server.

### "Stream ended" with running=True

**Symptom:** The `stream_frames` generator exited unexpectedly.

**Cause:** The kld7 library's `stream_frames` can exit if it receives a `DONE` frame code or encounters a packet error. This shouldn't happen during normal operation.

**Fix:** Restart the server. If it persists, check USB cable and connections.

## Angle Accuracy

### Launch angles consistently too high or too low

**Cause:** The `--kld7-angle-offset` parameter needs calibration for your mounting geometry.

**Calibration process:**
1. Hit several full shots with a known club (e.g., 7-iron)
2. Check the session log for raw RADC angles (subtract your current offset from the reported angle)
3. Compare to expected launch angles for that club:
   - Wedge: 24-30°
   - 7-iron: 16-18°
   - 5-iron: 12-14°
   - Driver: 10-14°
4. Set offset = expected angle - raw angle

```bash
# Example: raw angle is 8°, expected 7i launch is 17°, offset = 9
scripts/start-kiosk.sh --kld7 --kld7-angle-offset 9
```

### "RADC extraction returned None"

**Symptom:** K-LD7 has RADC data but can't find the ball.

**Possible causes:**
1. **Ball speed too low** — The RADC extraction uses the OPS243 ball speed to find the ball's aliased velocity bin. Below ~35 mph, the ball signal may be too weak to detect.
2. **Ball outside velocity search window** — The search window is ±10 mph around the OPS speed. If the OPS speed is wrong, the search misses.
3. **Low SNR** — The ball has a small radar cross section. Single-frame detections require SNR ≥ 5.0. Weaker signals are rejected.

**Diagnostic:** Check the K-LD7 buffer log:
```json
{"type": "kld7_buffer", "ball_angle": null, "frame_count": 68}
```
If `frame_count` is 68 (full buffer) but `ball_angle` is null, the RADC data is there but the ball wasn't found. This is normal for some shots — the ball doesn't always produce a strong enough return in the K-LD7's beam.

## Velocity Aliasing

The K-LD7 operates at RSPI=3 (100 km/h max speed). Ball speeds above 62 mph (100 km/h) alias into the negative velocity range:

| Ball Speed | Aliased Velocity | FFT Region |
|-----------|-----------------|------------|
| < 62 mph | Positive (no alias) | Bins 0-1024 |
| 62-124 mph | Negative (-100 to 0 km/h) | Bins 1024-2048 |
| 124-186 mph | Positive again (wraps) | Bins 0-1024 |

The RADC extraction handles aliasing automatically using the OPS243 ball speed. No user action needed.

## Antenna Spacing

The K-LD7 has two receive antennas (Rx1, Rx2) spaced 6.223 mm apart (datasheet). The RADC angle extraction uses phase interferometry between these channels. The code uses a calibrated spacing of 8.0 mm (`ANTENNA_SPACING_M` in `kld7/radc.py`) which accounts for effective electrical spacing differences.

## Serial Port Bandwidth

| Baud Rate | Throughput | RADC Capable? |
|-----------|-----------|---------------|
| 115200 | ~11.5 KB/s | No — RADC needs ~55 KB/s |
| 3000000 | ~300 KB/s | Yes |

The kld7 library negotiates the baud rate during INIT. OpenFlight always requests 3Mbaud for RADC support. If you see `Connected at 115200 baud` in the logs, the negotiation failed and you won't get RADC data.

## Log Lines Reference

Healthy startup sequence:
```
[KLD7] Connected on /dev/ttyUSB0 at 3000000 baud (attempt 1/5)
[KLD7] Configured: range=5m, speed=100km/h, orientation=vertical
[KLD7] Ready: port=/dev/ttyUSB0, baud=3000000, range=5m, speed=100km/h, orientation=vertical
[KLD7] Streaming started (orientation=vertical)
[KLD7] Stream started: RADC only (3Mbaud)
[KLD7] First RADC frame received (3072 bytes)
[KLD7] Stream health: 50 RADC frames
```

Per-shot angle extraction:
```
[KLD7] Angle extraction: ball_speed=87.2 mph, buffer=68 frames
[KLD7] RADC: examining 68 frames, ball_speed=87.2 mph
[KLD7] RADC: angle=8.0° speed=86.5 mph snr=10.7 conf=0.90 frames=1
```

Recovery from prior crash:
```
[KLD7] Connect attempt 1/5 failed: Wrong length reply
[KLD7] Sent GBYE at 3Mbaud to reset prior session
[KLD7] Connected on /dev/ttyUSB0 at 3000000 baud (attempt 2/5)
```
