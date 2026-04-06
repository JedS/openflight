#!/usr/bin/env python3
"""Capture K-LD7 raw ADC (RADC) data alongside PDAT/TDAT for offline analysis.

Usage:
    ./scripts/capture_kld7_radc.py --port /dev/ttyUSB0 --duration 60
    ./scripts/capture_kld7_radc.py --port /dev/ttyUSB0 --baud 3000000 --duration 30

Output:
    .pkl file with RADC + PDAT + TDAT per frame, plus metadata.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from kld7 import KLD7, FrameCode, KLD7Exception
except ImportError:
    print("kld7 package not installed. Run: pip install kld7")
    sys.exit(1)


def target_to_dict(target):
    if target is None:
        return None
    return {
        "distance": target.distance,
        "speed": target.speed,
        "angle": target.angle,
        "magnitude": target.magnitude,
    }


def read_all_params(radar):
    """Read all configurable parameters from the K-LD7."""
    param_names = [
        "RBFR", "RSPI", "RRAI", "THOF", "TRFT", "VISU",
        "MIRA", "MARA", "MIAN", "MAAN", "MISP", "MASP", "DEDI",
        "RATH", "ANTH", "SPTH", "DIG1", "DIG2", "DIG3", "HOLD", "MIDE", "MIDS",
    ]
    params = {}
    for name in param_names:
        try:
            params[name] = getattr(radar.params, name)
        except Exception:
            pass
    return params


def configure_for_golf(radar, range_m=5, speed_kmh=100):
    """Configure K-LD7 for golf ball detection."""
    range_settings = {5: 0, 10: 1, 30: 2, 100: 3}
    speed_settings = {12: 0, 25: 1, 50: 2, 100: 3}

    params = radar.params
    params.RRAI = range_settings.get(range_m, 0)
    params.RSPI = speed_settings.get(speed_kmh, 3)
    params.DEDI = 2    # Both directions
    params.THOF = 10   # Max sensitivity
    params.TRFT = 1    # Fast tracking
    params.MIAN = -90
    params.MAAN = 90
    params.MIRA = 0
    params.MARA = 100
    params.MISP = 0
    params.MASP = 100
    params.VISU = 0    # No vibration suppression


def main():
    parser = argparse.ArgumentParser(
        description="Capture K-LD7 raw ADC data for offline signal processing.",
    )
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if not set)")
    parser.add_argument("--baud", type=int, default=3000000, help="Baud rate (default: 3000000)")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration in seconds")
    parser.add_argument("--orientation", default="vertical", choices=["vertical", "horizontal"])
    parser.add_argument("--output", default=None, help="Output .pkl path")
    parser.add_argument("--club", default=None, help="Club label for metadata")
    parser.add_argument("--shots", type=int, default=None, help="Expected shot count")
    parser.add_argument("--notes", default=None, help="Freeform notes")
    args = parser.parse_args()

    # Auto-detect port
    port = args.port
    if port is None:
        from serial.tools.list_ports import comports
        for p in comports():
            desc = (p.description or "").lower()
            mfg = (p.manufacturer or "").lower()
            if any(kw in desc for kw in ["ftdi", "cp210", "usb-serial", "uart"]):
                port = p.device
                break
            if any(kw in mfg for kw in ["ftdi", "silicon labs"]):
                port = p.device
                break
        if port is None:
            print("No K-LD7 detected. Use --port to specify.")
            sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path(__file__).resolve().parent.parent / "session_logs"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"-{args.club}" if args.club else ""
        output_path = output_dir / f"kld7_radc_{timestamp}{suffix}.pkl"

    print("=" * 60)
    print("  K-LD7 Raw ADC Capture")
    print("=" * 60)
    print(f"  Port:        {port}")
    print(f"  Baud:        {args.baud}")
    print(f"  Duration:    {args.duration}s")
    print(f"  Orientation: {args.orientation}")
    print(f"  Output:      {output_path}")
    print()

    # Connect
    print("Connecting...")
    try:
        radar = KLD7(port, baudrate=args.baud)
    except (KLD7Exception, Exception) as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"  Connected: {radar}")

    # Configure
    print("Configuring for golf...")
    configure_for_golf(radar)
    all_params = read_all_params(radar)
    print()

    # Stream RADC + PDAT + TDAT
    frame_codes = FrameCode.RADC | FrameCode.PDAT | FrameCode.TDAT

    metadata = {
        "module": "K-LD7",
        "mode": "RADC",
        "port": port,
        "baud_rate": args.baud,
        "orientation": args.orientation,
        "capture_start": datetime.now().isoformat(),
        "params": all_params,
        "club": args.club,
        "expected_shots": args.shots,
        "notes": args.notes,
    }

    frames = []
    frame_count = 0
    radc_count = 0
    pdat_detection_count = 0
    start_time = time.time()

    print("-" * 60)
    print(f"Streaming RADC + PDAT + TDAT for {args.duration}s (Ctrl+C to stop)")
    print("-" * 60)

    try:
        current_frame = {"timestamp": time.time()}
        seen_in_frame = set()

        for code, payload in radar.stream_frames(frame_codes, max_count=-1):
            if time.time() - start_time >= args.duration:
                break

            if code in seen_in_frame:
                frames.append(current_frame)
                current_frame = {"timestamp": time.time()}
                seen_in_frame = set()

            seen_in_frame.add(code)

            if code == "RADC":
                current_frame["radc"] = payload  # raw bytes, parse offline
                radc_count += 1

            elif code == "TDAT":
                current_frame["tdat"] = target_to_dict(payload)
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                n_pdat = len(current_frame.get("pdat", []))
                has_radc = "Y" if "radc" in current_frame else "N"
                print(
                    f"\r  Frames: {frame_count}  RADC: {radc_count}  "
                    f"PDAT targets: {pdat_detection_count}  "
                    f"FPS: {fps:.1f}  Elapsed: {elapsed:.0f}s",
                    end="",
                    flush=True,
                )

            elif code == "PDAT":
                current_frame["pdat"] = [target_to_dict(t) for t in payload] if payload else []
                pdat_detection_count += sum(1 for _ in (payload or []))

        if seen_in_frame:
            frames.append(current_frame)

    except KeyboardInterrupt:
        pass
    except KLD7Exception as e:
        print(f"\nK-LD7 error: {e}")
    finally:
        try:
            radar.close()
        except Exception:
            pass

    metadata["capture_end"] = datetime.now().isoformat()
    metadata["total_frames"] = len(frames)
    metadata["radc_frames"] = radc_count
    metadata["pdat_detection_count"] = pdat_detection_count

    print()
    print()
    print("=" * 60)
    print(f"  Captured {len(frames)} frames ({radc_count} with RADC)")
    print(f"  PDAT detections: {pdat_detection_count}")
    print(f"  Saving to {output_path}")

    with open(output_path, "wb") as f:
        pickle.dump({"metadata": metadata, "frames": frames}, f)

    print(f"  Done ({output_path.stat().st_size / 1024:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
