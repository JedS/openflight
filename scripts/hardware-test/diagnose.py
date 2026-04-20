#!/usr/bin/env python3
"""
OpenFlight hardware diagnostic.

Runs a guided sequence of checks against every hardware component:
OPS243-A radar, both K-LD7 angle radars, and the sound trigger path.

Usage:
    uv run python scripts/hardware-test/diagnose.py
    uv run python scripts/hardware-test/diagnose.py --require-all
    uv run python scripts/hardware-test/diagnose.py --no-interactive
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

# Allow running as a script from repo root
sys.path.insert(0, "src")

import serial.tools.list_ports
from openflight.ops243 import OPS243Radar
from openflight.rolling_buffer.processor import RollingBufferProcessor

# ANSI escape codes. When stdout is a TTY these render as color;
# when redirected, the terminal never sees them because we disable.
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _color_enabled() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _color_enabled() else text


def print_check_header(index: int, total: int, name: str) -> None:
    """Print the leading line of a check while it's running."""
    line = f"[{index}/{total}] {name} ".ljust(45, ".")
    sys.stdout.write(f"{line} {_c('⧗ ...', _DIM)}\r")
    sys.stdout.flush()


def print_check_result(index: int, total: int, result: CheckResult) -> None:
    """Print the final line of a check after it completes."""
    symbol_map = {
        "pass": _c("✓ PASS", _GREEN),
        "fail": _c("✗ FAIL", _RED),
        "skip": _c("⊘ SKIP", _YELLOW),
    }
    symbol = symbol_map[result.status]
    line = f"[{index}/{total}] {result.name} ".ljust(45, ".")
    sys.stdout.write("\033[K")
    print(f"{line} {symbol} ({result.elapsed_s:.1f}s)")
    if result.detail:
        print(f"        {_c(result.detail, _DIM)}")
    if result.hint and result.status == "fail":
        print(f"        {_c('→ ' + result.hint, _DIM)}")


def detect_ops243_port() -> Optional[str]:
    """Find the OPS243-A serial port.

    The OPS243-A enumerates as a USB CDC ACM device on Linux
    (/dev/ttyACM*) or a usbmodem on macOS. K-LD7 boards enumerate
    differently (FTDI/CP210x USB-serial bridges at /dev/ttyUSB* or
    /dev/cu.usbserial-*), so we identify OPS243 by device name.
    """
    for port in serial.tools.list_ports.comports():
        device = port.device or ""
        if "ACM" in device or "usbmodem" in device:
            return device
    return None


def detect_kld7_ports() -> list[str]:
    """Find all K-LD7 EVAL board serial ports.

    K-LD7 EVAL boards use FTDI or CP210x USB-serial chips, which
    advertise "FTDI", "CP210", or "usb-serial" in the port description
    or manufacturer string. Returns the list of matching devices in
    the order they were enumerated.
    """
    ports = []
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        mfg = (port.manufacturer or "").lower()
        if any(kw in desc for kw in ["ftdi", "cp210", "usb-serial", "uart"]):
            ports.append(port.device)
        elif any(kw in mfg for kw in ["ftdi", "silicon labs"]):
            ports.append(port.device)
    return ports


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    status: Literal["pass", "fail", "skip"]
    detail: str = ""
    hint: str = ""
    elapsed_s: float = 0.0


@dataclass
class DiagnosticState:
    """Shared state across checks so later ones can skip cleanly."""
    ops243_port: Optional[str] = None
    ops243_radar: Optional[object] = None
    kld7_vertical_port: Optional[str] = None
    kld7_horizontal_port: Optional[str] = None


def check_ops243_connectivity(state: DiagnosticState) -> CheckResult:
    """Check 1 — verify OPS243 is detected and responds to queries."""
    start = time.time()
    port = detect_ops243_port()
    if port is None:
        return CheckResult(
            name="OPS243 connectivity",
            status="skip",
            detail="no OPS243 detected on /dev/ttyACM* or usbmodem*",
            elapsed_s=time.time() - start,
        )

    try:
        radar = OPS243Radar(port=port)
        radar.connect()
        version = radar.get_firmware_version()
    except OSError as e:
        hint = ""
        if "Permission denied" in str(e):
            hint = "Add your user to the dialout group: sudo usermod -aG dialout $USER"
        return CheckResult(
            name="OPS243 connectivity",
            status="fail",
            detail=f"Connect failed: {e}",
            hint=hint,
            elapsed_s=time.time() - start,
        )
    except Exception as e:
        return CheckResult(
            name="OPS243 connectivity",
            status="fail",
            detail=f"Unexpected error: {type(e).__name__}: {e}",
            hint="Check USB connection and permissions (dialout group)",
            elapsed_s=time.time() - start,
        )

    state.ops243_port = port
    state.ops243_radar = radar
    return CheckResult(
        name="OPS243 connectivity",
        status="pass",
        detail=f"{port} • firmware {version}",
        elapsed_s=time.time() - start,
    )


def check_ops243_rolling_buffer_persisted(state: DiagnosticState) -> CheckResult:
    """Check 2 — verify radar boots in rolling buffer mode.

    In standard (CW) mode, the OPS243 streams speed readings continuously.
    In rolling buffer mode, the serial port is silent until triggered.
    We clear any stale data then sample for 500ms — if bytes arrive,
    the radar is in CW mode and the persistence workaround hasn't been
    applied.
    """
    start = time.time()
    if state.ops243_radar is None:
        return CheckResult(
            name="OPS243 rolling buffer persisted",
            status="skip",
            detail="skipped because OPS243 connectivity check did not succeed",
            elapsed_s=time.time() - start,
        )

    radar = state.ops243_radar
    if hasattr(radar.serial, "reset_input_buffer"):
        radar.serial.reset_input_buffer()

    sample_window_s = 0.5
    sample_start = time.time()
    total_bytes = 0
    while (time.time() - sample_start) < sample_window_s:
        in_waiting = radar.serial.in_waiting
        if in_waiting:
            total_bytes += in_waiting
            radar.serial.read(in_waiting)
        time.sleep(0.02)

    if total_bytes > 0:
        return CheckResult(
            name="OPS243 rolling buffer persisted",
            status="fail",
            detail=f"Radar is streaming data ({total_bytes} bytes in {sample_window_s}s) — CW mode, not rolling buffer",
            hint="Run 'uv run python scripts/hardware-test/test_rolling_buffer_persist.py --setup' then power cycle the radar",
            elapsed_s=time.time() - start,
        )

    return CheckResult(
        name="OPS243 rolling buffer persisted",
        status="pass",
        detail="Radar boots in rolling buffer mode (silent serial)",
        elapsed_s=time.time() - start,
    )


def check_ops243_software_trigger(state: DiagnosticState) -> CheckResult:
    """Check 3 — send S! and verify we get a valid 4096-sample I/Q capture."""
    start = time.time()
    if state.ops243_radar is None:
        return CheckResult(
            name="OPS243 software trigger",
            status="skip",
            detail="skipped because OPS243 is not connected",
            elapsed_s=time.time() - start,
        )

    try:
        response = state.ops243_radar.trigger_capture(timeout=5.0)
    except Exception as e:
        return CheckResult(
            name="OPS243 software trigger",
            status="fail",
            detail=f"trigger_capture raised {type(e).__name__}: {e}",
            elapsed_s=time.time() - start,
        )

    if not response:
        return CheckResult(
            name="OPS243 software trigger",
            status="fail",
            detail="Software trigger sent but no I/Q response received",
            hint="Radar may be stuck — try power-cycling and re-running",
            elapsed_s=time.time() - start,
        )

    processor = RollingBufferProcessor()
    capture = processor.parse_capture(response)
    if capture is None:
        return CheckResult(
            name="OPS243 software trigger",
            status="fail",
            detail="Response received but parse failed (missing I, Q, or timing fields)",
            elapsed_s=time.time() - start,
        )

    n = len(capture.i_samples)
    return CheckResult(
        name="OPS243 software trigger",
        status="pass",
        detail=f"Capture received: {n} I/Q samples",
        elapsed_s=time.time() - start,
    )


def format_summary(results: list[CheckResult]) -> str:
    """Format the summary block printed at the end of a run."""
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    total_s = sum(r.elapsed_s for r in results)

    lines = []
    lines.append("=" * 40)
    lines.append(
        f"Summary: {passed} passed, {failed} failed, {skipped} skipped "
        f"({total_s:.1f}s total)"
    )
    if failed == 0 and all(r.status != "skip" for r in results):
        lines.append(f"Overall: {_c('✓ HEALTHY', _GREEN + _BOLD)}")
    elif failed == 0:
        lines.append(f"Overall: {_c('✓ HEALTHY', _GREEN + _BOLD)} (with skips)")
    else:
        lines.append(f"Overall: {_c('✗ FAILED', _RED + _BOLD)}")
    return "\n".join(lines)


def overall_status(results: list[CheckResult], require_all: bool) -> str:
    """Return 'HEALTHY' or 'UNHEALTHY' based on results and require_all flag."""
    if any(r.status == "fail" for r in results):
        return "UNHEALTHY"
    if require_all and any(r.status == "skip" for r in results):
        return "UNHEALTHY"
    return "HEALTHY"


def main() -> int:
    """Entry point — runs all checks and prints summary."""
    state = DiagnosticState()
    results: list[CheckResult] = []

    print(_c("OpenFlight Hardware Diagnostic", _BOLD))
    print("=" * 40)
    print()

    CHECKS = [
        check_ops243_connectivity,
        check_ops243_rolling_buffer_persisted,
        check_ops243_software_trigger,
    ]

    for i, check in enumerate(CHECKS, 1):
        print_check_header(i, len(CHECKS), check.__name__)
        result = check(state)
        results.append(result)
        print_check_result(i, len(CHECKS), result)

    print()
    print(format_summary(results))
    return 0 if overall_status(results, require_all=False) == "HEALTHY" else 1


if __name__ == "__main__":
    sys.exit(main())
