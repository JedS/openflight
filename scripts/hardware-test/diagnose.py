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
from dataclasses import dataclass
from typing import Literal, Optional

# Allow running as a script from repo root
sys.path.insert(0, "src")

import serial.tools.list_ports

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

    # Checks will be added in subsequent tasks
    CHECKS: list = []

    for check in CHECKS:
        result = check(state)
        results.append(result)

    print()
    print(format_summary(results))
    return 0 if overall_status(results, require_all=False) == "HEALTHY" else 1


if __name__ == "__main__":
    sys.exit(main())
