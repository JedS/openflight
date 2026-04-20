"""Tests for the hardware diagnostic script."""

import sys
from pathlib import Path

# diagnose.py is a script — import it as a module for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "hardware-test"))

import diagnose


class TestCheckResult:
    def test_default_status_fields(self):
        r = diagnose.CheckResult(name="Test", status="pass")
        assert r.name == "Test"
        assert r.status == "pass"
        assert r.detail == ""
        assert r.hint == ""
        assert r.elapsed_s == 0.0

    def test_with_all_fields(self):
        r = diagnose.CheckResult(
            name="Test", status="fail", detail="something broke",
            hint="try this", elapsed_s=1.5,
        )
        assert r.detail == "something broke"
        assert r.hint == "try this"
        assert r.elapsed_s == 1.5


class TestFormatSummary:
    def test_all_pass(self):
        results = [
            diagnose.CheckResult(name="A", status="pass", elapsed_s=1.0),
            diagnose.CheckResult(name="B", status="pass", elapsed_s=2.0),
        ]
        summary = diagnose.format_summary(results)
        assert "2 passed" in summary
        assert "0 failed" in summary
        assert "HEALTHY" in summary

    def test_with_failure(self):
        results = [
            diagnose.CheckResult(name="A", status="pass"),
            diagnose.CheckResult(name="B", status="fail"),
        ]
        summary = diagnose.format_summary(results)
        assert "1 passed" in summary
        assert "1 failed" in summary
        assert "HEALTHY" not in summary

    def test_with_skip(self):
        results = [
            diagnose.CheckResult(name="A", status="pass"),
            diagnose.CheckResult(name="B", status="skip"),
        ]
        summary = diagnose.format_summary(results)
        assert "1 skipped" in summary


class TestOverallStatus:
    def test_healthy_when_all_pass(self):
        results = [diagnose.CheckResult(name="A", status="pass")]
        assert diagnose.overall_status(results, require_all=False) == "HEALTHY"

    def test_healthy_with_skips_when_not_require_all(self):
        results = [
            diagnose.CheckResult(name="A", status="pass"),
            diagnose.CheckResult(name="B", status="skip"),
        ]
        assert diagnose.overall_status(results, require_all=False) == "HEALTHY"

    def test_unhealthy_with_skips_when_require_all(self):
        results = [
            diagnose.CheckResult(name="A", status="pass"),
            diagnose.CheckResult(name="B", status="skip"),
        ]
        assert diagnose.overall_status(results, require_all=True) == "UNHEALTHY"

    def test_unhealthy_on_any_fail(self):
        results = [diagnose.CheckResult(name="A", status="fail")]
        assert diagnose.overall_status(results, require_all=False) == "UNHEALTHY"


from unittest.mock import MagicMock, patch


class TestDetectOps243Port:
    @patch("diagnose.serial.tools.list_ports.comports")
    def test_finds_acm_device(self, mock_comports):
        mock_port = MagicMock()
        mock_port.device = "/dev/ttyACM0"
        mock_comports.return_value = [mock_port]
        assert diagnose.detect_ops243_port() == "/dev/ttyACM0"

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_finds_usbmodem_device(self, mock_comports):
        mock_port = MagicMock()
        mock_port.device = "/dev/cu.usbmodem14301"
        mock_comports.return_value = [mock_port]
        assert diagnose.detect_ops243_port() == "/dev/cu.usbmodem14301"

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_returns_none_when_nothing_matches(self, mock_comports):
        mock_port = MagicMock()
        mock_port.device = "/dev/ttyS0"
        mock_comports.return_value = [mock_port]
        assert diagnose.detect_ops243_port() is None

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_empty_port_list(self, mock_comports):
        mock_comports.return_value = []
        assert diagnose.detect_ops243_port() is None


class TestDetectKld7Ports:
    def _make_port(self, device, desc="", mfg=""):
        p = MagicMock()
        p.device = device
        p.description = desc
        p.manufacturer = mfg
        return p

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_finds_ftdi_port(self, mock_comports):
        mock_comports.return_value = [
            self._make_port("/dev/ttyUSB0", desc="FTDI USB-Serial"),
        ]
        assert diagnose.detect_kld7_ports() == ["/dev/ttyUSB0"]

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_finds_cp210x_port(self, mock_comports):
        mock_comports.return_value = [
            self._make_port("/dev/ttyUSB1", desc="CP210x UART Bridge"),
        ]
        assert diagnose.detect_kld7_ports() == ["/dev/ttyUSB1"]

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_finds_two_ports(self, mock_comports):
        mock_comports.return_value = [
            self._make_port("/dev/ttyUSB0", desc="FTDI USB-Serial"),
            self._make_port("/dev/ttyUSB1", desc="CP210x UART"),
        ]
        assert diagnose.detect_kld7_ports() == ["/dev/ttyUSB0", "/dev/ttyUSB1"]

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_ignores_acm_devices(self, mock_comports):
        """ACM devices are OPS243, not K-LD7."""
        mock_comports.return_value = [
            self._make_port("/dev/ttyACM0", desc="OPS243"),
        ]
        assert diagnose.detect_kld7_ports() == []

    @patch("diagnose.serial.tools.list_ports.comports")
    def test_empty_list(self, mock_comports):
        mock_comports.return_value = []
        assert diagnose.detect_kld7_ports() == []
