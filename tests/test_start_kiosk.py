"""Tests for the kiosk entry script experiment flag wiring."""


def _dry_run(*args: str):
    import subprocess

    repo_root = __file__.rsplit("/tests/", 1)[0]
    return subprocess.run(
        ["bash", "scripts/start-kiosk.sh", *args, "--dry-run"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_trackman_test_dry_run_enables_raw_capture_without_calibration():
    """The field preset should capture raw replay data without changing live angles."""
    result = _dry_run("--trackman-test")
    command = result.stdout.strip()

    assert command.startswith("openflight-server --web-port 8080")
    assert "--session-location trackman" in command
    assert "--experimental-kld7-raw-radc-logging" in command
    assert "--experimental-kld7-trackman-calibration" not in command
    assert "--experimental-kld7-radc-tuning" not in command
    assert "--kld7 --kld7-port /dev/kld7_vertical --kld7-angle-offset 8" in command
    assert "--kld7-horizontal" in command
    assert "--kld7-horizontal-port /dev/kld7_horizontal" in command
    assert "--no-camera" in command
    assert "--trigger sound" in command


def test_trackman_test_allows_explicit_session_location():
    """A bay/location override should survive the TrackMan preset defaults."""
    result = _dry_run("--trackman-test", "--session-location", "trackman-bay-2")

    assert "--session-location trackman-bay-2" in result.stdout
    assert "--session-location trackman " not in result.stdout


def test_radc_tuning_values_are_ignored_without_experimental_gate():
    """Loose tuning flags should not alter production extraction by accident."""
    result = _dry_run(
        "--experimental-kld7-speed-tolerance", "6", "--experimental-kld7-spectrum-source", "sum12"
    )

    assert "--experimental-kld7-speed-tolerance 6" not in result.stdout
    assert "--experimental-kld7-spectrum-source sum12" not in result.stdout
    assert "Ignoring experimental K-LD7 RADC tuning values" in result.stdout


def test_radc_tuning_values_are_forwarded_with_experimental_gate():
    """When explicitly enabled, replay-discovered RADC knobs reach the server."""
    result = _dry_run(
        "--experimental-kld7-raw-radc-logging",
        "--experimental-kld7-radc-tuning",
        "--experimental-kld7-speed-tolerance",
        "6",
        "--experimental-kld7-spectrum-source",
        "sum12",
        "--experimental-kld7-horizontal-angle-limit",
        "30",
    )
    command = result.stdout.strip()

    assert "--experimental-kld7-raw-radc-logging" in command
    assert "--experimental-kld7-radc-tuning" in command
    assert "--experimental-kld7-speed-tolerance 6" in command
    assert "--experimental-kld7-spectrum-source sum12" in command
    assert "--experimental-kld7-horizontal-angle-limit 30" in command
