"""Tests for packaging metadata that affects setup/install behavior."""

from pathlib import Path

import tomllib


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_camera_dependencies_are_not_installed_by_default():
    """Camera tracking packages should not be part of the base install."""
    dependencies = _pyproject()["project"]["dependencies"]

    assert not any(dep.startswith("trackers ") for dep in dependencies)
    assert not any(dep.startswith("supervision") for dep in dependencies)


def test_camera_extra_is_disabled_until_camera_support_returns():
    """The camera extra should not pull fragile camera packages in setup."""
    camera_dependencies = _pyproject()["project"]["optional-dependencies"]["camera"]

    assert camera_dependencies == []
