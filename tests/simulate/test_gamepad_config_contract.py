"""Behavioral and structural contracts for shipped simulator gamepad config."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from src.gamepad_calibrator.models import (
    DeviceCapabilities,
    DeviceIdentity,
    GamepadProfile,
    TemplateProfile,
    UnsupportedBinding,
    validate_profile,
)
from src.gamepad_calibrator.profile_store import ProfileStore

ROOT = Path(__file__).parents[2]
GAMEPAD_ROOT = ROOT / "simulate/config/gamepads"


@pytest.mark.parametrize(
    ("template_id", "axis_count", "button_count", "expected"),
    (
        (
            "xbox_standard",
            8,
            8,
            {
                "sticks": {"left_x": 0, "left_y": 1, "right_x": 3, "right_y": 4},
                "triggers": {"lt": 2, "rt": 5},
                "buttons": {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 4, "rb": 5, "back": 6, "start": 7},
            },
        ),
        (
            "switch_standard",
            8,
            12,
            {
                "sticks": {"left_x": 0, "left_y": 1, "right_x": 2, "right_y": 3},
                "triggers": {"lt": 5, "rt": 4},
                "buttons": {"a": 0, "b": 1, "x": 3, "y": 4, "lb": 6, "rb": 7, "back": 10, "start": 11},
            },
        ),
    ),
)
def test_shipped_template_materializes_and_validates_for_compatible_device(
    template_id: str,
    axis_count: int,
    button_count: int,
    expected: dict[str, dict[str, int]],
) -> None:
    """Layout drift or embedded identity would make a shipped template unsafe to reuse."""
    store = ProfileStore(GAMEPAD_ROOT)
    records = {record.template_id: record for record in store.list_template_records()}
    assert set(records) == {"switch_standard", "xbox_standard"}
    payload = yaml.safe_load(records[template_id].path.read_text(encoding="utf-8"))
    template = TemplateProfile.model_validate(payload)
    assert "device" not in payload
    assert template.schema_version == 1
    assert template.template_name

    device = DeviceIdentity(
        vendor_id="1234", product_id="abcd", name="Compatible Test Pad", serial=None
    )
    capabilities = DeviceCapabilities(axis_count=axis_count, button_count=button_count)
    profile = store.materialize_template(template_id, device, capabilities)
    assert isinstance(profile, GamepadProfile)
    assert profile.device == device
    validate_profile(profile, capabilities)

    assert {name: binding.axis for name, binding in profile.sticks.items()} == expected["sticks"]
    assert {name: binding.index for name, binding in profile.triggers.items()} == expected["triggers"]
    assert {
        name: binding.index
        for name, binding in profile.buttons.items()
        if name in expected["buttons"]
    } == expected["buttons"]
    assert all(
        isinstance(profile.buttons[name], UnsupportedBinding)
        for name in ("left_stick", "right_stick")
    )
    assert {name: binding.index for name, binding in profile.dpad.items()} == {
        "up": 7,
        "down": 7,
        "left": 6,
        "right": 6,
    }
    for binding in profile.sticks.values():
        assert (binding.min, binding.max) == (-32768, 32767)
    for binding in profile.triggers.values():
        assert (binding.released, binding.pressed, binding.threshold) == (
            -32768,
            32767,
            0.5,
        )


def test_simulator_uses_only_profile_based_gamepad_config() -> None:
    """The simulator config must expose one active profile path and no legacy layout knobs."""
    cfg = yaml.safe_load((ROOT / "simulate/config.yaml").read_text(encoding="utf-8"))
    assert cfg["use_joystick"] == 1
    assert cfg["gamepad_config"] == "config/gamepads/active.yaml"
    assert "joystick_type" not in cfg
    assert "joystick_device" not in cfg
    assert "joystick_bits" not in cfg
    assert not (ROOT / "simulate/src/physics_joystick.h").exists()


def test_generated_gamepad_profiles_are_ignored_but_directory_is_shipped() -> None:
    """Machine identity profiles must stay local while the empty storage directory ships."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "simulate/config/gamepads/profiles/local-device.yaml"],
        cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0
    assert (GAMEPAD_ROOT / "profiles/.gitkeep").is_file()
    keep = subprocess.run(
        ["git", "check-ignore", "-q", "simulate/config/gamepads/profiles/.gitkeep"],
        cwd=ROOT,
        check=False,
    )
    assert keep.returncode == 1
