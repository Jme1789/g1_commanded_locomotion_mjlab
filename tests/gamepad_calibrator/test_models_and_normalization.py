"""Behavioral contracts for immutable gamepad profiles and normalization."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.gamepad_calibrator.models import (
    AxisDpadBinding,
    AxisTriggerBinding,
    ButtonBinding,
    DeviceCapabilities,
    DeviceIdentity,
    GamepadProfile,
    StickBinding,
    UnsupportedBinding,
    validate_profile,
)
from src.gamepad_calibrator.normalization import (
    normalize_profile,
    normalize_stick,
    normalize_trigger,
)


def valid_profile() -> GamepadProfile:
    return GamepadProfile(
        schema_version=1,
        device=DeviceIdentity(
            vendor_id="1234", product_id="5678", name="Test pad", serial=None
        ),
        sticks={
            "left_x": StickBinding(
                axis=0, center=0, min=-32768, max=32767, invert=False, deadzone=0.05
            ),
            "left_y": StickBinding(
                axis=1, center=0, min=-32768, max=32767, invert=True, deadzone=0.05
            ),
            "right_x": StickBinding(
                axis=2, center=0, min=-32768, max=32767, invert=False, deadzone=0.05
            ),
            "right_y": UnsupportedBinding(unsupported=True),
        },
        triggers={
            "lt": AxisTriggerBinding(
                source="axis", index=3, released=-32767, pressed=32767, threshold=0.5
            ),
            "rt": AxisTriggerBinding(
                source="axis", index=4, released=32767, pressed=-32767, threshold=0.5
            ),
        },
        buttons={
            name: ButtonBinding(source="button", index=index)
            for index, name in enumerate(
                ("a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick")
            )
        },
        dpad={
            "up": AxisDpadBinding(source="axis", index=5, direction="negative", threshold=0.5),
            "down": AxisDpadBinding(source="axis", index=5, direction="positive", threshold=0.5),
            "left": UnsupportedBinding(unsupported=True),
            "right": UnsupportedBinding(unsupported=True),
        },
    )


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (DeviceIdentity, {"vendor_id": "1234", "product_id": "5678", "name": "pad", "serial": None, "extra": 1}),
        (DeviceIdentity, {"vendor_id": "123", "product_id": "5678", "name": "pad", "serial": None}),
        (DeviceIdentity, {"vendor_id": "1234", "product_id": "ABCD", "name": "pad", "serial": None}),
        (DeviceIdentity, {"vendor_id": "1234", "product_id": "5678", "name": "", "serial": None}),
        (StickBinding, {"axis": -1, "center": 0, "min": -1, "max": 1, "invert": False, "deadzone": 0.1}),
        (StickBinding, {"axis": 0, "center": 0, "min": 0, "max": 1, "invert": False, "deadzone": 0.1}),
        (StickBinding, {"axis": 0, "center": 0, "min": -1, "max": 0, "invert": False, "deadzone": 0.1}),
        (StickBinding, {"axis": 0, "center": 0, "min": -1, "max": 1, "invert": False, "deadzone": math.nan}),
        (AxisTriggerBinding, {"source": "axis", "index": -1, "released": 0, "pressed": 1, "threshold": 0.5}),
        (AxisTriggerBinding, {"source": "axis", "index": 0, "released": 0, "pressed": 1, "threshold": math.inf}),
        (AxisDpadBinding, {"source": "axis", "index": 0, "direction": "negative", "threshold": 1.1}),
    ],
)
def test_model_rejects_invalid_data(factory, kwargs) -> None:
    """A missing validator must reject impossible device and binding data."""
    with pytest.raises(ValidationError):
        factory(**kwargs)


def test_model_rejects_numeric_and_boolean_strings() -> None:
    """Accepting wire-format strings would allow schema-invalid profile data through."""
    with pytest.raises(ValidationError):
        StickBinding(
            axis="0",
            center="0",
            min="-1",
            max="1",
            invert="false",
            deadzone="0.05",
        )


def test_profile_rejects_schema_other_than_one() -> None:
    """Changing the fixed format version must not silently parse as a supported profile."""
    payload = valid_profile().model_dump(mode="json")
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        GamepadProfile.model_validate(payload)


def test_validation_requires_each_required_control_to_be_bound() -> None:
    """A required G1 command must never fall back to an unsupported binding."""
    profile = valid_profile().model_copy(
        update={"sticks": {**valid_profile().sticks, "left_x": UnsupportedBinding(unsupported=True)}}
    )
    with pytest.raises(ValueError, match="left_x"):
        validate_profile(profile)


def test_validation_rejects_duplicate_logical_sources_except_dpad_opposites() -> None:
    """Duplicating physical sources would make a single input drive multiple commands."""
    profile = valid_profile().model_copy(
        update={"buttons": {**valid_profile().buttons, "b": ButtonBinding(source="button", index=0)}}
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_profile(profile)


def test_validation_rejects_more_than_opposite_dpad_directions_on_one_axis() -> None:
    """A third D-pad direction on one axis creates an ambiguous physical binding."""
    profile = valid_profile().model_copy(
        update={
            "dpad": {
                **valid_profile().dpad,
                "left": AxisDpadBinding(
                    source="axis", index=5, direction="negative", threshold=0.5
                ),
            }
        }
    )
    with pytest.raises(ValueError, match="dpad"):
        validate_profile(profile)
def test_validation_allows_axis_sharing_only_for_a_matching_dpad_pair() -> None:
    """Sharing Up's axis with Right changes the physical direction being represented."""
    profile = valid_profile().model_copy(
        update={
            "dpad": {
                **valid_profile().dpad,
                "down": UnsupportedBinding(unsupported=True),
                "left": AxisDpadBinding(
                    source="axis", index=5, direction="positive", threshold=0.5
                ),
            }
        }
    )
    with pytest.raises(ValueError, match="dpad"):
        validate_profile(profile)

def test_validation_enforces_detected_capability_bounds() -> None:
    """A profile for another controller must not reference absent axes or buttons."""
    with pytest.raises(ValueError, match="axis"):
        validate_profile(valid_profile(), DeviceCapabilities(axis_count=4, button_count=10))


def test_normalizes_sticks_and_triggers_using_fixed_electrical_contract() -> None:
    """Incorrect sign, deadzone, or reversed trigger handling corrupts logical input."""
    assert normalize_stick(-16384, center=0, minimum=-32768, maximum=32767, invert=False, deadzone=0.05) == pytest.approx(-0.5, abs=1e-4)
    assert normalize_stick(16384, center=0, minimum=-32768, maximum=32767, invert=True, deadzone=0.05) == pytest.approx(-0.5, abs=1e-4)
    assert normalize_stick(500, center=0, minimum=-32768, maximum=32767, invert=False, deadzone=0.05) == 0.0
    assert normalize_trigger(-32767, released=-32767, pressed=32767) == 0.0
    assert normalize_trigger(32767, released=-32767, pressed=32767) == 1.0
    assert normalize_trigger(-32767, released=32767, pressed=-32767) == 1.0


def test_normalize_profile_rejects_incomplete_profile_before_reading_inputs() -> None:
    """An incomplete mapping must fail closed instead of publishing partial logical state."""
    incomplete = GamepadProfile(
        schema_version=1,
        device=DeviceIdentity(
            vendor_id="1234", product_id="5678", name="Test pad", serial=None
        ),
        sticks={},
        triggers={},
        buttons={},
        dpad={},
    )
    with pytest.raises(ValueError, match="missing"):
        normalize_profile(incomplete, axes=(), buttons=())


def test_normalize_profile_returns_all_logical_outputs_with_neutral_unsupported() -> None:
    """Consumers require a complete stable logical state, including optional neutral controls."""
    state = normalize_profile(valid_profile(), axes=(-16384, 16384, 0, -32767, -32767, -16384), buttons=(1,) + (0,) * 9)
    assert state.sticks == {"left_x": pytest.approx(-0.5, abs=1e-4), "left_y": pytest.approx(-0.5, abs=1e-4), "right_x": 0.0, "right_y": 0.0}
    assert state.triggers == {"lt": 0.0, "rt": 1.0}
    assert state.buttons["a"] is True
    assert state.buttons["b"] is False
    assert state.dpad == {"up": True, "down": False, "left": False, "right": False}


@pytest.mark.parametrize(
    ("raw_axis", "expected_up", "expected_down"),
    [
        (-1, False, False),
        (1, False, False),
        (-16383, False, False),
        (16383, False, False),
        (-16384, True, False),
        (16384, False, True),
        (-32768, True, False),
        (32767, False, True),
    ],
)
def test_normalize_profile_applies_dpad_threshold_to_normalized_axis(
    raw_axis: int, expected_up: bool, expected_down: bool
) -> None:
    """Raw int16 noise must not satisfy a D-pad threshold expressed as a ratio."""
    state = normalize_profile(
        valid_profile(),
        axes=(0, 0, 0, -32767, -32767, raw_axis),
        buttons=(0,) * 10,
    )

    assert state.dpad["up"] is expected_up
    assert state.dpad["down"] is expected_down
