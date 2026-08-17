"""Pure raw joystick-to-logical-state normalization."""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    AxisTriggerBinding,
    ButtonBinding,
    GamepadProfile,
    LogicalState,
    UnsupportedBinding,
    validate_profile,
)


def normalize_signed_axis(value: int) -> float:
    """Normalize one raw signed int16 axis into ``[-1, 1]``."""
    denominator = 32768 if value < 0 else 32767
    scaled = value / denominator
    return max(-1.0, min(1.0, scaled))


def normalize_stick(value: int, *, center: int, minimum: int, maximum: int, invert: bool, deadzone: float) -> float:
    """Normalize one signed stick axis into ``[-1, 1]``."""
    denominator = center - minimum if value < center else maximum - center
    scaled = 0.0 if denominator <= 0 else (value - center) / denominator
    scaled = max(-1.0, min(1.0, scaled))
    if invert:
        scaled = -scaled
    return 0.0 if abs(scaled) < deadzone else scaled


def normalize_trigger(value: int, *, released: int, pressed: int) -> float:
    """Normalize an axis trigger, preserving reversed electrical ranges."""
    denominator = pressed - released
    scaled = 0.0 if denominator == 0 else (value - released) / denominator
    return max(0.0, min(1.0, scaled))


def _normalize_profile_unchecked(
    profile: GamepadProfile, axes: Sequence[int], buttons: Sequence[int]
) -> LogicalState:
    """Normalize a complete draft without applying saved-profile validation."""
    sticks = {
        name: 0.0
        if isinstance(binding, UnsupportedBinding)
        else normalize_stick(
            axes[binding.axis], center=binding.center, minimum=binding.min,
            maximum=binding.max, invert=binding.invert, deadzone=binding.deadzone,
        )
        for name, binding in profile.sticks.items()
    }
    triggers: dict[str, float] = {}
    for name, binding in profile.triggers.items():
        if isinstance(binding, UnsupportedBinding):
            triggers[name] = 0.0
        elif isinstance(binding, AxisTriggerBinding):
            triggers[name] = normalize_trigger(
                axes[binding.index], released=binding.released, pressed=binding.pressed
            )
        else:
            triggers[name] = 1.0 if buttons[binding.index] else 0.0
    logical_buttons = {
        name: False if isinstance(binding, UnsupportedBinding) else bool(buttons[binding.index])
        for name, binding in profile.buttons.items()
    }
    dpad: dict[str, bool] = {}
    for name, binding in profile.dpad.items():
        if isinstance(binding, UnsupportedBinding):
            dpad[name] = False
        elif isinstance(binding, ButtonBinding):
            dpad[name] = bool(buttons[binding.index])
        else:
            value = normalize_signed_axis(axes[binding.index])
            dpad[name] = (
                value <= -binding.threshold
                if binding.direction == "negative"
                else value >= binding.threshold
            )
    return LogicalState(sticks=sticks, triggers=triggers, buttons=logical_buttons, dpad=dpad)


def normalize_profile(profile: GamepadProfile, axes: Sequence[int], buttons: Sequence[int]) -> LogicalState:
    """Return all standard logical controls, using neutral values when unsupported."""
    validate_profile(profile)
    return _normalize_profile_unchecked(profile, axes, buttons)
