"""Immutable, strict version-one gamepad profile models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: int = 1

STICK_CONTROLS = ("left_x", "left_y", "right_x", "right_y")
TRIGGER_CONTROLS = ("lt", "rt")
BUTTON_CONTROLS = ("a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick", "right_stick")
DPAD_CONTROLS = ("up", "down", "left", "right")
REQUIRED_G1_CONTROLS: frozenset[str] = frozenset(
    {"left_x", "left_y", "right_x", "dpad_up", "a", "b", "rb", "lt", "rt"}
)


_OPPOSITE_DPAD_PAIRS = frozenset(
    {frozenset({"dpad_up", "dpad_down"}), frozenset({"dpad_left", "dpad_right"})}
)
class StrictModel(BaseModel):
    """Base model that rejects format drift and prevents in-place mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DeviceIdentity(StrictModel):
    vendor_id: str = Field(pattern=r"^[0-9a-f]{4}$")
    product_id: str = Field(pattern=r"^[0-9a-f]{4}$")
    name: str
    serial: str | None

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be empty")
        return value


class DeviceCapabilities(StrictModel):
    axis_count: int = Field(ge=0)
    button_count: int = Field(ge=0)


class UnsupportedBinding(StrictModel):
    unsupported: Literal[True]


class CorrelatedButton(StrictModel):
    index: int = Field(ge=0)
    observed_within_ms: int = Field(ge=0)


class StickBinding(StrictModel):
    axis: int = Field(ge=0)
    center: int
    min: int
    max: int
    invert: bool
    deadzone: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def has_usable_range(self) -> StickBinding:
        if not self.min < self.center < self.max:
            raise ValueError("stick range must contain center")
        return self


class AxisTriggerBinding(StrictModel):
    source: Literal["axis"]
    index: int = Field(ge=0)
    released: int
    pressed: int
    threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    correlated_button: CorrelatedButton | None = None

    @model_validator(mode="after")
    def has_usable_range(self) -> AxisTriggerBinding:
        if self.released == self.pressed:
            raise ValueError("trigger released and pressed values must differ")
        return self


class ButtonTriggerBinding(StrictModel):
    source: Literal["button"]
    index: int = Field(ge=0)
    threshold: Literal[0.5] = 0.5


TriggerBinding = AxisTriggerBinding | ButtonTriggerBinding


class ButtonBinding(StrictModel):
    source: Literal["button"]
    index: int = Field(ge=0)


class AxisDpadBinding(StrictModel):
    source: Literal["axis"]
    index: int = Field(ge=0)
    direction: Literal["negative", "positive"]
    threshold: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)


DpadBinding = AxisDpadBinding | ButtonBinding


class LogicalState(StrictModel):
    sticks: dict[str, float]
    triggers: dict[str, float]
    buttons: dict[str, bool]
    dpad: dict[str, bool]


class GamepadProfile(StrictModel):
    schema_version: Literal[1]
    device: DeviceIdentity
    sticks: dict[str, StickBinding | UnsupportedBinding]
    triggers: dict[str, TriggerBinding | UnsupportedBinding]
    buttons: dict[str, ButtonBinding | UnsupportedBinding]
    dpad: dict[str, DpadBinding | UnsupportedBinding]


class TemplateProfile(StrictModel):
    schema_version: Literal[1]
    template_name: str
    sticks: dict[str, StickBinding | UnsupportedBinding]
    triggers: dict[str, TriggerBinding | UnsupportedBinding]
    buttons: dict[str, ButtonBinding | UnsupportedBinding]
    dpad: dict[str, DpadBinding | UnsupportedBinding]


class ActiveSelection(StrictModel):
    schema_version: Literal[1]
    profile: str
    device: DeviceIdentity


class StoredProfile(NamedTuple):
    profile_id: str
    path: Path
    profile: GamepadProfile


def _require_controls(values: dict[str, object], expected: tuple[str, ...], kind: str) -> None:
    missing = set(expected) - set(values)
    extra = set(values) - set(expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {kind}: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown {kind}: {', '.join(sorted(extra))}")
        raise ValueError("; ".join(details))


def _is_unsupported(binding: object) -> bool:
    return isinstance(binding, UnsupportedBinding)


def _check_capability(index: int, count: int, label: str) -> None:
    if index >= count:
        raise ValueError(f"{label} index {index} is outside detected capability {count}")


def validate_profile(profile: GamepadProfile, capabilities: DeviceCapabilities | None = None) -> None:
    """Validate cross-field profile invariants and optional detected bounds."""
    _require_controls(profile.sticks, STICK_CONTROLS, "sticks")
    _require_controls(profile.triggers, TRIGGER_CONTROLS, "triggers")
    _require_controls(profile.buttons, BUTTON_CONTROLS, "buttons")
    _require_controls(profile.dpad, DPAD_CONTROLS, "dpad")

    bindings = {
        **profile.sticks,
        **profile.triggers,
        **profile.buttons,
        **{f"dpad_{name}": binding for name, binding in profile.dpad.items()},
    }
    for name in REQUIRED_G1_CONTROLS:
        if _is_unsupported(bindings[name]):
            raise ValueError(f"required control {name} is unsupported")

    button_sources: dict[int, str] = {}
    analog_axes: dict[int, str] = {}
    dpad_axes: dict[int, tuple[str, str]] = {}
    for name, binding in bindings.items():
        if _is_unsupported(binding):
            continue
        if isinstance(binding, StickBinding):
            if binding.axis in analog_axes:
                raise ValueError(f"duplicate analog axis {binding.axis}")
            analog_axes[binding.axis] = name
            if capabilities:
                _check_capability(binding.axis, capabilities.axis_count, "axis")
        elif isinstance(binding, AxisTriggerBinding):
            if binding.index in analog_axes:
                raise ValueError(f"duplicate analog axis {binding.index}")
            analog_axes[binding.index] = name
            if capabilities:
                _check_capability(binding.index, capabilities.axis_count, "axis")
                if binding.correlated_button:
                    _check_capability(binding.correlated_button.index, capabilities.button_count, "button")
        elif isinstance(binding, AxisDpadBinding):
            existing = dpad_axes.get(binding.index)
            if existing and (
                binding.direction == existing[1]
                or frozenset({name, existing[0]}) not in _OPPOSITE_DPAD_PAIRS
            ):
                raise ValueError(f"duplicate dpad axis direction {binding.index}")
            dpad_axes[binding.index] = (name, binding.direction)
            if capabilities:
                _check_capability(binding.index, capabilities.axis_count, "axis")
        else:
            if binding.index in button_sources:
                raise ValueError(f"duplicate button {binding.index}")
            button_sources[binding.index] = name
            if capabilities:
                _check_capability(binding.index, capabilities.button_count, "button")


def profile_id_for(device: DeviceIdentity) -> str:
    """Generate a stable, path-safe profile identifier from device identity."""
    slug = re.sub(r"[^a-z0-9]+", "-", device.name.lower()).strip("-") or "gamepad"
    parts = [device.vendor_id, device.product_id, slug]
    if device.serial:
        serial_slug = re.sub(r"[^a-z0-9]+", "-", device.serial.lower()).strip("-")
        if serial_slug:
            parts.append(serial_slug)
    return "_".join(parts[:2]) + "_" + "-".join(parts[2:])
