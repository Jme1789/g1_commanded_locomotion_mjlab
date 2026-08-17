"""Pure, fixed-contract mapping and diagnostics for a quickly configured gamepad."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from pydantic import JsonValue

from .linux_joystick import EventBatch, JoystickDescriptor, RawJoystickEvent
from .models import (
    BUTTON_CONTROLS,
    DPAD_CONTROLS,
    REQUIRED_G1_CONTROLS,
    STICK_CONTROLS,
    TRIGGER_CONTROLS,
    AxisDpadBinding,
    AxisTriggerBinding,
    ButtonBinding,
    ButtonTriggerBinding,
    GamepadProfile,
    LogicalState,
    StickBinding,
    UnsupportedBinding,
    validate_profile,
)
from .normalization import _normalize_profile_unchecked

QUICK_CONTROL_ORDER = (
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "lt",
    "rt",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "a",
    "b",
    "x",
    "y",
    "lb",
    "rb",
    "start",
    "back",
    "left_stick",
    "right_stick",
)
AXIS_CAPTURE_START_DELTA = 8192
AXIS_CAPTURE_MINIMUM_PEAK = 16384
AXIS_CAPTURE_WINDOW_SECONDS = 0.250
AXIS_CAPTURE_DOMINANCE_RATIO = 1.5
AXIS_CAPTURE_RECENTER_DELTA = 4096
FIXED_AXIS_MINIMUM = -32768
DIGITAL_CONTROL_ORDER = tuple(control for control in QUICK_CONTROL_ORDER if control not in STICK_CONTROLS)
FIXED_AXIS_MAXIMUM = 32767
FIXED_DEADZONE = 0.05
FIXED_THRESHOLD = 0.5

Binding = (
    StickBinding
    | AxisTriggerBinding
    | ButtonTriggerBinding
    | ButtonBinding
    | AxisDpadBinding
    | UnsupportedBinding
)


@dataclass
class _AxisCaptureWindow:
    baseline: tuple[int, ...]
    signed_peaks: list[int]
    started_at: float | None = None
    waiting_for_neutral: bool = False


class QuickMappingEngine:
    """Map one deliberate input gesture, with logical edge diagnostics."""

    def __init__(
        self,
        descriptor: JoystickDescriptor,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.descriptor = descriptor
        self._clock = clock
        self._armed_control: str | None = None
        self._axes = (0,) * descriptor.capabilities.axis_count
        self._buttons = (0,) * descriptor.capabilities.button_count
        self._previous_pressed = {control: False for control in QUICK_CONTROL_ORDER}
        self._connected = True
        self._replacement: dict[str, JsonValue] | None = None
        self._last_raw: dict[str, JsonValue] = {
            "axes": list(self._axes),
            "buttons": list(self._buttons),
            "transitions": [],
        }
        self._bindings = self._unsupported_bindings()
        self._axis_capture: _AxisCaptureWindow | None = None
        self._capture: dict[str, JsonValue] = self._capture_payload("idle")

    @staticmethod
    def _capture_payload(status: str, **values: JsonValue) -> dict[str, JsonValue]:
        return {
            "status": status,
            "control": None,
            "source": None,
            "index": None,
            "direction": None,
            "primary_axis": None,
            "secondary_axis": None,
            **values,
        }

    def arm(self, control: str) -> dict[str, JsonValue]:
        if control not in QUICK_CONTROL_ORDER:
            raise ValueError(f"unknown quick control {control}")
        if not self._connected:
            raise ValueError("gamepad is disconnected")
        self._armed_control = control
        self._axis_capture = _AxisCaptureWindow(
            baseline=self._axes,
            signed_peaks=[0] * len(self._axes),
        )
        self._capture = self._capture_payload("armed", control=control)
        self._replacement = None
        return self.snapshot()

    def observe(self, batch: EventBatch) -> dict[str, JsonValue]:
        transitions = self._classify_transitions(batch.events)
        if batch.connected:
            self._capture_input(transitions, tuple(batch.axes))
        else:
            self._armed_control = None
            self._axis_capture = None
            self._capture = self._capture_payload("idle")
        self._connected = batch.connected
        axes = batch.axes if batch.connected else (0,) * len(self._axes)
        buttons = batch.buttons if batch.connected else (0,) * len(self._buttons)
        self._axes = tuple(axes)
        self._buttons = tuple(buttons)
        self._last_raw = {
            "axes": list(self._axes),
            "buttons": list(self._buttons),
            "transitions": transitions,
        }
        return self._update_preview()

    def snapshot(self) -> dict[str, JsonValue]:
        return self._snapshot_payload(on_pressed=(), on_released=())

    def build_profile(self) -> GamepadProfile:
        profile = self._draft_profile()
        validate_profile(profile, self.descriptor.capabilities)
        return profile

    def _unsupported_bindings(self) -> dict[str, Binding]:
        return {
            control: UnsupportedBinding(unsupported=True)
            for control in QUICK_CONTROL_ORDER
        }

    def _classify_transitions(
        self, events: tuple[RawJoystickEvent, ...]
    ) -> list[dict[str, JsonValue]]:
        axes = list(self._axes)
        buttons = list(self._buttons)
        transitions: list[dict[str, JsonValue]] = []
        for event in events:
            state = axes if event.kind == "axis" else buttons
            old_value = state[event.number]
            if event.kind == "button":
                phase = (
                    "pressed"
                    if old_value == 0 and event.value != 0
                    else "released"
                    if old_value != 0 and event.value == 0
                    else "repeat"
                )
            else:
                phase = (
                    "repeat"
                    if old_value == event.value
                    else "centered"
                    if event.value == 0
                    else "changed"
                )
            transitions.append(
                {
                    "time_ms": event.time_ms,
                    "kind": event.kind,
                    "number": event.number,
                    "old_value": old_value,
                    "value": event.value,
                    "initial": event.initial,
                    "phase": phase,
                }
            )
            state[event.number] = event.value
        return transitions

    def _capture_input(
        self,
        transitions: list[dict[str, JsonValue]],
        axes: tuple[int, ...],
    ) -> None:
        if self._armed_control is None:
            return
        if self._capture_button_press(transitions):
            return
        if self._armed_control in BUTTON_CONTROLS:
            return
        self._capture_axis_window(transitions, axes)

    def _capture_button_press(
        self, transitions: list[dict[str, JsonValue]]
    ) -> bool:
        if self._armed_control is None:
            return False
        for transition in transitions:
            if (
                transition["kind"] != "button"
                or transition["initial"]
                or transition["old_value"] != 0
                or transition["value"] == 0
            ):
                continue
            binding = self._binding_for_transition(self._armed_control, transition)
            if binding is None:
                continue
            self._complete_capture(
                self._armed_control,
                binding,
                source="button",
                index=int(transition["number"]),
                direction=None,
            )
            return True
        return False

    def _capture_axis_window(
        self,
        transitions: list[dict[str, JsonValue]],
        axes: tuple[int, ...],
    ) -> None:
        control = self._armed_control
        window = self._axis_capture
        if control is None or window is None:
            return
        if window.waiting_for_neutral:
            if all(
                abs(value - baseline) <= AXIS_CAPTURE_RECENTER_DELTA
                for value, baseline in zip(axes, window.baseline, strict=True)
            ):
                window.signed_peaks = [0] * len(window.signed_peaks)
                window.started_at = None
                window.waiting_for_neutral = False
                self._capture = self._capture_payload("armed", control=control)
            return

        for transition in transitions:
            if transition["kind"] != "axis" or transition["initial"]:
                continue
            index = int(transition["number"])
            delta = int(transition["value"]) - window.baseline[index]
            if abs(delta) > abs(window.signed_peaks[index]):
                window.signed_peaks[index] = delta

        largest_peak = max((abs(value) for value in window.signed_peaks), default=0)
        now = self._clock()
        if window.started_at is None:
            if largest_peak < AXIS_CAPTURE_START_DELTA:
                return
            window.started_at = now
            self._capture = self._capture_payload("collecting", control=control)
        if now - window.started_at + 1e-12 < AXIS_CAPTURE_WINDOW_SECONDS:
            return

        ranked = sorted(
            range(len(window.signed_peaks)),
            key=lambda index: (-abs(window.signed_peaks[index]), index),
        )
        primary_axis = ranked[0]
        secondary_axis = ranked[1] if len(ranked) > 1 else None
        primary_peak = abs(window.signed_peaks[primary_axis])
        secondary_peak = (
            abs(window.signed_peaks[secondary_axis])
            if secondary_axis is not None
            else 0
        )
        is_dominant = (
            primary_peak >= AXIS_CAPTURE_MINIMUM_PEAK
            and primary_peak >= secondary_peak * AXIS_CAPTURE_DOMINANCE_RATIO
        )
        if not is_dominant:
            window.waiting_for_neutral = True
            self._capture = self._capture_payload(
                "ambiguous",
                control=control,
                primary_axis=primary_axis,
                secondary_axis=secondary_axis,
            )
            return

        signed_peak = window.signed_peaks[primary_axis]
        transition: dict[str, JsonValue] = {
            "kind": "axis",
            "number": primary_axis,
            "old_value": window.baseline[primary_axis],
            "value": window.baseline[primary_axis] + signed_peak,
        }
        binding = self._binding_for_transition(control, transition)
        if binding is None:
            window.waiting_for_neutral = True
            self._capture = self._capture_payload(
                "ambiguous",
                control=control,
                primary_axis=primary_axis,
                secondary_axis=secondary_axis,
            )
            return
        self._complete_capture(
            control,
            binding,
            source="axis",
            index=primary_axis,
            direction="negative" if signed_peak < 0 else "positive",
            primary_axis=primary_axis,
            secondary_axis=secondary_axis,
        )

    def _complete_capture(
        self,
        control: str,
        binding: Binding,
        *,
        source: str,
        index: int,
        direction: str | None,
        primary_axis: int | None = None,
        secondary_axis: int | None = None,
    ) -> None:
        self._set_binding(control, binding)
        self._armed_control = None
        self._axis_capture = None
        self._capture = self._capture_payload(
            "captured",
            control=control,
            source=source,
            index=index,
            direction=direction,
            primary_axis=primary_axis,
            secondary_axis=secondary_axis,
        )

    def _binding_for_transition(
        self, control: str, transition: dict[str, JsonValue]
    ) -> Binding | None:
        source = str(transition["kind"])
        index = int(transition["number"])
        delta = int(transition["value"]) - int(transition["old_value"])
        if control in STICK_CONTROLS:
            if source != "axis":
                return None
            return StickBinding(
                axis=index,
                center=0,
                min=FIXED_AXIS_MINIMUM,
                max=FIXED_AXIS_MAXIMUM,
                invert=delta < 0,
                deadzone=FIXED_DEADZONE,
            )
        if control in BUTTON_CONTROLS:
            return ButtonBinding(source="button", index=index) if source == "button" else None
        if control in TRIGGER_CONTROLS:
            if source == "button":
                return ButtonTriggerBinding(source="button", index=index)
            return AxisTriggerBinding(
                source="axis",
                index=index,
                released=FIXED_AXIS_MINIMUM if delta > 0 else FIXED_AXIS_MAXIMUM,
                pressed=FIXED_AXIS_MAXIMUM if delta > 0 else FIXED_AXIS_MINIMUM,
                threshold=FIXED_THRESHOLD,
            )
        if source == "button":
            return ButtonBinding(source="button", index=index)
        return AxisDpadBinding(
            source="axis",
            index=index,
            direction="negative" if delta < 0 else "positive",
            threshold=FIXED_THRESHOLD,
        )

    def _set_binding(self, control: str, binding: Binding) -> None:
        displaced = [
            existing_control
            for existing_control, existing in self._bindings.items()
            if existing_control != control
            and self._bindings_conflict(control, binding, existing_control, existing)
        ]
        for existing_control in displaced:
            self._bindings[existing_control] = UnsupportedBinding(unsupported=True)
        self._bindings[control] = binding
        self._replacement = (
            {"replaced_controls": sorted(displaced)} if displaced else None
        )

    def _bindings_conflict(
        self,
        control: str,
        binding: Binding,
        existing_control: str,
        existing: Binding,
    ) -> bool:
        if isinstance(existing, UnsupportedBinding):
            return False
        if isinstance(binding, (ButtonBinding, ButtonTriggerBinding)) and isinstance(
            existing, (ButtonBinding, ButtonTriggerBinding)
        ):
            return binding.index == existing.index
        if isinstance(binding, (StickBinding, AxisTriggerBinding)) and isinstance(
            existing, (StickBinding, AxisTriggerBinding)
        ):
            new_index = binding.axis if isinstance(binding, StickBinding) else binding.index
            old_index = existing.axis if isinstance(existing, StickBinding) else existing.index
            return new_index == old_index
        if isinstance(binding, AxisDpadBinding) and isinstance(existing, AxisDpadBinding):
            opposite = frozenset({control, existing_control}) in (
                frozenset({"dpad_up", "dpad_down"}),
                frozenset({"dpad_left", "dpad_right"}),
            )
            return binding.index == existing.index and (
                binding.direction == existing.direction or not opposite
            )
        return False

    def _draft_profile(self) -> GamepadProfile:
        return GamepadProfile(
            schema_version=1,
            device=self.descriptor.identity,
            sticks={name: self._bindings[name] for name in STICK_CONTROLS},
            triggers={name: self._bindings[name] for name in TRIGGER_CONTROLS},
            buttons={name: self._bindings[name] for name in BUTTON_CONTROLS},
            dpad={name: self._bindings[f"dpad_{name}"] for name in DPAD_CONTROLS},
        )

    def _logical_and_pressed(self) -> tuple[LogicalState, dict[str, bool]]:
        logical = _normalize_profile_unchecked(
            self._draft_profile(), self._axes, self._buttons
        )
        pressed = {
            **logical.buttons,
            **{f"dpad_{name}": value for name, value in logical.dpad.items()},
            **{
                name: logical.triggers[name] > self._bindings[name].threshold
                if isinstance(
                    self._bindings[name], (AxisTriggerBinding, ButtonTriggerBinding)
                )
                else False
                for name in TRIGGER_CONTROLS
            },
        }
        return logical, {control: pressed[control] for control in DIGITAL_CONTROL_ORDER}

    def _update_preview(self) -> dict[str, JsonValue]:
        _, pressed = self._logical_and_pressed()
        on_pressed = tuple(
            name
            for name in DIGITAL_CONTROL_ORDER
            if pressed[name] and not self._previous_pressed[name]
        )
        on_released = tuple(
            name
            for name in DIGITAL_CONTROL_ORDER
            if not pressed[name] and self._previous_pressed[name]
        )
        self._previous_pressed = pressed
        return self._snapshot_payload(
            on_pressed=on_pressed,
            on_released=on_released,
        )

    def _snapshot_payload(
        self, *, on_pressed: tuple[str, ...], on_released: tuple[str, ...]
    ) -> dict[str, JsonValue]:
        logical, pressed = self._logical_and_pressed()
        return {
            "connected": self._connected,
            "armed_control": self._armed_control,
            "capture": dict(self._capture),
            "bindings": {
                control: self._bindings[control].model_dump(mode="json")
                for control in QUICK_CONTROL_ORDER
            },
            "missing_required": sorted(
                control
                for control in REQUIRED_G1_CONTROLS
                if isinstance(self._bindings[control], UnsupportedBinding)
            ),
            "logical": logical.model_dump(mode="json"),
            "edges": {
                "pressed": pressed,
                "on_pressed": list(on_pressed),
                "on_released": list(on_released),
                "combos": {
                    "lt_up": pressed["lt"] and pressed["dpad_up"],
                    "rt_a": pressed["rt"] and pressed["a"],
                    "lb": pressed["lb"],
                    "rb": pressed["rb"],
                },
            },
            "raw": self._last_raw,
            "replacement": self._replacement,
        }
