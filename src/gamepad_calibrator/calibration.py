"""Deterministic guided calibration and logical input preview."""

from __future__ import annotations

import statistics
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum

from .linux_joystick import EventBatch, JoystickDescriptor
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
    TemplateProfile,
    UnsupportedBinding,
    validate_profile,
)
from .normalization import (
    normalize_signed_axis,
    normalize_stick,
    normalize_trigger,
)

NEUTRAL_SAMPLE_SECONDS = 1.0
PUMP_INTERVAL_SECONDS = 0.02
MIN_AXIS_EXCURSION = 4096
MIN_FULL_RANGE_EXCURSION = 16384
DOMINANCE_RATIO = 1.5
STABLE_SAMPLE_COUNT = 3
STABLE_SECONDS = 0.10

CALIBRATION_ORDER = (
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


class CalibrationState(StrEnum):
    NEUTRAL = "neutral"
    READY = "ready"
    CAPTURING = "capturing"
    REVIEW = "review"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class Candidate:
    control: str
    binding: dict[str, object]
    score: float
    ambiguous_with: tuple[dict[str, object], ...]


Binding = (
    StickBinding
    | AxisTriggerBinding
    | ButtonTriggerBinding
    | ButtonBinding
    | AxisDpadBinding
    | UnsupportedBinding
)


class CalibrationEngine:
    """Advance calibration only when a caller supplies a coherent event batch."""

    def __init__(self, descriptor: JoystickDescriptor, started_at: float) -> None:
        self._descriptor = descriptor
        self._started_at = started_at
        self._state = CalibrationState.NEUTRAL
        self._template_applied = False
        self._neutral_axes: list[list[int]] = [
            [] for _ in range(descriptor.capabilities.axis_count)
        ]
        self._neutral_buttons: list[list[int]] = [
            [] for _ in range(descriptor.capabilities.button_count)
        ]
        self._centers = [0] * descriptor.capabilities.axis_count
        self._noise = [0] * descriptor.capabilities.axis_count
        self._button_baseline = [0] * descriptor.capabilities.button_count
        self._axes = tuple(0 for _ in range(descriptor.capabilities.axis_count))
        self._buttons = tuple(0 for _ in range(descriptor.capabilities.button_count))
        self._bindings: dict[str, Binding] = {}
        self._control: str | None = None
        self._candidate: Candidate | None = None
        self._capture_min: list[int] = []
        self._capture_max: list[int] = []
        self._capture_buttons_seen: set[int] = set()
        self._stable_source: tuple[str, int] | None = None
        self._stable_count = 0
        self._stable_since = 0.0
        self._changed_axis_events: list[tuple[int, int]] = []
        self._changed_button_events: list[tuple[int, int]] = []

    @property
    def state(self) -> CalibrationState:
        return self._state

    def observe(self, batch: EventBatch, now: float) -> None:
        """Advance raw state, neutral sampling, and candidate stability."""
        if self._state is CalibrationState.DISCONNECTED:
            return
        if not batch.connected:
            self._state = CalibrationState.DISCONNECTED
            self._candidate = None
            return
        if len(batch.axes) != self._descriptor.capabilities.axis_count:
            raise ValueError("axis snapshot does not match detected capability")
        if len(batch.buttons) != self._descriptor.capabilities.button_count:
            raise ValueError("button snapshot does not match detected capability")
        self._axes = batch.axes
        self._buttons = batch.buttons

        if self._state is CalibrationState.NEUTRAL:
            self._observe_neutral(now)
        elif self._state is CalibrationState.CAPTURING:
            self._observe_capture(batch, now)

    def _observe_neutral(self, now: float) -> None:
        for index, value in enumerate(self._axes):
            self._neutral_axes[index].append(value)
        for index, value in enumerate(self._buttons):
            self._neutral_buttons[index].append(value)
        if now - self._started_at < NEUTRAL_SAMPLE_SECONDS:
            return

        for index, samples in enumerate(self._neutral_axes):
            center = int(statistics.median(samples))
            self._centers[index] = center
            self._noise[index] = max(abs(value - center) for value in samples)
        for index, samples in enumerate(self._neutral_buttons):
            self._button_baseline[index] = int(statistics.median(samples))
        self._state = (
            CalibrationState.REVIEW
            if self._template_applied
            else CalibrationState.READY
        )

    def begin_step(self, control: str) -> None:
        if control not in CALIBRATION_ORDER:
            raise ValueError(f"unknown control {control}")
        if self._state not in (CalibrationState.READY, CalibrationState.REVIEW):
            raise RuntimeError(f"cannot begin calibration while {self._state.value}")
        self._start_capture(control)

    def _start_capture(self, control: str) -> None:
        self._control = control
        self._candidate = None
        self._capture_min = list(self._centers)
        self._capture_max = list(self._centers)
        self._capture_buttons_seen = set()
        self._stable_source = None
        self._stable_count = 0
        self._stable_since = 0.0
        self._changed_axis_events = []
        self._changed_button_events = []
        self._state = CalibrationState.CAPTURING

    def _observe_capture(self, batch: EventBatch, now: float) -> None:
        for index, value in enumerate(self._axes):
            self._capture_min[index] = min(self._capture_min[index], value)
            self._capture_max[index] = max(self._capture_max[index], value)
        for index, value in enumerate(self._buttons):
            if value != self._button_baseline[index]:
                self._capture_buttons_seen.add(index)
        for event in batch.events:
            if event.initial:
                continue
            if event.kind == "axis" and event.value != self._centers[event.number]:
                self._changed_axis_events.append((event.number, event.time_ms))
            elif (
                event.kind == "button"
                and event.value != self._button_baseline[event.number]
            ):
                self._changed_button_events.append((event.number, event.time_ms))

        ranked_sources = self._rank_current_sources()
        if not ranked_sources:
            self._reset_stability()
            return
        source = ranked_sources[0][0]
        if source == self._stable_source:
            self._stable_count += 1
        else:
            self._stable_source = source
            self._stable_count = 1
            self._stable_since = now
        if (
            self._stable_count < STABLE_SAMPLE_COUNT
            or now - self._stable_since < STABLE_SECONDS
        ):
            self._candidate = None
            return
        self._candidate = self._build_candidate(source)

    def _reset_stability(self) -> None:
        self._stable_source = None
        self._stable_count = 0
        self._candidate = None

    def _rank_current_sources(self) -> list[tuple[tuple[str, int], float]]:
        assert self._control is not None
        sources: list[tuple[tuple[str, int], float]] = []
        if self._control in STICK_CONTROLS or self._control in TRIGGER_CONTROLS or self._control.startswith("dpad_"):
            for index, value in enumerate(self._axes):
                excursion = abs(value - self._centers[index])
                minimum = max(MIN_AXIS_EXCURSION, 4 * self._noise[index])
                if excursion > minimum:
                    sources.append((("axis", index), float(excursion)))
        if self._control not in STICK_CONTROLS:
            for index, value in enumerate(self._buttons):
                if value != self._button_baseline[index]:
                    sources.append((("button", index), 1.0))
        return sorted(sources, key=lambda item: (-item[1], item[0][0], item[0][1]))

    def _axis_score(self, index: int) -> float:
        return float(
            max(
                abs(self._capture_min[index] - self._centers[index]),
                abs(self._capture_max[index] - self._centers[index]),
            )
        )

    def _build_candidate(self, source: tuple[str, int]) -> Candidate | None:
        binding = self._binding_for_source(source)
        if binding is None:
            return None
        entries: list[tuple[float, tuple[str, int], dict[str, object]]] = []
        for candidate_source in self._historical_sources():
            if candidate_source == source:
                continue
            score = (
                self._axis_score(candidate_source[1])
                if candidate_source[0] == "axis"
                else 1.0
            )
            candidate_binding = self._binding_for_source(candidate_source)
            description = (
                candidate_binding.model_dump(mode="python")
                if candidate_binding is not None
                else self._incomplete_source_description(candidate_source, score)
            )
            entries.append((score, candidate_source, description))
        entries.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        top_score = self._axis_score(source[1]) if source[0] == "axis" else 1.0
        ambiguous: tuple[dict[str, object], ...] = ()
        if entries and top_score < DOMINANCE_RATIO * entries[0][0]:
            ambiguous = tuple(
                description
                for score, _candidate_source, description in entries
                if top_score < DOMINANCE_RATIO * score
            )
        assert self._control is not None
        return Candidate(
            control=self._control,
            binding=binding.model_dump(mode="python"),
            score=top_score,
            ambiguous_with=ambiguous,
        )

    def _incomplete_source_description(
        self, source: tuple[str, int], score: float
    ) -> dict[str, object]:
        kind, index = source
        description: dict[str, object] = {
            "source": kind,
            "index": index,
            "score": score,
            "complete_binding": False,
        }
        if kind == "axis":
            description.update(
                observed_min=self._capture_min[index],
                observed_max=self._capture_max[index],
            )
        return description

    def _historical_sources(self) -> list[tuple[str, int]]:
        assert self._control is not None
        sources: list[tuple[str, int]] = []
        if (
            self._control in STICK_CONTROLS
            or self._control in TRIGGER_CONTROLS
            or self._control.startswith("dpad_")
        ):
            for index in range(len(self._axes)):
                minimum = max(MIN_AXIS_EXCURSION, 4 * self._noise[index])
                if self._axis_score(index) > minimum:
                    sources.append(("axis", index))
        if self._control not in STICK_CONTROLS:
            sources.extend(("button", index) for index in self._capture_buttons_seen)
        return sources

    def _binding_for_source(self, source: tuple[str, int]) -> Binding | None:
        assert self._control is not None
        kind, index = source
        if kind == "button":
            if self._control in TRIGGER_CONTROLS:
                return ButtonTriggerBinding(source="button", index=index)
            return ButtonBinding(source="button", index=index)
        if self._control in STICK_CONTROLS:
            center = self._centers[index]
            minimum = self._capture_min[index]
            maximum = self._capture_max[index]
            if (
                center - minimum < MIN_FULL_RANGE_EXCURSION
                or maximum - center < MIN_FULL_RANGE_EXCURSION
            ):
                return None
            span = max(abs(minimum - center), abs(maximum - center), 1)
            deadzone = min(
                0.25,
                max(0.02, self._noise[index] / span + 0.01),
            )
            return StickBinding(
                axis=index,
                center=center,
                min=minimum,
                max=maximum,
                invert=self._control.endswith("_y"),
                deadzone=deadzone,
            )
        if self._control in TRIGGER_CONTROLS:
            released = self._centers[index]
            minimum = self._capture_min[index]
            maximum = self._capture_max[index]
            pressed = (
                minimum
                if abs(minimum - released) > abs(maximum - released)
                else maximum
            )
            if pressed == released:
                return None
            correlated = self._correlated_button(index)
            return AxisTriggerBinding(
                source="axis",
                index=index,
                released=released,
                pressed=pressed,
                threshold=0.5,
                correlated_button=correlated,
            )
        direction = (
            "negative"
            if abs(self._capture_min[index] - self._centers[index])
            >= abs(self._capture_max[index] - self._centers[index])
            else "positive"
        )
        return AxisDpadBinding(
            source="axis", index=index, direction=direction, threshold=0.5
        )

    def _correlated_button(self, axis_index: int) -> dict[str, int] | None:
        axis_times = [
            time_ms
            for changed_index, time_ms in self._changed_axis_events
            if changed_index == axis_index
        ]
        matches = [
            (abs(axis_time - button_time), button_index)
            for axis_time in axis_times
            for button_index, button_time in self._changed_button_events
            if abs(axis_time - button_time) <= 100
        ]
        if not matches:
            return None
        elapsed, button_index = min(matches)
        return {"index": button_index, "observed_within_ms": elapsed}

    def candidate(self) -> Candidate | None:
        return self._candidate

    def confirm(self, override: dict[str, object] | None = None) -> None:
        self._require_connected()
        if self._state is not CalibrationState.CAPTURING or self._control is None:
            raise RuntimeError("no calibration step is being captured")
        if override is None:
            if self._candidate is None:
                raise RuntimeError("no stable candidate")
            if self._candidate.ambiguous_with:
                raise RuntimeError("candidate is ambiguous")
            payload = self._candidate.binding
        else:
            payload = override
        binding = self._validate_control_binding(self._control, payload)
        self._validate_source_available(self._control, binding)
        self._bindings[self._control] = binding
        self._control = None
        self._candidate = None
        self._state = CalibrationState.REVIEW

    def redo(self, control: str) -> None:
        self._require_connected()
        if control not in CALIBRATION_ORDER:
            raise ValueError(f"unknown control {control}")
        if self._state not in (CalibrationState.READY, CalibrationState.REVIEW, CalibrationState.CAPTURING):
            raise RuntimeError(f"cannot redo calibration while {self._state.value}")
        self._bindings.pop(control, None)
        self._start_capture(control)

    def mark_unsupported(self, control: str) -> None:
        self._require_connected()
        if control not in CALIBRATION_ORDER:
            raise ValueError(f"unknown control {control}")
        if control in REQUIRED_G1_CONTROLS:
            raise ValueError(f"required control {control} cannot be unsupported")
        if self._state not in (
            CalibrationState.READY,
            CalibrationState.REVIEW,
            CalibrationState.CAPTURING,
        ):
            raise RuntimeError(f"cannot mark unsupported while {self._state.value}")
        self._bindings[control] = UnsupportedBinding(unsupported=True)
        self._control = None
        self._candidate = None
        self._state = CalibrationState.REVIEW

    def preview(self) -> LogicalState:
        self._require_connected()
        sticks: dict[str, float] = {}
        for name in STICK_CONTROLS:
            binding = self._bindings.get(name)
            sticks[name] = (
                normalize_stick(
                    self._axes[binding.axis],
                    center=binding.center,
                    minimum=binding.min,
                    maximum=binding.max,
                    invert=binding.invert,
                    deadzone=binding.deadzone,
                )
                if isinstance(binding, StickBinding)
                else 0.0
            )
        triggers: dict[str, float] = {}
        for name in TRIGGER_CONTROLS:
            binding = self._bindings.get(name)
            if isinstance(binding, AxisTriggerBinding):
                triggers[name] = normalize_trigger(
                    self._axes[binding.index],
                    released=binding.released,
                    pressed=binding.pressed,
                )
            elif isinstance(binding, ButtonTriggerBinding):
                triggers[name] = 1.0 if self._buttons[binding.index] else 0.0
            else:
                triggers[name] = 0.0
        buttons = {
            name: bool(self._buttons[binding.index])
            if isinstance((binding := self._bindings.get(name)), ButtonBinding)
            else False
            for name in BUTTON_CONTROLS
        }
        dpad: dict[str, bool] = {}
        for name in DPAD_CONTROLS:
            binding = self._bindings.get(f"dpad_{name}")
            if isinstance(binding, ButtonBinding):
                dpad[name] = bool(self._buttons[binding.index])
            elif isinstance(binding, AxisDpadBinding):
                value = normalize_signed_axis(self._axes[binding.index])
                dpad[name] = (
                    value <= -binding.threshold
                    if binding.direction == "negative"
                    else value >= binding.threshold
                )
            else:
                dpad[name] = False
        return LogicalState(
            sticks=sticks, triggers=triggers, buttons=buttons, dpad=dpad
        )

    def build_profile(
        self, preview_confirmations: Collection[str]
    ) -> GamepadProfile:
        self._require_connected()
        if self._state is not CalibrationState.REVIEW:
            raise RuntimeError(
                f"cannot build profile while {self._state.value} sampling is active"
            )
        unknown = set(preview_confirmations) - set(CALIBRATION_ORDER)
        if unknown:
            raise ValueError(f"unknown preview confirmation {min(unknown)}")
        missing_mappings = set(CALIBRATION_ORDER) - set(self._bindings)
        if missing_mappings:
            raise ValueError(f"missing mappings: {', '.join(sorted(missing_mappings))}")
        supported = {
            name
            for name, binding in self._bindings.items()
            if not isinstance(binding, UnsupportedBinding)
        }
        missing_confirmations = supported - set(preview_confirmations)
        if missing_confirmations:
            raise ValueError(
                "missing preview confirmation: "
                + ", ".join(sorted(missing_confirmations))
            )
        profile = self._profile_from_bindings()
        validate_profile(profile, self._descriptor.capabilities)
        return profile

    def apply_template(self, template: TemplateProfile) -> None:
        self._require_connected()
        if self._state is not CalibrationState.NEUTRAL:
            raise RuntimeError("template can only be applied during neutral sampling")
        bindings: dict[str, Binding] = {
            **template.sticks,
            **template.triggers,
            **template.buttons,
            **{f"dpad_{name}": binding for name, binding in template.dpad.items()},
        }
        profile = GamepadProfile(
            schema_version=1,
            device=self._descriptor.identity,
            sticks=template.sticks,
            triggers=template.triggers,
            buttons=template.buttons,
            dpad=template.dpad,
        )
        validate_profile(profile, self._descriptor.capabilities)
        self._bindings = bindings
        self._template_applied = True

    def _profile_from_bindings(self) -> GamepadProfile:
        return self._profile_from_mapping(self._bindings)

    def _profile_from_mapping(self, bindings: dict[str, Binding]) -> GamepadProfile:
        return GamepadProfile(
            schema_version=1,
            device=self._descriptor.identity,
            sticks={name: bindings[name] for name in STICK_CONTROLS},
            triggers={name: bindings[name] for name in TRIGGER_CONTROLS},
            buttons={name: bindings[name] for name in BUTTON_CONTROLS},
            dpad={name: bindings[f"dpad_{name}"] for name in DPAD_CONTROLS},
        )

    def _validate_control_binding(
        self, control: str, payload: dict[str, object]
    ) -> Binding:
        if control in STICK_CONTROLS:
            binding: Binding = StickBinding.model_validate(payload)
        elif control in TRIGGER_CONTROLS:
            if payload.get("source") == "axis":
                binding = AxisTriggerBinding.model_validate(payload)
            else:
                binding = ButtonTriggerBinding.model_validate(payload)
        elif control.startswith("dpad_"):
            if payload.get("source") == "axis":
                binding = AxisDpadBinding.model_validate(payload)
            else:
                binding = ButtonBinding.model_validate(payload)
        else:
            binding = ButtonBinding.model_validate(payload)
        self._validate_capability(binding)
        return binding

    def _validate_capability(self, binding: Binding) -> None:
        if isinstance(binding, StickBinding):
            if binding.axis >= self._descriptor.capabilities.axis_count:
                raise ValueError("axis index is outside detected capability")
        elif isinstance(binding, (AxisTriggerBinding, AxisDpadBinding)):
            if binding.index >= self._descriptor.capabilities.axis_count:
                raise ValueError("axis index is outside detected capability")
            if (
                isinstance(binding, AxisTriggerBinding)
                and binding.correlated_button is not None
                and binding.correlated_button.index
                >= self._descriptor.capabilities.button_count
            ):
                raise ValueError("button index is outside detected capability")
        elif isinstance(binding, (ButtonBinding, ButtonTriggerBinding)) and (
            binding.index >= self._descriptor.capabilities.button_count
        ):
            raise ValueError("button index is outside detected capability")

    def _validate_source_available(self, control: str, binding: Binding) -> None:
        for existing_control, existing in self._bindings.items():
            if existing_control == control or isinstance(existing, UnsupportedBinding):
                continue
            if isinstance(binding, (StickBinding, AxisTriggerBinding)) and isinstance(
                existing, (StickBinding, AxisTriggerBinding)
            ):
                new_index = binding.axis if isinstance(binding, StickBinding) else binding.index
                old_index = existing.axis if isinstance(existing, StickBinding) else existing.index
                if new_index == old_index:
                    raise ValueError(f"duplicate analog axis {new_index}")
            elif isinstance(binding, AxisDpadBinding) and isinstance(
                existing, AxisDpadBinding
            ):
                if binding.index == existing.index and (
                    binding.direction == existing.direction
                    or {control, existing_control}
                    not in (
                        {"dpad_up", "dpad_down"},
                        {"dpad_left", "dpad_right"},
                    )
                ):
                    raise ValueError(f"duplicate dpad axis direction {binding.index}")
            elif isinstance(binding, (ButtonBinding, ButtonTriggerBinding)) and isinstance(
                existing, (ButtonBinding, ButtonTriggerBinding)
            ):
                if binding.index == existing.index:
                    raise ValueError(f"duplicate button {binding.index}")

    def _require_connected(self) -> None:
        if self._state is CalibrationState.DISCONNECTED:
            raise RuntimeError("joystick is disconnected")
