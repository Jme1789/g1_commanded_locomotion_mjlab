"""Behavioral contracts for the pure quick gamepad mapping engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gamepad_calibrator.linux_joystick import (
    EventBatch,
    JoystickDescriptor,
    RawJoystickEvent,
)
from src.gamepad_calibrator.models import (
    REQUIRED_G1_CONTROLS,
    DeviceCapabilities,
    DeviceIdentity,
    validate_profile,
)
from src.gamepad_calibrator.quick_mapping import QuickMappingEngine


def descriptor() -> JoystickDescriptor:
    return JoystickDescriptor(
        path=Path("/dev/input/js0"),
        identity=DeviceIdentity(
            vendor_id="20bc",
            product_id="5500",
            name="Test Pad",
            serial=None,
        ),
        capabilities=DeviceCapabilities(axis_count=8, button_count=16),
        by_id_path=None,
    )


def batch(
    *,
    events: tuple[RawJoystickEvent, ...],
    axes: tuple[int, ...] = (0,) * 8,
    buttons: tuple[int, ...] = (0,) * 16,
    connected: bool = True,
) -> EventBatch:
    return EventBatch(events=events, axes=axes, buttons=buttons, connected=connected)


def button_event(value: int, number: int = 0, time_ms: int = 1, *, initial: bool = False) -> RawJoystickEvent:
    return RawJoystickEvent(
        time_ms=time_ms,
        value=value,
        kind="button",
        number=number,
        initial=initial,
    )


def axis_event(value: int, number: int, time_ms: int = 1) -> RawJoystickEvent:
    return RawJoystickEvent(
        time_ms=time_ms,
        value=value,
        kind="axis",
        number=number,
        initial=False,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def finish_axis_capture(
    engine: QuickMappingEngine,
    clock: FakeClock,
    *,
    events: tuple[RawJoystickEvent, ...],
    axes: tuple[int, ...],
    buttons: tuple[int, ...] = (0,) * 16,
) -> dict:
    engine.observe(batch(events=events, axes=axes, buttons=buttons))
    clock.advance(0.251)
    return engine.observe(batch(events=(), axes=axes, buttons=buttons))


def test_repeated_button_frames_emit_one_press_and_one_release_edge() -> None:
    """Reducing each batch only once prevents repeat frames from duplicating commands."""
    engine = QuickMappingEngine(descriptor())
    engine.arm("a")

    pressed = list((0,) * 16)
    pressed[0] = 1
    first = engine.observe(batch(events=(button_event(1),), buttons=tuple(pressed)))
    repeated = engine.observe(
        batch(
            events=(button_event(1, time_ms=2), button_event(1, time_ms=3)),
            buttons=tuple(pressed),
        )
    )
    released = engine.observe(batch(events=(button_event(0, time_ms=4),)))
    repeated_release = engine.observe(batch(events=(button_event(0, time_ms=5),)))

    assert first["edges"]["on_pressed"] == ["a"]
    assert repeated["edges"]["on_pressed"] == []
    assert released["edges"]["on_released"] == ["a"]
    assert repeated_release["edges"]["on_released"] == []
    assert [item["phase"] for item in repeated["raw"]["transitions"]] == [
        "repeat",
        "repeat",
    ]


def test_arm_ignores_initial_and_already_held_inputs() -> None:
    """A held control at arming must not be mistaken for a newly chosen source."""
    engine = QuickMappingEngine(descriptor())
    held = list((0,) * 16)
    held[2] = 1
    engine.observe(
        batch(
            events=(button_event(1, number=2, initial=True),),
            buttons=tuple(held),
        )
    )
    engine.arm("a")
    engine.observe(batch(events=(button_event(1, number=2, time_ms=2),), buttons=tuple(held)))
    engine.observe(batch(events=(button_event(0, number=2, time_ms=3),)))
    pressed = list((0,) * 16)
    pressed[3] = 1
    result = engine.observe(
        batch(events=(button_event(1, number=3, time_ms=4),), buttons=tuple(pressed))
    )

    assert result["bindings"]["a"] == {"source": "button", "index": 3}


def test_stick_axis_records_index_and_invert_from_expected_positive_motion() -> None:
    """A vertical stick's upward raw motion must yield positive logical motion."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("left_y")
    axes = [0] * 8
    axes[1] = -32768
    result = finish_axis_capture(
        engine,
        clock,
        events=(axis_event(-32768, 1),),
        axes=tuple(axes),
    )

    assert result["bindings"]["left_y"] == {
        "axis": 1,
        "center": 0,
        "min": -32768,
        "max": 32767,
        "invert": True,
        "deadzone": 0.05,
    }


def test_dominant_axis_window_chooses_large_axis_over_small_diagonal_noise() -> None:
    """A small secondary-axis wobble must not steal a deliberate X gesture."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("left_x")
    axes = (24000, 5000, 0, 0, 0, 0, 0, 0)
    collecting = engine.observe(
        batch(
            events=(axis_event(24000, 0), axis_event(5000, 1)),
            axes=axes,
        )
    )

    assert collecting["capture"]["status"] == "collecting"
    clock.advance(0.251)
    result = engine.observe(batch(events=(), axes=axes))

    assert result["bindings"]["left_x"] == {
        "axis": 0,
        "center": 0,
        "min": -32768,
        "max": 32767,
        "invert": False,
        "deadzone": 0.05,
    }
    assert result["capture"] == {
        "status": "captured",
        "control": "left_x",
        "source": "axis",
        "index": 0,
        "direction": "positive",
        "primary_axis": 0,
        "secondary_axis": 1,
    }


def test_similar_diagonal_axes_stay_armed_until_recenter_and_clean_retry() -> None:
    """An ambiguous diagonal gesture must never write a binding or consume retry."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("left_x")
    diagonal = (24000, 20000, 0, 0, 0, 0, 0, 0)
    engine.observe(
        batch(
            events=(axis_event(24000, 0), axis_event(20000, 1)),
            axes=diagonal,
        )
    )
    clock.advance(0.251)
    ambiguous = engine.observe(batch(events=(), axes=diagonal))

    assert ambiguous["bindings"]["left_x"] == {"unsupported": True}
    assert ambiguous["armed_control"] == "left_x"
    assert ambiguous["capture"] == {
        "status": "ambiguous",
        "control": "left_x",
        "source": None,
        "index": None,
        "direction": None,
        "primary_axis": 0,
        "secondary_axis": 1,
    }

    neutral = (0,) * 8
    rearmed = engine.observe(
        batch(
            events=(axis_event(0, 0, 2), axis_event(0, 1, 2)),
            axes=neutral,
        )
    )
    assert rearmed["capture"]["status"] == "armed"
    clean = (26000, 3000, 0, 0, 0, 0, 0, 0)
    engine.observe(
        batch(
            events=(axis_event(26000, 0, 3), axis_event(3000, 1, 3)),
            axes=clean,
        )
    )
    clock.advance(0.251)
    captured = engine.observe(batch(events=(), axes=clean))
    assert captured["bindings"]["left_x"]["axis"] == 0
    assert captured["capture"]["status"] == "captured"


def test_negative_dominant_axis_preserves_expected_positive_logical_direction() -> None:
    """An upward negative raw Y gesture must still record invert=true."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("left_y")
    axes = (2000, -26000, 0, 0, 0, 0, 0, 0)
    engine.observe(
        batch(
            events=(axis_event(2000, 0), axis_event(-26000, 1)),
            axes=axes,
        )
    )
    clock.advance(0.251)
    result = engine.observe(batch(events=(), axes=axes))

    assert result["bindings"]["left_y"]["axis"] == 1
    assert result["bindings"]["left_y"]["invert"] is True
    assert result["capture"]["direction"] == "negative"


def test_button_target_ignores_simultaneous_large_axis_noise() -> None:
    """Axis movement must not compete with a button target's physical press."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("a")
    axes = (28000, 0, 0, 0, 0, 0, 0, 0)
    buttons = (1,) + (0,) * 15
    result = engine.observe(
        batch(
            events=(axis_event(28000, 0), button_event(1, 0)),
            axes=axes,
            buttons=buttons,
        )
    )

    assert result["bindings"]["a"] == {"source": "button", "index": 0}
    assert result["capture"] == {
        "status": "captured",
        "control": "a",
        "source": "button",
        "index": 0,
        "direction": None,
        "primary_axis": None,
        "secondary_axis": None,
    }


def test_empty_batch_finishes_held_axis_window_after_deadline() -> None:
    """A held joystick need not emit another event for the timer to decide."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("right_x")
    held = (0, 0, 27000, 0, 0, 0, 0, 0)
    engine.observe(batch(events=(axis_event(27000, 2),), axes=held))
    clock.advance(0.249)
    waiting = engine.observe(batch(events=(), axes=held))
    assert waiting["capture"]["status"] == "collecting"
    clock.advance(0.001)
    captured = engine.observe(batch(events=(), axes=held))
    assert captured["bindings"]["right_x"]["axis"] == 2


def test_dpad_and_trigger_accept_button_or_axis_sources() -> None:
    """Quick capture keeps valid alternate source types instead of forcing one device layout."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("dpad_up")
    axes = [0] * 8
    axes[6] = -32768
    dpad = finish_axis_capture(
        engine,
        clock,
        events=(axis_event(-32768, 6),),
        axes=tuple(axes),
    )
    engine.arm("lt")
    buttons = [0] * 16
    buttons[4] = 1
    trigger = engine.observe(
        batch(events=(button_event(1, number=4, time_ms=2),), axes=tuple(axes), buttons=tuple(buttons))
    )

    assert dpad["bindings"]["dpad_up"] == {
        "source": "axis",
        "index": 6,
        "direction": "negative",
        "threshold": 0.5,
    }
    assert trigger["bindings"]["lt"] == {"source": "button", "index": 4, "threshold": 0.5}


@pytest.mark.parametrize(
    ("control", "direction", "capture_value", "neutral_values", "threshold_value"),
    [
        ("dpad_up", "up", -32768, (-1, -16383), -16384),
        ("dpad_down", "down", 32767, (1, 16383), 16384),
    ],
)
def test_quick_dpad_preview_and_edges_use_normalized_axis_threshold(
    control: str,
    direction: str,
    capture_value: int,
    neutral_values: tuple[int, int],
    threshold_value: int,
) -> None:
    """D-pad noise must stay neutral while the threshold emits one real press edge."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm(control)
    axes = [0] * 8
    axes[6] = capture_value
    captured = finish_axis_capture(
        engine,
        clock,
        events=(axis_event(capture_value, 6),),
        axes=tuple(axes),
    )
    assert captured["logical"]["dpad"][direction] is True
    assert captured["edges"]["on_pressed"] == [control]

    axes[6] = 0
    centered = engine.observe(
        batch(events=(axis_event(0, 6, time_ms=2),), axes=tuple(axes))
    )
    assert centered["edges"]["on_released"] == [control]

    for time_ms, raw_axis in enumerate(neutral_values, start=3):
        axes[6] = raw_axis
        neutral = engine.observe(
            batch(
                events=(axis_event(raw_axis, 6, time_ms=time_ms),),
                axes=tuple(axes),
            )
        )
        assert neutral["logical"]["dpad"][direction] is False
        assert neutral["edges"]["on_pressed"] == []

    axes[6] = threshold_value
    pressed = engine.observe(
        batch(events=(axis_event(threshold_value, 6, time_ms=5),), axes=tuple(axes))
    )
    assert pressed["logical"]["dpad"][direction] is True
    assert pressed["edges"]["on_pressed"] == [control]

    axes[6] = capture_value
    endpoint = engine.observe(
        batch(events=(axis_event(capture_value, 6, time_ms=6),), axes=tuple(axes))
    )
    assert endpoint["logical"]["dpad"][direction] is True
    assert endpoint["edges"]["on_pressed"] == []


def test_rebinding_displaces_conflicting_button_but_keeps_opposite_dpad_axis() -> None:
    """A physical button has one owner, while opposite directions can share a D-pad axis."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    engine.arm("a")
    buttons = [0] * 16
    buttons[0] = 1
    engine.observe(batch(events=(button_event(1),), buttons=tuple(buttons)))
    engine.observe(batch(events=(button_event(0, time_ms=2),)))
    engine.arm("b")
    rebound = engine.observe(batch(events=(button_event(1, time_ms=3),), buttons=tuple(buttons)))
    engine.observe(batch(events=(button_event(0, time_ms=4),)))
    engine.arm("dpad_up")
    up_axes = [0] * 8
    up_axes[7] = -32768
    finish_axis_capture(
        engine,
        clock,
        events=(axis_event(-32768, 7, 5),),
        axes=tuple(up_axes),
    )
    engine.arm("dpad_down")
    down_axes = [0] * 8
    down_axes[7] = 32767
    result = finish_axis_capture(
        engine,
        clock,
        events=(axis_event(32767, 7, 6),),
        axes=tuple(down_axes),
    )

    assert result["bindings"]["a"] == {"unsupported": True}
    assert rebound["replacement"] is not None
    assert result["bindings"]["dpad_up"]["index"] == 7
    assert result["bindings"]["dpad_down"]["index"] == 7


def test_disconnect_forces_neutral_and_one_release() -> None:
    """Disconnecting a held input produces the single release needed to clear UI state."""
    engine = QuickMappingEngine(descriptor())
    engine.arm("a")
    buttons = [0] * 16
    buttons[0] = 1
    engine.observe(batch(events=(button_event(1),), buttons=tuple(buttons)))
    disconnected = engine.observe(batch(events=(), connected=False))
    repeated = engine.observe(batch(events=(), connected=False))

    assert disconnected["connected"] is False
    assert disconnected["edges"]["pressed"]["a"] is False
    assert disconnected["edges"]["on_released"] == ["a"]
    assert repeated["edges"]["on_released"] == []


def test_build_profile_lists_missing_required_then_validates_capabilities() -> None:
    """A quick draft stays incomplete until every required backend control has a valid source."""
    clock = FakeClock()
    engine = QuickMappingEngine(descriptor(), clock=clock)
    assert engine.snapshot()["missing_required"] == sorted(REQUIRED_G1_CONTROLS)

    for control, event, axes, buttons in (
        ("left_x", axis_event(32767, 0), (32767, 0, 0, 0, 0, 0, 0, 0), (0,) * 16),
        ("left_y", axis_event(-32768, 1), (32767, -32768, 0, 0, 0, 0, 0, 0), (0,) * 16),
        ("right_x", axis_event(32767, 2), (32767, -32768, 32767, 0, 0, 0, 0, 0), (0,) * 16),
        ("lt", axis_event(32767, 3), (32767, -32768, 32767, 32767, 0, 0, 0, 0), (0,) * 16),
        ("rt", axis_event(32767, 4), (32767, -32768, 32767, 32767, 32767, 0, 0, 0), (0,) * 16),
        ("dpad_up", axis_event(-32768, 5), (32767, -32768, 32767, 32767, 32767, -32768, 0, 0), (0,) * 16),
        ("a", button_event(1, 0), (32767, -32768, 32767, 32767, 32767, -32768, 0, 0), (1,) + (0,) * 15),
        ("b", button_event(1, 1), (32767, -32768, 32767, 32767, 32767, -32768, 0, 0), (1, 1) + (0,) * 14),
        ("rb", button_event(1, 2), (32767, -32768, 32767, 32767, 32767, -32768, 0, 0), (1, 1, 1) + (0,) * 13),
    ):
        engine.arm(control)
        if event.kind == "axis":
            finish_axis_capture(
                engine,
                clock,
                events=(event,),
                axes=axes,
                buttons=buttons,
            )
        else:
            engine.observe(batch(events=(event,), axes=axes, buttons=buttons))

    profile = engine.build_profile()
    validate_profile(profile, descriptor().capabilities)
