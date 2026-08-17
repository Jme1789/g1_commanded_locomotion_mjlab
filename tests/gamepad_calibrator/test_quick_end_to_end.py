"""End-to-end acceptance coverage for quick gamepad profile activation."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.gamepad_calibrator.linux_joystick import (
    EventBatch,
    JoystickDescriptor,
    RawJoystickEvent,
)
from src.gamepad_calibrator.models import (
    DeviceCapabilities,
    DeviceIdentity,
    validate_profile,
)
from src.gamepad_calibrator.normalization import normalize_profile
from src.gamepad_calibrator.profile_store import ProfileStore
from src.gamepad_calibrator.quick_mapping import QuickMappingEngine


class FakeClock:
    """Deterministically advance the dominant-axis capture window."""

    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class QuickInputDriver:
    """Drive coherent raw input frames through the real quick mapper."""

    def __init__(self, engine: QuickMappingEngine, clock: FakeClock) -> None:
        self.engine = engine
        self.clock = clock
        self.axes = [0] * engine.descriptor.capabilities.axis_count
        self.buttons = [0] * engine.descriptor.capabilities.button_count
        self.time_ms = 1

    def move_axis(self, control: str, index: int, value: int) -> None:
        self.engine.arm(control)
        old = self.axes[index]
        self.axes[index] = value
        event = RawJoystickEvent(
            time_ms=self.time_ms,
            value=value,
            kind="axis",
            number=index,
            initial=False,
        )
        self.time_ms += 1
        self.engine.observe(
            EventBatch(
                events=(event,),
                axes=tuple(self.axes),
                buttons=tuple(self.buttons),
                connected=True,
            )
        )
        self.clock.advance(0.251)
        self.engine.observe(
            EventBatch(
                events=(),
                axes=tuple(self.axes),
                buttons=tuple(self.buttons),
                connected=True,
            )
        )
        assert old != value

    def press_button(self, control: str, index: int) -> None:
        self.engine.arm(control)
        self.buttons[index] = 1
        event = RawJoystickEvent(
            time_ms=self.time_ms,
            value=1,
            kind="button",
            number=index,
            initial=False,
        )
        self.time_ms += 1
        self.engine.observe(
            EventBatch(
                events=(event,),
                axes=tuple(self.axes),
                buttons=tuple(self.buttons),
                connected=True,
            )
        )

    def observe_button(self, index: int, value: int) -> dict[str, object]:
        self.buttons[index] = value
        event = RawJoystickEvent(
            time_ms=self.time_ms,
            value=value,
            kind="button",
            number=index,
            initial=False,
        )
        self.time_ms += 1
        return self.engine.observe(
            EventBatch(
                events=(event,),
                axes=tuple(self.axes),
                buttons=tuple(self.buttons),
                connected=True,
            )
        )


def _descriptor() -> JoystickDescriptor:
    return JoystickDescriptor(
        path=Path("/dev/input/js0"),
        identity=DeviceIdentity(
            vendor_id="20bc",
            product_id="5500",
            name="Quick Acceptance Pad",
            serial="acceptance-1",
        ),
        capabilities=DeviceCapabilities(axis_count=4, button_count=5),
        by_id_path=None,
    )


def test_quick_mapping_profile_activation_preserves_unique_bindings_and_edges(
    tmp_path: Path,
) -> None:
    """Wrong bindings, persistence, activation, or repeat-edge handling must fail."""
    clock = FakeClock()
    engine = QuickMappingEngine(_descriptor(), clock=clock)
    driver = QuickInputDriver(engine, clock)

    driver.move_axis("left_x", 0, 32767)
    driver.move_axis("left_y", 1, -32768)
    driver.move_axis("right_x", 2, 32767)
    driver.move_axis("dpad_up", 3, -32768)
    driver.press_button("lt", 0)
    driver.press_button("rt", 1)
    driver.press_button("a", 2)
    driver.press_button("b", 3)
    driver.press_button("rb", 4)

    assert engine.snapshot()["missing_required"] == []
    repeated = driver.observe_button(2, 1)
    released = driver.observe_button(2, 0)
    assert repeated["edges"] == {
        "pressed": {
            "lt": True,
            "rt": True,
            "dpad_up": True,
            "dpad_down": False,
            "dpad_left": False,
            "dpad_right": False,
            "a": True,
            "b": True,
            "x": False,
            "y": False,
            "lb": False,
            "rb": True,
            "start": False,
            "back": False,
            "left_stick": False,
            "right_stick": False,
        },
        "on_pressed": [],
        "on_released": [],
        "combos": {"lt_up": True, "rt_a": True, "lb": False, "rb": True},
    }
    assert released["edges"]["on_released"] == ["a"]

    profile = engine.build_profile()
    validate_profile(profile, engine.descriptor.capabilities)
    store = ProfileStore(tmp_path / "gamepads")
    saved = store.save(profile)
    exported = store.export_yaml(saved.profile_id)
    imported = store.import_yaml(exported)
    selection = store.activate(imported.profile_id)

    assert selection.profile == f"profiles/{saved.profile_id}.yaml"
    assert yaml.safe_load((store.gamepad_root / "active.yaml").read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "profile": f"profiles/{saved.profile_id}.yaml",
        "device": {
            "vendor_id": "20bc",
            "product_id": "5500",
            "name": "Quick Acceptance Pad",
            "serial": "acceptance-1",
        },
    }

    exported_payload = yaml.safe_load(exported)
    assert exported_payload["sticks"] == {
        "left_x": {
            "axis": 0,
            "center": 0,
            "min": -32768,
            "max": 32767,
            "invert": False,
            "deadzone": 0.05,
        },
        "left_y": {
            "axis": 1,
            "center": 0,
            "min": -32768,
            "max": 32767,
            "invert": True,
            "deadzone": 0.05,
        },
        "right_x": {
            "axis": 2,
            "center": 0,
            "min": -32768,
            "max": 32767,
            "invert": False,
            "deadzone": 0.05,
        },
        "right_y": {"unsupported": True},
    }
    assert exported_payload["triggers"] == {
        "lt": {"source": "button", "index": 0, "threshold": 0.5},
        "rt": {"source": "button", "index": 1, "threshold": 0.5},
    }
    assert exported_payload["dpad"]["up"] == {
        "source": "axis",
        "index": 3,
        "direction": "negative",
        "threshold": 0.5,
    }
    assert not ({"session", "raw", "edge", "edges", "history"} & set(exported_payload))

    reloaded = store.load(saved.profile_id).profile
    logical = normalize_profile(
        reloaded,
        axes=(32767, -32768, 32767, -32768),
        buttons=(1, 1, 1, 1, 1),
    )
    assert logical.sticks["left_x"] > 0
    assert logical.sticks["left_y"] > 0
    assert logical.dpad["up"] is True
    assert logical.triggers["lt"] == 1.0
    assert logical.triggers["rt"] == 1.0
    assert logical.buttons["a"] is True
    assert logical.buttons["b"] is True
    assert logical.buttons["rb"] is True
