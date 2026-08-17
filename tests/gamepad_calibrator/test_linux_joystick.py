from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.gamepad_calibrator.linux_joystick import (
    JS_EVENT_AXIS,
    JS_EVENT_BUTTON,
    JS_EVENT_INIT,
    JS_EVENT_STRUCT,
    JoystickDescriptor,
    JoystickDiscoveryError,
    JoystickReadError,
    LinuxJoystickReader,
    discover_joysticks,
)
from src.gamepad_calibrator.models import DeviceCapabilities, DeviceIdentity


def _write_identity(
    sys_root: Path,
    js_name: str,
    *,
    vendor: str = "046d",
    product: str = "c216",
    name: str = "Test Pad",
    serial: str | None = "serial-1",
) -> None:
    device = sys_root / js_name / "device"
    (device / "id").mkdir(parents=True)
    (device / "id" / "vendor").write_text(vendor + "\n")
    (device / "id" / "product").write_text(product + "\n")
    (device / "name").write_text(name + "\n")
    if serial is not None:
        (device / "uniq").write_text(serial + "\n")


def test_discovery_sorts_numerically_and_uses_stable_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Would fail if js numbers affect ordering or identity after re-enumeration."""
    dev_root = tmp_path / "dev"
    sys_root = tmp_path / "sys"
    by_id_root = tmp_path / "by-id"
    dev_root.mkdir()
    by_id_root.mkdir()
    for js_name in ("js10", "js2"):
        (dev_root / js_name).touch()
    _write_identity(sys_root, "js10", name="Other Pad", vendor="1234", product="5678", serial=None)
    _write_identity(sys_root, "js2")
    (by_id_root / "usb-Test_Pad-joystick").symlink_to(dev_root / "js2")

    def capabilities(path: Path) -> tuple[DeviceCapabilities, str]:
        return DeviceCapabilities(axis_count=6, button_count=12), (
            "Test Pad" if path.name in {"js1", "js2"} else "Other Pad"
        )

    monkeypatch.setattr("src.gamepad_calibrator.linux_joystick._read_capabilities", capabilities)

    discovered = discover_joysticks(dev_root, sys_root, by_id_root)

    assert [descriptor.path.name for descriptor in discovered] == ["js2", "js10"]
    assert discovered[0].identity == DeviceIdentity(
        vendor_id="046d", product_id="c216", name="Test Pad", serial="serial-1"
    )
    assert discovered[0].capabilities == DeviceCapabilities(axis_count=6, button_count=12)
    assert discovered[0].by_id_path == by_id_root / "usb-Test_Pad-joystick"

    (dev_root / "js2").rename(dev_root / "js1")
    (sys_root / "js2").rename(sys_root / "js1")
    (by_id_root / "usb-Test_Pad-joystick").unlink()
    (by_id_root / "usb-Test_Pad-joystick").symlink_to(dev_root / "js1")
    reenumerated = discover_joysticks(dev_root, sys_root, by_id_root)

    assert reenumerated[0].identity == discovered[0].identity
    assert reenumerated[0].path.name == "js1"
    assert reenumerated[0].by_id_path == by_id_root / "usb-Test_Pad-joystick"


def test_discovery_rejects_incomplete_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Would fail if missing vendor/product/name data becomes a usable descriptor."""
    dev_root = tmp_path / "dev"
    sys_root = tmp_path / "sys"
    by_id_root = tmp_path / "by-id"
    dev_root.mkdir()
    by_id_root.mkdir()
    (dev_root / "js0").touch()
    _write_identity(sys_root, "js0", vendor="", name="")
    monkeypatch.setattr(
        "src.gamepad_calibrator.linux_joystick._read_capabilities",
        lambda path: (DeviceCapabilities(axis_count=2, button_count=2), ""),
    )

    with pytest.raises(JoystickDiscoveryError, match="identity"):
        discover_joysticks(dev_root, sys_root, by_id_root)


def test_reader_decodes_axis_button_and_initial_records_from_nonblocking_fifo(tmp_path: Path) -> None:
    """Would fail if the init bit is not masked or records update the wrong state slot."""
    device_path = tmp_path / "js0"
    os.mkfifo(device_path)
    descriptor = JoystickDescriptor(
        path=device_path,
        identity=DeviceIdentity(vendor_id="046d", product_id="c216", name="Test Pad", serial=None),
        capabilities=DeviceCapabilities(axis_count=2, button_count=3),
        by_id_path=None,
    )
    reader = LinuxJoystickReader(descriptor)
    writer = os.open(device_path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(
            writer,
            b"".join(
                (
                    JS_EVENT_STRUCT.pack(100, -123, JS_EVENT_AXIS, 1),
                    JS_EVENT_STRUCT.pack(110, 1, JS_EVENT_BUTTON, 2),
                    JS_EVENT_STRUCT.pack(120, 456, JS_EVENT_AXIS | JS_EVENT_INIT, 0),
                )
            ),
        )

        batch = reader.drain()

        assert batch.connected is True
        assert batch.axes == (456, -123)
        assert batch.buttons == (0, 0, 1)
        assert [(event.kind, event.number, event.initial) for event in batch.events] == [
            ("axis", 1, False),
            ("button", 2, False),
            ("axis", 0, True),
        ]
    finally:
        os.close(writer)
        reader.close()


def test_reader_marks_eof_disconnected_and_rejects_partial_records(tmp_path: Path) -> None:
    """Would fail if unplug-like EOF is treated as connected or truncated data is decoded."""
    device_path = tmp_path / "js0"
    os.mkfifo(device_path)
    descriptor = JoystickDescriptor(
        path=device_path,
        identity=DeviceIdentity(vendor_id="046d", product_id="c216", name="Test Pad", serial=None),
        capabilities=DeviceCapabilities(axis_count=1, button_count=1),
        by_id_path=None,
    )
    reader = LinuxJoystickReader(descriptor)
    writer = os.open(device_path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(writer, b"bad")
        with pytest.raises(JoystickReadError, match="partial"):
            reader.drain()
    finally:
        os.close(writer)
        reader.close()
