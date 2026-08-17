"""Read-only Linux joystick discovery and event decoding."""

from __future__ import annotations

import errno
import fcntl
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .models import DeviceCapabilities, DeviceIdentity

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_STRUCT = struct.Struct("<IhBB")

_JSIOCGAXES = 0x80016A11
_JSIOCGBUTTONS = 0x80016A12
_JSIOCGNAME_LENGTH = 128
_JSIOCGNAME = 0x80006A13 | (_JSIOCGNAME_LENGTH << 16)


class JoystickDiscoveryError(RuntimeError):
    """A joystick cannot be represented as a stable calibration target."""


class JoystickReadError(RuntimeError):
    """The kernel joystick stream returned invalid or incomplete event data."""


@dataclass(frozen=True, slots=True)
class RawJoystickEvent:
    time_ms: int
    value: int
    kind: Literal["axis", "button"]
    number: int
    initial: bool


@dataclass(frozen=True, slots=True)
class JoystickDescriptor:
    path: Path
    identity: DeviceIdentity
    capabilities: DeviceCapabilities
    by_id_path: Path | None


@dataclass(frozen=True, slots=True)
class EventBatch:
    events: tuple[RawJoystickEvent, ...]
    axes: tuple[int, ...]
    buttons: tuple[int, ...]
    connected: bool


def _read_capabilities(path: Path) -> tuple[DeviceCapabilities, str]:
    """Read joystick counts and the kernel-reported name through Linux ioctls."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        axes = bytearray(1)
        buttons = bytearray(1)
        name = bytearray(_JSIOCGNAME_LENGTH)
        fcntl.ioctl(fd, _JSIOCGAXES, axes, True)
        fcntl.ioctl(fd, _JSIOCGBUTTONS, buttons, True)
        fcntl.ioctl(fd, _JSIOCGNAME, name, True)
    finally:
        os.close(fd)
    return (
        DeviceCapabilities(axis_count=axes[0], button_count=buttons[0]),
        name.partition(b"\0")[0].decode(errors="replace"),
    )


def _read_required(path: Path, field: str) -> str:
    try:
        value = path.read_text().strip()
    except OSError as error:
        raise JoystickDiscoveryError(f"incomplete identity: cannot read {field}") from error
    if not value:
        raise JoystickDiscoveryError(f"incomplete identity: missing {field}")
    return value


def _read_optional(path: Path) -> str | None:
    try:
        value = path.read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise JoystickDiscoveryError("incomplete identity: cannot read serial") from error
    return value or None


def _by_id_path(path: Path, by_id_root: Path) -> Path | None:
    try:
        candidates = sorted(by_id_root.iterdir())
    except FileNotFoundError:
        return None
    except OSError as error:
        raise JoystickDiscoveryError(f"cannot read by-id directory {by_id_root}") from error
    resolved_path = path.resolve(strict=False)
    for candidate in candidates:
        try:
            if candidate.resolve(strict=False) == resolved_path:
                return candidate
        except OSError:
            continue
    return None


def discover_joysticks(
    dev_root: Path = Path("/dev/input"),
    sys_class_root: Path = Path("/sys/class/input"),
    by_id_root: Path = Path("/dev/input/by-id"),
) -> tuple[JoystickDescriptor, ...]:
    """Discover joystick devices without making their jsN path part of identity."""
    try:
        paths = [path for path in dev_root.iterdir() if path.name.startswith("js") and path.name[2:].isdigit()]
    except OSError as error:
        raise JoystickDiscoveryError(f"cannot read joystick directory {dev_root}") from error

    descriptors = []
    for path in sorted(paths, key=lambda item: int(item.name[2:])):
        device = sys_class_root / path.name / "device"
        vendor = _read_required(device / "id" / "vendor", "vendor id").lower()
        product = _read_required(device / "id" / "product", "product id").lower()
        name = _read_required(device / "name", "name")
        serial = _read_optional(device / "uniq")
        capabilities, kernel_name = _read_capabilities(path)
        if not kernel_name:
            raise JoystickDiscoveryError("incomplete identity: missing kernel name")
        if kernel_name != name:
            raise JoystickDiscoveryError(
                f"kernel/sysfs joystick name mismatch for {path}: {kernel_name!r} != {name!r}"
            )
        try:
            identity = DeviceIdentity(vendor_id=vendor, product_id=product, name=name, serial=serial)
        except ValueError as error:
            raise JoystickDiscoveryError(f"incomplete identity for {path}") from error
        descriptors.append(
            JoystickDescriptor(
                path=path,
                identity=identity,
                capabilities=capabilities,
                by_id_path=_by_id_path(path, by_id_root),
            )
        )
    return tuple(descriptors)


class LinuxJoystickReader:
    """Maintain a coherent current-state snapshot from a nonblocking joystick fd."""

    def __init__(self, descriptor: JoystickDescriptor):
        self._descriptor = descriptor
        self._fd = os.open(descriptor.path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        self._axes = [0] * descriptor.capabilities.axis_count
        self._buttons = [0] * descriptor.capabilities.button_count

    def _batch(self, events: list[RawJoystickEvent], connected: bool) -> EventBatch:
        return EventBatch(tuple(events), tuple(self._axes), tuple(self._buttons), connected)

    def drain(self) -> EventBatch:
        if self._fd is None:
            raise JoystickReadError("joystick reader is closed")
        events: list[RawJoystickEvent] = []
        while True:
            try:
                payload = os.read(self._fd, JS_EVENT_STRUCT.size * 64)
            except BlockingIOError:
                return self._batch(events, connected=True)
            except OSError as error:
                if error.errno in (errno.ENODEV, errno.EIO):
                    return self._batch(events, connected=False)
                raise JoystickReadError(f"cannot read joystick {self._descriptor.path}") from error
            if not payload:
                return self._batch(events, connected=False)
            if len(payload) % JS_EVENT_STRUCT.size:
                raise JoystickReadError("partial joystick event record")
            for time_ms, value, event_type, number in JS_EVENT_STRUCT.iter_unpack(payload):
                initial = bool(event_type & JS_EVENT_INIT)
                event_type &= ~JS_EVENT_INIT
                if event_type == JS_EVENT_AXIS:
                    kind: Literal["axis", "button"] = "axis"
                    state = self._axes
                elif event_type == JS_EVENT_BUTTON:
                    kind = "button"
                    state = self._buttons
                else:
                    raise JoystickReadError(f"unsupported joystick event type {event_type}")
                if number >= len(state):
                    raise JoystickReadError(f"{kind} index {number} exceeds detected capability")
                state[number] = value
                events.append(RawJoystickEvent(time_ms, value, kind, number, initial))

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
