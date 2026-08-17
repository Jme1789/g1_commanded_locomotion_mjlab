"""Single-owner calibration sessions and bounded event delivery."""

from __future__ import annotations

import copy
import threading
import uuid
from collections import deque
from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass

from pydantic import JsonValue

from .calibration import PUMP_INTERVAL_SECONDS, CalibrationEngine, CalibrationState
from .linux_joystick import (
    EventBatch,
    JoystickDescriptor,
    JoystickReadError,
    LinuxJoystickReader,
)
from .models import DeviceIdentity, GamepadProfile, LogicalState, TemplateProfile
from .process_interlock import BlockingProcess
from .profile_store import ProfileStore
from .quick_mapping import QuickMappingEngine

EVENT_HISTORY_LIMIT = 500


class SessionError(RuntimeError):
    """Structured failures raised by session ownership and safety checks."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message_zh: str,
        field_path: str | None = None,
        details: JsonValue | None = None,
    ) -> None:
        super().__init__(message_zh)
        self.status_code = status_code
        self.code = code
        self.message_zh = message_zh
        self.field_path = field_path
        self.details = details


@dataclass(frozen=True, slots=True)
class SessionEvent:
    sequence_id: int
    event: str
    data: dict[str, JsonValue]


class CalibrationSession:
    """Own one reader and serialize all access to its calibration engine."""

    def __init__(
        self,
        *,
        descriptor: JoystickDescriptor,
        reader: LinuxJoystickReader,
        engine: CalibrationEngine,
        clock: Callable[[], float],
        pump_interval: float = PUMP_INTERVAL_SECONDS,
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self.descriptor = descriptor
        self.engine = engine
        self._reader = reader
        self._clock = clock
        self._pump_interval = pump_interval
        self._engine_lock = threading.RLock()
        self._condition = threading.Condition(self._engine_lock)
        self._history: deque[SessionEvent] = deque(maxlen=EVENT_HISTORY_LIMIT)
        self._next_sequence_id = 1
        self._active_control: str | None = None
        self._stop = threading.Event()
        self._close_lock = threading.Lock()
        self._closed = False
        self._pump_thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._pump,
            name=f"gamepad-calibration-{self.session_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def pump_thread_id(self) -> int | None:
        return self._pump_thread_id

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _pump(self) -> None:
        self._pump_thread_id = threading.get_ident()
        try:
            while not self._stop.is_set():
                try:
                    batch = self._reader.drain()
                    now = self._clock()
                    with self._condition:
                        if self._closed:
                            break
                        self.engine.observe(batch, now)
                        self._append_event_locked("snapshot", self._snapshot_locked(batch))
                    if not batch.connected:
                        break
                except (JoystickReadError, OSError, ValueError) as error:
                    with self._condition:
                        disconnected = EventBatch(
                            events=(),
                            axes=(0,) * self.descriptor.capabilities.axis_count,
                            buttons=(0,) * self.descriptor.capabilities.button_count,
                            connected=False,
                        )
                        self.engine.observe(disconnected, self._clock())
                        data = self._snapshot_locked(disconnected)
                        data["error"] = {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        self._append_event_locked("disconnected", data)
                    break
                if self._stop.wait(self._pump_interval):
                    break
        finally:
            self._reader.close()

    def _snapshot_locked(
        self, batch: EventBatch | None = None
    ) -> dict[str, JsonValue]:
        candidate = self.engine.candidate()
        data: dict[str, JsonValue] = {
            "session_id": self.session_id,
            "state": self.engine.state.value,
            "connected": self.engine.state is not CalibrationState.DISCONNECTED,
            "candidate": asdict(candidate) if candidate is not None else None,
        }
        if batch is not None:
            data["raw"] = {
                "axes": list(batch.axes),
                "buttons": list(batch.buttons),
                "events": [asdict(event) for event in batch.events],
            }
        return data

    def _append_event_locked(
        self, event: str, data: dict[str, JsonValue]
    ) -> None:
        self._history.append(
            SessionEvent(
                sequence_id=self._next_sequence_id,
                event=event,
                data=copy.deepcopy(data),
            )
        )
        self._next_sequence_id += 1
        self._condition.notify_all()

    def snapshot(self) -> dict[str, JsonValue]:
        with self._engine_lock:
            return copy.deepcopy(self._snapshot_locked())

    def wait_events(
        self,
        after_id: int,
        timeout: float,
    ) -> tuple[SessionEvent, ...]:
        def available() -> bool:
            return bool(self._history and self._history[-1].sequence_id > after_id)

        with self._condition:
            if not available():
                self._condition.wait_for(available, timeout=max(0.0, timeout))
            return tuple(
                SessionEvent(event.sequence_id, event.event, copy.deepcopy(event.data))
                for event in self._history
                if event.sequence_id > after_id
            )

    def _require_connected_locked(self) -> None:
        if self._closed:
            raise SessionError(
                status_code=410,
                code="session_closed",
                message_zh="校准会话已关闭。",
            )
        if self.engine.state is CalibrationState.DISCONNECTED:
            raise SessionError(
                status_code=410,
                code="session_disconnected",
                message_zh="游戏手柄已断开，当前校准会话只能查看或关闭。",
            )

    def _record_command_locked(self) -> dict[str, JsonValue]:
        snapshot = self._snapshot_locked()
        self._append_event_locked("state", snapshot)
        return copy.deepcopy(snapshot)

    def active_control(self) -> str | None:
        with self._engine_lock:
            self._require_connected_locked()
            return self._active_control

    def begin_step(self, control: str) -> dict[str, JsonValue]:
        with self._condition:
            self._require_connected_locked()
            self.engine.begin_step(control)
            self._active_control = control
            return self._record_command_locked()

    def confirm(
        self,
        control: str,
        binding_override: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        with self._condition:
            self._require_connected_locked()
            if self._active_control != control:
                raise SessionError(
                    status_code=422,
                    code="invalid_control",
                    message_zh="确认的逻辑控制项与当前采集步骤不一致。",
                    field_path="control",
                )
            self.engine.confirm(binding_override)
            self._active_control = None
            return self._record_command_locked()

    def redo(self, control: str) -> dict[str, JsonValue]:
        with self._condition:
            self._require_connected_locked()
            self.engine.redo(control)
            self._active_control = control
            return self._record_command_locked()

    def mark_unsupported(self, control: str) -> dict[str, JsonValue]:
        with self._condition:
            self._require_connected_locked()
            self.engine.mark_unsupported(control)
            self._active_control = None
            return self._record_command_locked()

    def preview(self) -> LogicalState:
        with self._engine_lock:
            self._require_connected_locked()
            return self.engine.preview()

    def build_profile(self, preview_confirmations: Collection[str]) -> GamepadProfile:
        with self._engine_lock:
            self._require_connected_locked()
            return self.engine.build_profile(preview_confirmations)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._condition:
                self._closed = True
                self._stop.set()
                self._condition.notify_all()
            self._thread.join()


class QuickMappingSession:
    """Own one reader and expose the automatic mapping snapshot lifecycle."""

    def __init__(
        self,
        *,
        descriptor: JoystickDescriptor,
        device_count: int,
        reader: LinuxJoystickReader,
        engine: QuickMappingEngine,
        clock: Callable[[], float],
        pump_interval: float = PUMP_INTERVAL_SECONDS,
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self.descriptor = descriptor
        self.device_count = device_count
        self.engine = engine
        self._reader = reader
        self._clock = clock
        self._pump_interval = pump_interval
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._history: deque[SessionEvent] = deque(maxlen=EVENT_HISTORY_LIMIT)
        self._next_sequence_id = 1
        self._stop = threading.Event()
        self._close_lock = threading.Lock()
        self._closed = False
        self._pump_thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._pump,
            name=f"gamepad-quick-map-{self.session_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def pump_thread_id(self) -> int | None:
        return self._pump_thread_id

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _snapshot_locked(
        self,
        engine_snapshot: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        source = engine_snapshot if engine_snapshot is not None else self.engine.snapshot()
        data = copy.deepcopy(source)
        raw = data.get("raw")
        if isinstance(raw, dict):
            transitions = raw.get("transitions")
            if isinstance(transitions, list):
                for transition in transitions:
                    if isinstance(transition, dict) and "value" in transition:
                        transition["new_value"] = transition.pop("value")
        data["session_id"] = self.session_id
        if self._closed:
            data["state"] = "closed"
        elif data["connected"]:
            data["state"] = "monitoring"
        else:
            data["state"] = "disconnected"
        return data

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            return self._snapshot_locked()

    def _append_event_locked(self, event: str, data: dict[str, JsonValue]) -> None:
        self._history.append(
            SessionEvent(
                sequence_id=self._next_sequence_id,
                event=event,
                data=copy.deepcopy(data),
            )
        )
        self._next_sequence_id += 1
        self._condition.notify_all()

    def wait_events(self, after_id: int, timeout: float) -> tuple[SessionEvent, ...]:
        def available() -> bool:
            return bool(self._history and self._history[-1].sequence_id > after_id)

        with self._condition:
            if not available():
                self._condition.wait_for(available, timeout=max(0.0, timeout))
            return tuple(
                SessionEvent(event.sequence_id, event.event, copy.deepcopy(event.data))
                for event in self._history
                if event.sequence_id > after_id
            )

    def _require_mutable_locked(self) -> None:
        if self._closed:
            raise SessionError(
                status_code=410,
                code="session_closed",
                message_zh="快速映射会话已关闭。",
            )
        if not bool(self.engine.snapshot()["connected"]):
            raise SessionError(
                status_code=410,
                code="session_disconnected",
                message_zh="游戏手柄已断开，当前快速映射会话只能查看或关闭。",
            )

    def arm(self, control: str) -> dict[str, JsonValue]:
        with self._condition:
            self._require_mutable_locked()
            self.engine.arm(control)
            snapshot = self._snapshot_locked()
            self._append_event_locked("state", snapshot)
            return copy.deepcopy(snapshot)

    def build_profile(self) -> GamepadProfile:
        with self._lock:
            self._require_mutable_locked()
            return self.engine.build_profile()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._condition:
                self._closed = True
                self._stop.set()
                self._condition.notify_all()
            self._thread.join()

    def _pump(self) -> None:
        self._pump_thread_id = threading.get_ident()
        try:
            while not self._stop.is_set():
                try:
                    batch = self._reader.drain()
                    with self._condition:
                        if self._closed:
                            break
                        before = self.engine.snapshot()["bindings"]
                        observed = self.engine.observe(batch)
                        snapshot = self._snapshot_locked(observed)
                        if not batch.connected:
                            event = "disconnected"
                        elif before != snapshot["bindings"]:
                            event = "binding"
                        else:
                            event = "snapshot"
                        self._append_event_locked(event, snapshot)
                        if not batch.connected:
                            break
                except (JoystickReadError, OSError, ValueError) as error:
                    with self._condition:
                        disconnected = EventBatch(
                            events=(),
                            axes=(0,) * self.descriptor.capabilities.axis_count,
                            buttons=(0,) * self.descriptor.capabilities.button_count,
                            connected=False,
                        )
                        observed = self.engine.observe(disconnected)
                        data = self._snapshot_locked(observed)
                        data["error"] = {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        self._append_event_locked("disconnected", data)
                    break
                if self._stop.wait(self._pump_interval):
                    break
        finally:
            self._reader.close()


class SessionManager:
    """Enforce the single-session, exact-discovery, process-safe contract."""

    def __init__(
        self,
        *,
        device_discovery: Callable[[], tuple[JoystickDescriptor, ...]],
        interlock_probe: Callable[[], tuple[BlockingProcess, ...]],
        reader_factory: Callable[[JoystickDescriptor], LinuxJoystickReader],
        clock: Callable[[], float],
        profile_store: ProfileStore,
        pump_interval: float = PUMP_INTERVAL_SECONDS,
    ) -> None:
        self.device_discovery = device_discovery
        self.interlock_probe = interlock_probe
        self.reader_factory = reader_factory
        self.clock = clock
        self.profile_store = profile_store
        self.pump_interval = pump_interval
        self._lock = threading.Lock()
        self._active: CalibrationSession | QuickMappingSession | None = None

    @staticmethod
    def _raise_for_blockers(blockers: tuple[BlockingProcess, ...]) -> None:
        if not blockers:
            return
        raise SessionError(
            status_code=409,
            code="process_conflict",
            message_zh="检测到正在运行的仿真或控制进程，请先手动停止。",
            details={
                "processes": [
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "argv": list(process.argv),
                    }
                    for process in blockers
                ]
            },
        )

    def ensure_interlock(self) -> None:
        self._raise_for_blockers(self.interlock_probe())

    def _require_empty_locked(self) -> None:
        if self._active is not None:
            raise SessionError(
                status_code=409,
                code="session_conflict",
                message_zh="已有手柄会话正在运行，请先关闭当前会话。",
            )

    def _open_reader(
        self,
        descriptor: JoystickDescriptor,
        *,
        field_path: str = "device",
        message_zh: str = "游戏手柄已不可用，请刷新后重试。",
    ) -> LinuxJoystickReader:
        try:
            return self.reader_factory(descriptor)
        except OSError as error:
            raise SessionError(
                status_code=409,
                code="device_unavailable",
                message_zh=message_zh,
                field_path=field_path,
            ) from error

    def create(
        self, device_path: str, template_id: str | None = None
    ) -> CalibrationSession:
        with self._lock:
            self.ensure_interlock()
            self._require_empty_locked()
            matches = tuple(
                descriptor
                for descriptor in self.device_discovery()
                if str(descriptor.path) == device_path
            )
            if len(matches) != 1:
                raise SessionError(
                    status_code=404,
                    code="device_not_found",
                    message_zh="未找到所选游戏手柄，请刷新设备列表。",
                    field_path="device_path",
                )
            descriptor = matches[0]
            engine = CalibrationEngine(descriptor, started_at=self.clock())
            if template_id is not None:
                materialized = self.profile_store.materialize_template(
                    template_id,
                    descriptor.identity,
                    descriptor.capabilities,
                )
                engine.apply_template(
                    TemplateProfile(
                        schema_version=materialized.schema_version,
                        template_name=template_id,
                        sticks=materialized.sticks,
                        triggers=materialized.triggers,
                        buttons=materialized.buttons,
                        dpad=materialized.dpad,
                    )
                )
            reader = self._open_reader(
                descriptor,
                field_path="device_path",
                message_zh="所选游戏手柄已不可用，请刷新设备列表后重试。",
            )
            session = CalibrationSession(
                descriptor=descriptor,
                reader=reader,
                engine=engine,
                clock=self.clock,
                pump_interval=self.pump_interval,
            )
            self._active = session
            return session

    def create_quick(
        self,
        expected_device: DeviceIdentity | None = None,
    ) -> QuickMappingSession:
        with self._lock:
            self.ensure_interlock()
            self._require_empty_locked()
            descriptors = self.device_discovery()
            if not descriptors:
                raise SessionError(
                    status_code=404,
                    code="device_not_found",
                    message_zh="未检测到可用游戏手柄。",
                )
            descriptor = descriptors[0]
            if expected_device is not None:
                matches = tuple(
                    item
                    for item in descriptors
                    if item.identity == expected_device
                )
                if len(matches) != 1:
                    raise SessionError(
                        status_code=409,
                        code="device_identity_mismatch",
                        message_zh="重新连接的手柄身份与当前映射会话不一致。",
                        field_path="expected_device",
                    )
                descriptor = matches[0]
            reader = self._open_reader(descriptor)
            session = QuickMappingSession(
                descriptor=descriptor,
                device_count=len(descriptors),
                reader=reader,
                engine=QuickMappingEngine(descriptor, clock=self.clock),
                clock=self.clock,
                pump_interval=self.pump_interval,
            )
            self._active = session
            return session

    def get_quick(self, session_id: str) -> QuickMappingSession:
        with self._lock:
            if (
                not isinstance(self._active, QuickMappingSession)
                or self._active.session_id != session_id
            ):
                raise SessionError(
                    status_code=404,
                    code="session_not_found",
                    message_zh="快速映射会话不存在。",
                    field_path="session_id",
                )
            return self._active

    def get(self, session_id: str) -> CalibrationSession:
        with self._lock:
            if (
                not isinstance(self._active, CalibrationSession)
                or self._active.session_id != session_id
            ):
                raise SessionError(
                    status_code=404,
                    code="session_not_found",
                    message_zh="校准会话不存在。",
                    field_path="session_id",
                )
            return self._active

    def close(self, session_id: str) -> None:
        with self._lock:
            if self._active is None or self._active.session_id != session_id:
                raise SessionError(
                    status_code=404,
                    code="session_not_found",
                    message_zh="校准会话不存在。",
                    field_path="session_id",
                )
            session = self._active
            try:
                session.close()
            finally:
                self._active = None

    def close_active(self) -> None:
        with self._lock:
            session = self._active
            if session is not None:
                try:
                    session.close()
                finally:
                    self._active = None
