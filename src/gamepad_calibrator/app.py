"""Versioned loopback service contract for standalone gamepad calibration."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api_errors import (
    ApiError,
    engine_api_error,
    profile_api_error,
    request_api_error,
)
from .api_models import (
    ActivationResponse,
    DeviceResponse,
    DevicesResponse,
    HealthResponse,
    ImportResponse,
    ProfilesResponse,
    QuickSessionCreateRequest,
    QuickSessionResponse,
    SaveResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionSaveRequest,
    StepConfirmRequest,
    StoredProfileResponse,
    TemplateResponse,
)
from .linux_joystick import (
    JoystickDescriptor,
    JoystickDiscoveryError,
    LinuxJoystickReader,
    discover_joysticks,
)
from .models import LogicalState, validate_profile
from .process_interlock import BlockingProcess, find_blocking_processes
from .profile_store import ProfileError, ProfileStore
from .sessions import (
    CalibrationSession,
    QuickMappingSession,
    SessionError,
    SessionManager,
)

API_VERSION = 1
YAML_MEDIA_TYPE = "application/yaml"
STATIC_DIRECTORY = Path(__file__).with_name("static")
SECURITY_HEADERS = {
    "Cache-Control": "no-cache",
    "Content-Security-Policy": (
        "default-src 'self'; connect-src 'self'; img-src 'self' blob:; "
        "style-src 'self'; script-src 'self'; object-src 'none'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


@dataclass
class CalibratorServices:
    project_root: Path
    device_discovery: Callable[[], tuple[JoystickDescriptor, ...]]
    interlock_probe: Callable[[], tuple[BlockingProcess, ...]]
    reader_factory: Callable[[JoystickDescriptor], LinuxJoystickReader]
    clock: Callable[[], float]
    profile_store: ProfileStore
    session_manager: SessionManager

    @classmethod
    def default(cls) -> CalibratorServices:
        project_root = Path(__file__).resolve().parents[2]
        profile_store = ProfileStore(
            project_root / "simulate" / "config" / "gamepads"
        )
        session_manager = SessionManager(
            device_discovery=discover_joysticks,
            interlock_probe=find_blocking_processes,
            reader_factory=LinuxJoystickReader,
            clock=time.monotonic,
            profile_store=profile_store,
        )
        return cls(
            project_root=project_root,
            device_discovery=discover_joysticks,
            interlock_probe=find_blocking_processes,
            reader_factory=LinuxJoystickReader,
            clock=time.monotonic,
            profile_store=profile_store,
            session_manager=session_manager,
        )


def _device_response(descriptor: JoystickDescriptor) -> DeviceResponse:
    return DeviceResponse(
        device_path=str(descriptor.path),
        by_id_path=(
            str(descriptor.by_id_path) if descriptor.by_id_path is not None else None
        ),
        identity=descriptor.identity,
        capabilities=descriptor.capabilities,
    )


def _session_response(
    session: CalibrationSession, *, include_device: bool = False
) -> SessionResponse:
    snapshot = session.snapshot()
    return SessionResponse(
        session_id=session.session_id,
        state=str(snapshot["state"]),
        connected=bool(snapshot["connected"]),
        candidate=snapshot["candidate"],
        device=_device_response(session.descriptor) if include_device else None,
    )


def _quick_session_response(
    session: QuickMappingSession,
) -> QuickSessionResponse:
    payload = {
        **session.snapshot(),
        "device_count": session.device_count,
        "device": _device_response(session.descriptor).model_dump(mode="json"),
    }
    return QuickSessionResponse.model_validate(payload)


def _json_error(error: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=error.envelope().model_dump(mode="json"),
    )


def _get_session(services: CalibratorServices, session_id: str) -> CalibrationSession:
    return services.session_manager.get(session_id)


def _raise_engine(error: Exception, operation: str) -> None:
    if isinstance(error, SessionError):
        raise error
    raise engine_api_error(error, operation) from error


async def _generate_session_events(
    session: CalibrationSession | QuickMappingSession,
    after_id: int,
) -> AsyncIterator[str]:
    cursor = after_id
    while True:
        events = await asyncio.to_thread(session.wait_events, cursor, 1.0)
        if not events:
            yield ": keepalive\n\n"
            if session.is_alive:
                continue
            events = await asyncio.to_thread(session.wait_events, cursor, 0.0)
            if not events:
                return
        for event in events:
            data = json.dumps(
                event.data,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield (
                f"id: {event.sequence_id}\n"
                f"event: {event.event}\n"
                f"data: {data}\n\n"
            )
            cursor = event.sequence_id
        if not session.is_alive:
            return


def create_app(services: CalibratorServices | None = None) -> FastAPI:
    service_container = services or CalibratorServices.default()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service_container.session_manager.close_active()

    app = FastAPI(
        title="G1 Gamepad Calibrator",
        version=str(API_VERSION),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.services = service_container

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.get("/", include_in_schema=False)
    def static_index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html", media_type="text/html")

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, error: ApiError) -> JSONResponse:
        return _json_error(error)

    @app.exception_handler(SessionError)
    async def handle_session_error(
        _request: Request, error: SessionError
    ) -> JSONResponse:
        return _json_error(ApiError.from_session(error))

    @app.exception_handler(ProfileError)
    async def handle_profile_error(
        _request: Request, error: ProfileError
    ) -> JSONResponse:
        return _json_error(profile_api_error(error))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        failures = error.errors()
        location = failures[0].get("loc", ()) if failures else ()
        field_path = ".".join(
            str(part) for part in location if part not in {"body", "path", "query"}
        )
        return _json_error(request_api_error(field_path or None))

    @app.exception_handler(JoystickDiscoveryError)
    async def handle_discovery_error(
        _request: Request, error: JoystickDiscoveryError
    ) -> JSONResponse:
        return _json_error(
            ApiError(
                status_code=503,
                code="device_discovery_failed",
                message_zh="无法读取本机游戏手柄设备。",
                details={"reason": str(error)},
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        if error.status_code == 404:
            api_error = ApiError(
                status_code=404,
                code="route_not_found",
                message_zh="请求的接口不存在。",
            )
        else:
            api_error = ApiError(
                status_code=error.status_code,
                code="http_error",
                message_zh="请求无法处理。",
            )
        return _json_error(api_error)

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", api_version=API_VERSION)

    @app.get("/api/v1/devices", response_model=DevicesResponse)
    def devices() -> DevicesResponse:
        try:
            discovered = service_container.device_discovery()
        except JoystickDiscoveryError as error:
            raise ApiError(
                status_code=503,
                code="device_discovery_failed",
                message_zh="无法读取本机游戏手柄设备。",
                details={"reason": str(error)},
            ) from error
        return DevicesResponse(
            devices=[_device_response(descriptor) for descriptor in discovered]
        )

    @app.get("/api/v1/profiles", response_model=ProfilesResponse)
    def profiles() -> ProfilesResponse:
        return ProfilesResponse(
            profiles=[
                StoredProfileResponse(
                    profile_id=stored.profile_id,
                    profile=stored.profile,
                )
                for stored in service_container.profile_store.list_profiles()
            ],
            templates=[
                TemplateResponse(
                    template_id=record.template_id,
                    **record.template.model_dump(mode="python"),
                )
                for record in service_container.profile_store.list_template_records()
            ],
        )

    @app.post(
        "/api/v1/sessions", response_model=SessionResponse, status_code=201
    )
    def create_session(body: SessionCreateRequest) -> SessionResponse:
        session = service_container.session_manager.create(
            body.device_path, template_id=body.template_id
        )
        return _session_response(session, include_device=True)

    @app.get("/api/v1/sessions/{session_id}/events")
    async def session_events(
        request: Request,
        session_id: str,
        after_id: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        session = _get_session(service_container, session_id)
        header_id = request.headers.get("last-event-id")
        if header_id is not None and after_id == 0:
            try:
                after_id = int(header_id)
            except ValueError as error:
                raise request_api_error("Last-Event-ID") from error
            if after_id < 0:
                raise request_api_error("Last-Event-ID")
        return StreamingResponse(
            _generate_session_events(session, after_id), media_type="text/event-stream"
        )

    @app.post(
        "/api/v1/sessions/{session_id}/steps/{control}",
        response_model=SessionResponse,
    )
    def begin_step(session_id: str, control: str) -> SessionResponse:
        session = _get_session(service_container, session_id)
        try:
            session.begin_step(control)
        except (SessionError, RuntimeError, ValueError) as error:
            _raise_engine(error, "control")
        return _session_response(session)

    @app.post(
        "/api/v1/sessions/{session_id}/steps/{control}/confirm",
        response_model=SessionResponse,
    )
    def confirm_step(
        session_id: str, control: str, body: StepConfirmRequest
    ) -> SessionResponse:
        session = _get_session(service_container, session_id)
        try:
            session.confirm(control, body.binding_override)
        except (SessionError, RuntimeError, ValueError, ValidationError) as error:
            _raise_engine(error, "binding")
        return _session_response(session)

    @app.post(
        "/api/v1/sessions/{session_id}/steps/{control}/redo",
        response_model=SessionResponse,
    )
    def redo_step(session_id: str, control: str) -> SessionResponse:
        session = _get_session(service_container, session_id)
        try:
            session.redo(control)
        except (SessionError, RuntimeError, ValueError) as error:
            _raise_engine(error, "control")
        return _session_response(session)

    @app.post(
        "/api/v1/sessions/{session_id}/steps/{control}/unsupported",
        response_model=SessionResponse,
    )
    def unsupported_step(session_id: str, control: str) -> SessionResponse:
        session = _get_session(service_container, session_id)
        try:
            session.mark_unsupported(control)
        except (SessionError, RuntimeError, ValueError) as error:
            _raise_engine(error, "control")
        return _session_response(session)

    @app.get(
        "/api/v1/sessions/{session_id}/preview", response_model=LogicalState
    )
    def preview(session_id: str) -> LogicalState:
        session = _get_session(service_container, session_id)
        try:
            return session.preview()
        except (SessionError, RuntimeError, ValueError) as error:
            _raise_engine(error, "state")

    @app.post(
        "/api/v1/sessions/{session_id}/save",
        response_model=SaveResponse,
        status_code=201,
    )
    def save_session(session_id: str, body: SessionSaveRequest) -> SaveResponse:
        session = _get_session(service_container, session_id)
        service_container.session_manager.ensure_interlock()
        try:
            profile = session.build_profile(body.preview_confirmations)
        except (SessionError, RuntimeError, ValueError, ValidationError) as error:
            _raise_engine(error, "save")
        stored = service_container.profile_store.save(profile)
        return SaveResponse(profile_id=stored.profile_id, profile=stored.profile)

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        service_container.session_manager.close(session_id)
        return Response(status_code=204)

    @app.post(
        "/api/v1/quick-sessions",
        response_model=QuickSessionResponse,
        status_code=201,
    )
    def create_quick_session(
        body: QuickSessionCreateRequest,
    ) -> QuickSessionResponse:
        session = service_container.session_manager.create_quick(
            body.expected_device
        )
        return _quick_session_response(session)

    @app.get("/api/v1/quick-sessions/{session_id}/events")
    async def quick_session_events(
        request: Request,
        session_id: str,
        after_id: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        session = service_container.session_manager.get_quick(session_id)
        header_id = request.headers.get("last-event-id")
        if header_id is not None and after_id == 0:
            try:
                after_id = int(header_id)
            except ValueError as error:
                raise request_api_error("Last-Event-ID") from error
            if after_id < 0:
                raise request_api_error("Last-Event-ID")
        return StreamingResponse(
            _generate_session_events(session, after_id), media_type="text/event-stream"
        )

    @app.post(
        "/api/v1/quick-sessions/{session_id}/arm/{control}",
        response_model=QuickSessionResponse,
    )
    def arm_quick_control(
        session_id: str,
        control: str,
    ) -> QuickSessionResponse:
        session = service_container.session_manager.get_quick(session_id)
        try:
            session.arm(control)
        except (SessionError, RuntimeError, ValueError) as error:
            _raise_engine(error, "control")
        return _quick_session_response(session)

    @app.post(
        "/api/v1/quick-sessions/{session_id}/save",
        response_model=SaveResponse,
        status_code=201,
    )
    def save_quick_session(session_id: str) -> SaveResponse:
        service_container.session_manager.ensure_interlock()
        session = service_container.session_manager.get_quick(session_id)
        try:
            profile = session.build_profile()
        except (SessionError, RuntimeError, ValueError, ValidationError) as error:
            _raise_engine(error, "save")
        stored = service_container.profile_store.save(profile)
        return SaveResponse(profile_id=stored.profile_id, profile=stored.profile)

    @app.delete("/api/v1/quick-sessions/{session_id}", status_code=204)
    def delete_quick_session(session_id: str) -> Response:
        service_container.session_manager.close(session_id)
        return Response(status_code=204)

    @app.post(
        "/api/v1/profiles/{profile_id}/activate",
        response_model=ActivationResponse,
    )
    def activate_profile(profile_id: str) -> ActivationResponse:
        service_container.session_manager.ensure_interlock()
        stored = service_container.profile_store.load(profile_id)
        matches = tuple(
            descriptor
            for descriptor in service_container.device_discovery()
            if descriptor.identity == stored.profile.device
        )
        if not matches:
            raise ApiError(
                status_code=409,
                code="device_identity_mismatch",
                message_zh="未检测到与配置身份一致的游戏手柄。",
            )
        if len(matches) > 1:
            raise ApiError(
                status_code=409,
                code="device_identity_ambiguous",
                message_zh="检测到多个身份相同的游戏手柄，无法安全激活。",
            )
        try:
            validate_profile(stored.profile, matches[0].capabilities)
        except ValueError as error:
            raise ApiError(
                status_code=422,
                code="capability_mismatch",
                message_zh="配置超出当前游戏手柄的能力范围。",
                details={"reason": str(error)},
            ) from error
        selection = service_container.profile_store.activate(profile_id)
        return ActivationResponse.model_validate(selection.model_dump(mode="json"))

    @app.post(
        "/api/v1/profiles/import",
        response_model=ImportResponse,
        status_code=201,
    )
    async def import_profile(request: Request) -> ImportResponse:
        media_type = (
            request.headers.get("content-type", "")
            .partition(";")[0]
            .strip()
            .lower()
        )
        if media_type != YAML_MEDIA_TYPE:
            raise ApiError(
                status_code=415,
                code="unsupported_media_type",
                message_zh="导入请求必须使用 application/yaml。",
                field_path="content-type",
            )
        try:
            yaml_text = (await request.body()).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ApiError(
                status_code=422,
                code="invalid_profile",
                message_zh="YAML 配置必须是有效的 UTF-8 文本。",
            ) from error
        stored = service_container.profile_store.import_yaml(yaml_text)
        return ImportResponse(profile_id=stored.profile_id, profile=stored.profile)

    @app.get("/api/v1/profiles/{profile_id}/export")
    def export_profile(profile_id: str) -> Response:
        yaml_text = service_container.profile_store.export_yaml(profile_id)
        return Response(
            content=yaml_text,
            media_type=YAML_MEDIA_TYPE,
            headers={
                "Content-Disposition": f'attachment; filename="{profile_id}.yaml"'
            },
        )

    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIRECTORY),
        name="gamepad-calibrator-static",
    )

    return app
