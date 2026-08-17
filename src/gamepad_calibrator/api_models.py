"""Strict request and response models for the version-one local API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .models import (
    DeviceCapabilities,
    DeviceIdentity,
    GamepadProfile,
    LogicalState,
    TemplateProfile,
)


class StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuickSessionCreateRequest(StrictApiModel):
    expected_device: DeviceIdentity | None = None


class SessionCreateRequest(StrictApiModel):
    device_path: str
    template_id: str | None = None


class StepConfirmRequest(StrictApiModel):
    binding_override: dict[str, JsonValue] | None = None


class SessionSaveRequest(StrictApiModel):
    preview_confirmations: list[str]


class ErrorEnvelope(StrictApiModel):
    code: str
    message_zh: str = Field(min_length=1)
    field_path: str | None = None
    details: JsonValue | None = None


class HealthResponse(StrictApiModel):
    status: str
    api_version: int


class DeviceResponse(StrictApiModel):
    device_path: str
    by_id_path: str | None
    identity: DeviceIdentity
    capabilities: DeviceCapabilities


class QuickRawTransitionResponse(StrictApiModel):
    time_ms: int
    kind: Literal["axis", "button"]
    number: int
    old_value: int
    new_value: int
    initial: bool
    phase: Literal["pressed", "released", "repeat", "changed", "centered"]


class QuickRawResponse(StrictApiModel):
    axes: list[int]
    buttons: list[int]
    transitions: list[QuickRawTransitionResponse]


class QuickEdgesResponse(StrictApiModel):
    pressed: dict[str, bool]
    on_pressed: list[str]
    on_released: list[str]
    combos: dict[str, bool]


class QuickCaptureResponse(StrictApiModel):
    status: Literal["idle", "armed", "collecting", "ambiguous", "captured"]
    control: str | None
    source: Literal["axis", "button"] | None
    index: int | None
    direction: Literal["negative", "positive"] | None
    primary_axis: int | None
    secondary_axis: int | None


class QuickSessionResponse(StrictApiModel):
    session_id: str
    state: str
    connected: bool
    device_count: int
    device: DeviceResponse
    armed_control: str | None
    capture: QuickCaptureResponse
    bindings: dict[str, JsonValue]
    missing_required: list[str]
    logical: LogicalState
    edges: QuickEdgesResponse
    raw: QuickRawResponse
    replacement: JsonValue | None


class DevicesResponse(StrictApiModel):
    devices: list[DeviceResponse]


class StoredProfileResponse(StrictApiModel):
    profile_id: str
    profile: GamepadProfile


class TemplateResponse(TemplateProfile):
    template_id: str


class ProfilesResponse(StrictApiModel):
    profiles: list[StoredProfileResponse]
    templates: list[TemplateResponse]


class SessionResponse(StrictApiModel):
    session_id: str
    state: str
    connected: bool
    candidate: JsonValue | None
    device: DeviceResponse | None = None


class SaveResponse(StrictApiModel):
    profile_id: str
    profile: GamepadProfile


class ImportResponse(StrictApiModel):
    profile_id: str
    profile: GamepadProfile


class ActivationResponse(StrictApiModel):
    schema_version: int
    profile: str
    device: DeviceIdentity
