"""Stable Chinese error envelopes for the local calibration API."""

from __future__ import annotations

from pydantic import JsonValue

from .api_models import ErrorEnvelope
from .profile_store import ProfileError
from .sessions import SessionError


class ApiError(RuntimeError):
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

    @classmethod
    def from_session(cls, error: SessionError) -> ApiError:
        return cls(
            status_code=error.status_code,
            code=error.code,
            message_zh=error.message_zh,
            field_path=error.field_path,
            details=error.details,
        )

    def envelope(self) -> ErrorEnvelope:
        return ErrorEnvelope(
            code=self.code,
            message_zh=self.message_zh,
            field_path=self.field_path,
            details=self.details,
        )


def profile_api_error(error: ProfileError) -> ApiError:
    status_by_code = {
        "profile_not_found": 404,
        "template_not_found": 404,
        "invalid_profile_id": 422,
        "invalid_profile": 422,
        "invalid_template": 422,
        "capability_mismatch": 422,
        "storage_error": 500,
    }
    return ApiError(
        status_code=status_by_code.get(error.code, 422),
        code=error.code,
        message_zh=error.message_zh,
        field_path=error.field_path,
    )


def request_api_error(field_path: str | None = None) -> ApiError:
    return ApiError(
        status_code=422,
        code="invalid_request",
        message_zh="请求字段无效，请检查标记的参数。",
        field_path=field_path,
    )


def engine_api_error(error: Exception, operation: str) -> ApiError:
    if operation == "control":
        code = "invalid_control"
        message = "逻辑控制项无效或当前不能执行此操作。"
    elif operation == "binding":
        code = "invalid_binding"
        message = "控制绑定无效或超出设备能力范围。"
    elif operation == "save":
        code = "save_gate_failed"
        message = "配置尚未通过保存前的映射与预览确认。"
    else:
        code = "invalid_state"
        message = "当前校准状态不能执行此操作。"
    return ApiError(
        status_code=422,
        code=code,
        message_zh=message,
        details={"reason": str(error)},
    )
