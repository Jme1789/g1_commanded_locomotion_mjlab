"""Constrained YAML storage for versioned gamepad profiles."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import yaml
from pydantic import ValidationError

from .models import (
    ActiveSelection,
    DeviceCapabilities,
    DeviceIdentity,
    GamepadProfile,
    StoredProfile,
    TemplateProfile,
    validate_profile,
)

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class StoredTemplate(NamedTuple):
    template_id: str
    path: Path
    template: TemplateProfile


class ProfileError(ValueError):
    """A structured profile-domain error suitable for a later local API."""

    def __init__(self, code: str, message_zh: str, field_path: str | None = None) -> None:
        super().__init__(message_zh)
        self.code = code
        self.message_zh = message_zh
        self.field_path = field_path


class ProfileStore:
    """Read and write only profiles, templates, and active selection below one root."""

    def __init__(self, gamepad_root: Path) -> None:
        self.gamepad_root = Path(gamepad_root).resolve()
        self._profiles_dir = self.gamepad_root / "profiles"
        self._templates_dir = self.gamepad_root / "templates"
        self._active_path = self.gamepad_root / "active.yaml"

    def _checked_id(self, value: str) -> str:
        if not _PROFILE_ID.fullmatch(value) or "\x00" in value:
            raise ProfileError("invalid_profile_id", "配置文件标识无效", "profile_id")
        return value

    def _profile_path(self, profile_id: str) -> Path:
        return self._profiles_dir / f"{self._checked_id(profile_id)}.yaml"

    def _template_path(self, template_id: str) -> Path:
        return self._templates_dir / f"{self._checked_id(template_id)}.yaml"

    @staticmethod
    def _read_yaml(path: Path) -> Any:
        try:
            with path.open(encoding="utf-8") as stream:
                return yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as error:
            raise ProfileError("invalid_profile", "YAML 配置文件无效") from error

    @staticmethod
    def _parse_profile(payload: Any) -> GamepadProfile:
        try:
            profile = GamepadProfile.model_validate(payload)
            validate_profile(profile)
            return profile
        except (TypeError, ValidationError, ValueError) as error:
            raise ProfileError("invalid_profile", "游戏手柄配置无效") from error

    @staticmethod
    def _parse_template(payload: Any) -> TemplateProfile:
        try:
            template = TemplateProfile.model_validate(payload)
            profile = GamepadProfile(
                schema_version=template.schema_version,
                device=DeviceIdentity(
                    vendor_id="0000", product_id="0000", name="template", serial=None
                ),
                sticks=template.sticks,
                triggers=template.triggers,
                buttons=template.buttons,
                dpad=template.dpad,
            )
            validate_profile(profile)
            return template
        except (TypeError, ValidationError, ValueError) as error:
            raise ProfileError("invalid_template", "模板配置无效") from error

    @staticmethod
    def _yaml_text(model: GamepadProfile | ActiveSelection) -> str:
        return yaml.safe_dump(
            model.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        )

    def _atomic_write(self, destination: Path, text: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            raise ProfileError("storage_error", "保存配置文件失败") from error
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def list_profiles(self) -> tuple[StoredProfile, ...]:
        if not self._profiles_dir.exists():
            return ()
        return tuple(self.load(path.stem) for path in sorted(self._profiles_dir.glob("*.yaml")))

    def list_templates(self) -> tuple[TemplateProfile, ...]:
        return tuple(record.template for record in self.list_template_records())

    def list_template_records(self) -> tuple[StoredTemplate, ...]:
        if not self._templates_dir.exists():
            return ()
        return tuple(
            StoredTemplate(
                template_id=path.stem,
                path=path,
                template=self._parse_template(self._read_yaml(path)),
            )
            for path in sorted(self._templates_dir.glob("*.yaml"))
        )

    def load(self, profile_id: str) -> StoredProfile:
        path = self._profile_path(profile_id)
        if not path.is_file():
            raise ProfileError("profile_not_found", "配置文件不存在", "profile_id")
        return StoredProfile(profile_id=profile_id, path=path, profile=self._parse_profile(self._read_yaml(path)))

    def save(self, profile: GamepadProfile) -> StoredProfile:
        try:
            validate_profile(profile)
        except ValueError as error:
            raise ProfileError("invalid_profile", "游戏手柄配置无效") from error
        from .models import profile_id_for

        profile_id = profile_id_for(profile.device)
        path = self._profile_path(profile_id)
        self._atomic_write(path, self._yaml_text(profile))
        return StoredProfile(profile_id=profile_id, path=path, profile=profile)

    def activate(self, profile_id: str) -> ActiveSelection:
        stored = self.load(profile_id)
        selection = ActiveSelection(
            schema_version=1,
            profile=f"profiles/{stored.path.name}",
            device=stored.profile.device,
        )
        self._atomic_write(self._active_path, self._yaml_text(selection))
        return selection

    def import_yaml(self, yaml_text: str) -> StoredProfile:
        try:
            payload = yaml.safe_load(yaml_text)
        except yaml.YAMLError as error:
            raise ProfileError("invalid_profile", "YAML 配置文件无效") from error
        return self.save(self._parse_profile(payload))

    def export_yaml(self, profile_id: str) -> str:
        return self._yaml_text(self.load(profile_id).profile)

    def materialize_template(
        self,
        template_id: str,
        device: DeviceIdentity,
        capabilities: DeviceCapabilities,
    ) -> GamepadProfile:
        path = self._template_path(template_id)
        if not path.is_file():
            raise ProfileError("template_not_found", "模板配置不存在", "template_id")
        template = self._parse_template(self._read_yaml(path))
        profile = GamepadProfile(
            schema_version=template.schema_version,
            device=device,
            sticks=template.sticks,
            triggers=template.triggers,
            buttons=template.buttons,
            dpad=template.dpad,
        )
        try:
            validate_profile(profile, capabilities)
        except ValueError as error:
            raise ProfileError("capability_mismatch", "模板超出设备能力范围") from error
        return profile
