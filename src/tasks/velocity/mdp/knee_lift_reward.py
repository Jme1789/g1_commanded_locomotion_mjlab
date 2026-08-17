from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_KNEE_BODY_NAMES = ("left_knee_link", "right_knee_link")
_HIP_JOINT_NAMES = ("left_hip_pitch_joint", "right_hip_pitch_joint")
_KNEE_JOINT_NAMES = ("left_knee_joint", "right_knee_joint")
_LEVEL_LABELS = ("low", "mid", "high")
_CONTACT_PRIMARIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _validated_levels(values: object, name: str) -> tuple[float, float, float]:
  if not isinstance(values, tuple) or len(values) != 3:
    raise ValueError(f"{name} must contain exactly three positive finite floats")
  if any(type(value) is not float for value in values):
    raise ValueError(f"{name} must contain exactly three positive finite floats")
  result = tuple(float(value) for value in values)
  if (
    any(not math.isfinite(value) or value <= 0.0 for value in result)
    or not result[0] < result[1] < result[2]
  ):
    raise ValueError(
      f"{name} must contain exactly three positive finite strictly increasing floats"
    )
  return result  # type: ignore[return-value]


def _id_sequence(ids: object, kind: str) -> tuple[int, int]:
  if (
    not isinstance(ids, list)
    or len(ids) != 2
    or any(type(value) is not int for value in ids)
  ):
    raise ValueError(f"{kind} must contain exactly two ordered resolved IDs")
  return tuple(ids)  # type: ignore[return-value]


def _validate_entity_cfg(
  cfg: SceneEntityCfg,
  *,
  kind: str,
  ids_attr: str,
  names_attr: str,
  expected_names: tuple[str, str],
) -> tuple[str, tuple[int, int], tuple[str, str]]:
  ids = _id_sequence(getattr(cfg, ids_attr), kind)
  names = getattr(cfg, names_attr)
  if (
    not isinstance(names, (list, tuple))
    or len(names) != 2
    or any(not isinstance(name, str) for name in names)
    or not cfg.preserve_order
    or tuple(names) != expected_names
  ):
    raise ValueError(f"{kind} must use ordered names {expected_names}")
  return cfg.name, ids, tuple(names)  # type: ignore[return-value]


class CommandedKneeLift:
  """Score a commanded, root-relative knee-lift peak once per swing landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    params = cfg.params
    self._sensor_name = self._string_param(params, "sensor_name")
    self._height_command_name = self._string_param(params, "height_command_name")
    self._command_name = self._string_param(params, "command_name")
    self._height_levels_tuple = _validated_levels(
      params["height_levels"], "height_levels"
    )
    self._knee_lift_targets_tuple = _validated_levels(
      params["knee_lift_targets"], "knee_lift_targets"
    )
    self._command_threshold = self._finite_float(
      params["command_threshold"], "command_threshold"
    )
    self._nominal_swing_time_s = self._positive_float(
      params["nominal_swing_time_s"], "nominal_swing_time_s"
    )
    self._tracking_window = self._validated_window(params["tracking_window"])

    knee_body_cfg: SceneEntityCfg = params["knee_body_cfg"]
    hip_joint_cfg: SceneEntityCfg = params["hip_joint_cfg"]
    knee_joint_cfg: SceneEntityCfg = params["knee_joint_cfg"]
    self._knee_body_signature = _validate_entity_cfg(
      knee_body_cfg,
      kind="knee bodies",
      ids_attr="body_ids",
      names_attr="body_names",
      expected_names=_KNEE_BODY_NAMES,
    )
    self._hip_joint_signature = _validate_entity_cfg(
      hip_joint_cfg,
      kind="hip joints",
      ids_attr="joint_ids",
      names_attr="joint_names",
      expected_names=_HIP_JOINT_NAMES,
    )
    self._knee_joint_signature = _validate_entity_cfg(
      knee_joint_cfg,
      kind="knee joints",
      ids_attr="joint_ids",
      names_attr="joint_names",
      expected_names=_KNEE_JOINT_NAMES,
    )
    if len({signature[0] for signature in (
      self._knee_body_signature,
      self._hip_joint_signature,
      self._knee_joint_signature,
    )}) != 1:
      raise ValueError("knee bodies and joints must belong to the same entity")

    self._height_levels = torch.tensor(
      self._height_levels_tuple, device=env.device, dtype=torch.float32
    )
    self._knee_lift_targets_by_level = torch.tensor(
      self._knee_lift_targets_tuple, device=env.device, dtype=torch.float32
    )
    contact_sensor: ContactSensor = env.scene[self._sensor_name]
    self._validate_contact_primaries(contact_sensor)
    self._validate_leg_shape(contact_sensor.data.found, env.num_envs, "contact found")

    shape = (env.num_envs, 2)
    self._contact_seen = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self._liftoff_relative_knee_z = torch.zeros(shape, device=env.device)
    self._peak_knee_lift = torch.zeros(shape, device=env.device)
    self._peak_phase = torch.zeros(shape, device=env.device)
    self._latched_targets = torch.zeros(shape, device=env.device)
    self._latched_level_indices = torch.full(
      shape, -1, device=env.device, dtype=torch.long
    )
    self._latched_active = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self._reference_initialized = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self._midpoint_sampled = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self._mid_hip_pitch = torch.zeros(shape, device=env.device)
    self._mid_knee_flexion = torch.zeros(shape, device=env.device)

  @staticmethod
  def _string_param(params: dict[str, object], name: str) -> str:
    value = params[name]
    if not isinstance(value, str):
      raise TypeError(f"{name} must be a string")
    return value

  @staticmethod
  def _finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
      raise ValueError(f"{name} must be finite")
    return float(value)

  @classmethod
  def _positive_float(cls, value: object, name: str) -> float:
    result = cls._finite_float(value, name)
    if result <= 0.0:
      raise ValueError(f"{name} must be positive and finite")
    return result

  @staticmethod
  def _validated_window(value: object) -> tuple[float, float]:
    if (
      not isinstance(value, tuple)
      or len(value) != 2
      or any(type(item) not in (int, float) for item in value)
    ):
      raise ValueError("tracking_window must contain two finite values")
    start, end = (float(item) for item in value)
    if not math.isfinite(start) or not math.isfinite(end) or not 0.0 <= start < end <= 1.0:
      raise ValueError("tracking_window must satisfy 0 <= start < end <= 1")
    return start, end

  @staticmethod
  def _validate_contact_primaries(contact_sensor: ContactSensor) -> None:
    names: list[str] = []
    for slot in contact_sensor._slots:
      primary_name = slot.primary_name
      if primary_name not in names:
        names.append(primary_name)
    if tuple(names) != _CONTACT_PRIMARIES:
      raise ValueError(
        "contact sensor primaries must be ordered left_ankle_roll_link then "
        "right_ankle_roll_link"
      )

  @staticmethod
  def _validate_leg_shape(
    tensor: torch.Tensor, num_envs: int, name: str
  ) -> None:
    expected = (num_envs, 2)
    if tensor.shape != expected:
      raise ValueError(f"{name} expected shape {expected}, got {tuple(tensor.shape)}")

  @staticmethod
  def _entity_signature(
    cfg: SceneEntityCfg,
    *,
    kind: str,
    ids_attr: str,
    names_attr: str,
    expected_names: tuple[str, str],
  ) -> tuple[str, tuple[int, int], tuple[str, str]]:
    return _validate_entity_cfg(
      cfg,
      kind=kind,
      ids_attr=ids_attr,
      names_attr=names_attr,
      expected_names=expected_names,
    )

  def _validate_runtime_params(
    self,
    sensor_name: str,
    height_command_name: str,
    height_levels: tuple[float, float, float],
    knee_lift_targets: tuple[float, float, float],
    command_name: str,
    command_threshold: float,
    nominal_swing_time_s: float,
    tracking_window: tuple[float, float],
    knee_body_cfg: SceneEntityCfg,
    hip_joint_cfg: SceneEntityCfg,
    knee_joint_cfg: SceneEntityCfg,
  ) -> None:
    if (
      sensor_name != self._sensor_name
      or height_command_name != self._height_command_name
      or command_name != self._command_name
      or self._finite_float(command_threshold, "command_threshold")
      != self._command_threshold
      or self._positive_float(nominal_swing_time_s, "nominal_swing_time_s")
      != self._nominal_swing_time_s
      or self._validated_window(tracking_window) != self._tracking_window
      or _validated_levels(height_levels, "height_levels") != self._height_levels_tuple
      or _validated_levels(knee_lift_targets, "knee_lift_targets")
      != self._knee_lift_targets_tuple
      or self._entity_signature(
        knee_body_cfg,
        kind="knee bodies",
        ids_attr="body_ids",
        names_attr="body_names",
        expected_names=_KNEE_BODY_NAMES,
      )
      != self._knee_body_signature
      or self._entity_signature(
        hip_joint_cfg,
        kind="hip joints",
        ids_attr="joint_ids",
        names_attr="joint_names",
        expected_names=_HIP_JOINT_NAMES,
      )
      != self._hip_joint_signature
      or self._entity_signature(
        knee_joint_cfg,
        kind="knee joints",
        ids_attr="joint_ids",
        names_attr="joint_names",
        expected_names=_KNEE_JOINT_NAMES,
      )
      != self._knee_joint_signature
    ):
      raise ValueError("commanded knee-lift reward parameters changed after initialization")

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self._contact_seen[env_ids] = False
    self._liftoff_relative_knee_z[env_ids] = 0.0
    self._peak_knee_lift[env_ids] = 0.0
    self._peak_phase[env_ids] = 0.0
    self._latched_targets[env_ids] = 0.0
    self._latched_level_indices[env_ids] = -1
    self._latched_active[env_ids] = False
    self._reference_initialized[env_ids] = False
    self._midpoint_sampled[env_ids] = False
    self._mid_hip_pitch[env_ids] = 0.0
    self._mid_knee_flexion[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_command_name: str,
    height_levels: tuple[float, float, float],
    knee_lift_targets: tuple[float, float, float],
    command_name: str,
    command_threshold: float,
    nominal_swing_time_s: float,
    tracking_window: tuple[float, float],
    knee_body_cfg: SceneEntityCfg,
    hip_joint_cfg: SceneEntityCfg,
    knee_joint_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    self._validate_runtime_params(
      sensor_name, height_command_name, height_levels, knee_lift_targets,
      command_name, command_threshold, nominal_swing_time_s, tracking_window,
      knee_body_cfg, hip_joint_cfg, knee_joint_cfg,
    )
    asset: Entity = env.scene[knee_body_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    knee_z = asset.data.body_link_pos_w[:, knee_body_cfg.body_ids, 2]
    root_z = asset.data.root_link_pos_w[:, 2].unsqueeze(1)
    relative_knee_z = knee_z - root_z
    air_time = sensor.data.current_air_time
    first_air = sensor.compute_first_air(env.step_dt)
    first_contact = sensor.compute_first_contact(env.step_dt)
    for name, tensor in (
      ("knee height", knee_z),
      ("relative knee height", relative_knee_z),
      ("air time", air_time),
      ("first air", first_air),
      ("first contact", first_contact),
      ("contact found", sensor.data.found),
    ):
      self._validate_leg_shape(tensor, env.num_envs, name)

    height_command = env.command_manager.get_command(height_command_name)
    if height_command is None or height_command.shape != (env.num_envs, 1):
      raise ValueError(
        f"Command '{height_command_name}' must have shape ({env.num_envs}, 1)"
      )
    if not torch.isfinite(height_command).all():
      raise ValueError(f"Command '{height_command_name}' must be finite")
    twist = env.command_manager.get_command(command_name)
    if twist is None or twist.shape != (env.num_envs, 3):
      raise ValueError(f"Command '{command_name}' must have shape ({env.num_envs}, 3)")
    if not torch.isfinite(twist).all():
      raise ValueError(f"Command '{command_name}' must be finite")

    level_matches = torch.isclose(
      height_command, self._height_levels.to(height_command), rtol=0.0, atol=1e-6
    )
    if not torch.all(level_matches.sum(dim=1) == 1):
      raise ValueError("each swing_height must match exactly one configured height level")
    level_index = level_matches.to(torch.long).argmax(dim=1)
    liftoff = first_air & self._contact_seen
    self._contact_seen |= sensor.data.found.to(dtype=torch.bool)
    self._liftoff_relative_knee_z = torch.where(
      liftoff, relative_knee_z, self._liftoff_relative_knee_z
    )
    self._peak_knee_lift = torch.where(
      liftoff, torch.zeros_like(self._peak_knee_lift), self._peak_knee_lift
    )
    self._peak_phase = torch.where(
      liftoff, torch.zeros_like(self._peak_phase), self._peak_phase
    )
    self._latched_targets = torch.where(
      liftoff,
      self._knee_lift_targets_by_level.to(relative_knee_z)[level_index]
      .unsqueeze(1).expand_as(self._latched_targets),
      self._latched_targets,
    )
    self._latched_level_indices = torch.where(
      liftoff,
      level_index.unsqueeze(1).expand_as(self._latched_level_indices),
      self._latched_level_indices,
    )
    self._latched_active = torch.where(
      liftoff,
      (twist[:, 0] > self._command_threshold)
      .unsqueeze(1).expand_as(self._latched_active),
      self._latched_active,
    )
    self._reference_initialized |= liftoff
    self._midpoint_sampled = torch.where(
      liftoff, torch.zeros_like(self._midpoint_sampled), self._midpoint_sampled
    )
    self._mid_hip_pitch = torch.where(
      liftoff, torch.zeros_like(self._mid_hip_pitch), self._mid_hip_pitch
    )
    self._mid_knee_flexion = torch.where(
      liftoff, torch.zeros_like(self._mid_knee_flexion), self._mid_knee_flexion
    )

    phase = torch.clamp(air_time / self._nominal_swing_time_s, 0.0, 1.0)
    knee_lift = torch.clamp_min(
      relative_knee_z - self._liftoff_relative_knee_z, 0.0
    )
    in_window = (phase >= self._tracking_window[0]) & (
      phase <= self._tracking_window[1]
    )
    tracking = self._reference_initialized & self._latched_active & in_window
    better_peak = tracking & (knee_lift > self._peak_knee_lift)
    self._peak_knee_lift = torch.where(
      better_peak, knee_lift, self._peak_knee_lift
    )
    self._peak_phase = torch.where(better_peak, phase, self._peak_phase)

    hip_pitch = asset.data.joint_pos[:, hip_joint_cfg.joint_ids]
    knee_flexion = asset.data.joint_pos[:, knee_joint_cfg.joint_ids]
    self._validate_leg_shape(hip_pitch, env.num_envs, "hip pitch")
    self._validate_leg_shape(knee_flexion, env.num_envs, "knee flexion")
    midpoint = tracking & (phase >= 0.5) & ~self._midpoint_sampled
    self._mid_hip_pitch = torch.where(midpoint, hip_pitch, self._mid_hip_pitch)
    self._mid_knee_flexion = torch.where(
      midpoint, knee_flexion, self._mid_knee_flexion
    )
    self._midpoint_sampled |= midpoint

    landing = first_contact
    scored_landing = landing & self._reference_initialized & self._latched_active
    normalized_error = torch.square(
      self._peak_knee_lift
      / self._latched_targets.clamp_min(torch.finfo(knee_z.dtype).eps)
      - 1.0
    )
    cost = torch.sum(normalized_error * scored_landing.float(), dim=1)
    self._log_metrics(env, scored_landing, normalized_error)
    self._clear_landed(landing)
    return cost

  def _log_metrics(
    self,
    env: ManagerBasedRlEnv,
    scored_landing: torch.Tensor,
    normalized_error: torch.Tensor,
  ) -> None:
    log = env.extras.setdefault("log", {})
    for index, label in enumerate(_LEVEL_LABELS):
      level_landings = scored_landing & (self._latched_level_indices == index)
      midpoint_landings = level_landings & self._midpoint_sampled
      log[f"Metrics/knee_lift_peak_{label}_m"] = self._peak_knee_lift[
        level_landings
      ]
      log[f"Metrics/knee_lift_peak_phase_{label}"] = self._peak_phase[
        level_landings
      ]
      log[f"Metrics/knee_mid_swing_hip_pitch_{label}_rad"] = self._mid_hip_pitch[
        midpoint_landings
      ]
      log[f"Metrics/knee_mid_swing_flexion_{label}_rad"] = self._mid_knee_flexion[
        midpoint_landings
      ]
    log["Metrics/knee_lift_normalized_error"] = normalized_error[scored_landing]

  def _clear_landed(self, landing: torch.Tensor) -> None:
    self._liftoff_relative_knee_z = torch.where(
      landing, torch.zeros_like(self._liftoff_relative_knee_z), self._liftoff_relative_knee_z
    )
    self._peak_knee_lift = torch.where(
      landing, torch.zeros_like(self._peak_knee_lift), self._peak_knee_lift
    )
    self._peak_phase = torch.where(
      landing, torch.zeros_like(self._peak_phase), self._peak_phase
    )
    self._latched_targets = torch.where(
      landing, torch.zeros_like(self._latched_targets), self._latched_targets
    )
    self._latched_level_indices = torch.where(
      landing, torch.full_like(self._latched_level_indices, -1), self._latched_level_indices
    )
    self._latched_active &= ~landing
    self._reference_initialized &= ~landing
    self._midpoint_sampled &= ~landing
    self._mid_hip_pitch = torch.where(
      landing, torch.zeros_like(self._mid_hip_pitch), self._mid_hip_pitch
    )
    self._mid_knee_flexion = torch.where(
      landing, torch.zeros_like(self._mid_knee_flexion), self._mid_knee_flexion
    )
