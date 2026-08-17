from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_KNEE_BODY_NAMES = ("left_knee_link", "right_knee_link")
_FOOT_SITE_NAMES = ("left_foot", "right_foot")
_CONTACT_PRIMARIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_LEVEL_LABELS = ("short", "medium", "long")


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


def _relative_forward(
  root_pos: torch.Tensor, root_quat: torch.Tensor, point_pos: torch.Tensor
) -> torch.Tensor:
  relative = point_pos - root_pos.unsqueeze(1)
  yaw = yaw_quat(root_quat).unsqueeze(1).expand(-1, relative.shape[1], -1)
  return quat_apply_inverse(yaw, relative)[..., 0]


class CommandedKneeForward:
  """Score heading-frame knee advance once per commanded swing landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    params = cfg.params
    self._sensor_name = self._string_param(params, "sensor_name")
    self._step_length_command_name = self._string_param(
      params, "step_length_command_name"
    )
    self._command_name = self._string_param(params, "command_name")
    self._step_length_levels_tuple = _validated_levels(
      params["step_length_levels"], "step_length_levels"
    )
    self._knee_forward_targets_tuple = _validated_levels(
      params["knee_forward_targets"], "knee_forward_targets"
    )
    self._command_threshold = self._finite_float(
      params["command_threshold"], "command_threshold"
    )
    self._nominal_swing_time_s = self._positive_float(
      params["nominal_swing_time_s"], "nominal_swing_time_s"
    )
    self._tracking_window = self._validated_window(params["tracking_window"])

    knee_body_cfg: SceneEntityCfg = params["knee_body_cfg"]
    foot_site_cfg: SceneEntityCfg = params["foot_site_cfg"]
    self._knee_body_signature = _validate_entity_cfg(
      knee_body_cfg,
      kind="knee bodies",
      ids_attr="body_ids",
      names_attr="body_names",
      expected_names=_KNEE_BODY_NAMES,
    )
    self._foot_site_signature = _validate_entity_cfg(
      foot_site_cfg,
      kind="foot sites",
      ids_attr="site_ids",
      names_attr="site_names",
      expected_names=_FOOT_SITE_NAMES,
    )
    if self._knee_body_signature[0] != self._foot_site_signature[0]:
      raise ValueError("knee bodies and foot sites must belong to the same entity")

    self._step_length_levels = torch.tensor(
      self._step_length_levels_tuple, device=env.device, dtype=torch.float32
    )
    self._knee_forward_targets_by_level = torch.tensor(
      self._knee_forward_targets_tuple, device=env.device, dtype=torch.float32
    )
    contact_sensor: ContactSensor = env.scene[self._sensor_name]
    self._validate_contact_primaries(contact_sensor)
    self._validate_leg_shape(contact_sensor.data.found, env.num_envs, "contact found")

    shape = (env.num_envs, 2)
    self._contact_seen = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self._reference_initialized = torch.zeros(
      shape, device=env.device, dtype=torch.bool
    )
    self._liftoff_knee_forward = torch.zeros(shape, device=env.device)
    self._liftoff_foot_forward = torch.zeros(shape, device=env.device)
    self._peak_knee_forward = torch.zeros(shape, device=env.device)
    self._peak_phase = torch.zeros(shape, device=env.device)
    self._latched_targets = torch.zeros(shape, device=env.device)
    self._latched_level_indices = torch.full(
      shape, -1, device=env.device, dtype=torch.long
    )
    self._latched_active = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self._sampled_landing_reach = torch.zeros(shape, device=env.device)
    self._pending_metrics: dict[str, torch.Tensor] = {}

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
    if (
      not math.isfinite(start)
      or not math.isfinite(end)
      or not 0.0 <= start < end <= 1.0
    ):
      raise ValueError("tracking_window must satisfy 0 <= start < end <= 1")
    return start, end

  @staticmethod
  def _validate_contact_primaries(contact_sensor: ContactSensor) -> None:
    names: list[str] = []
    for slot in contact_sensor._slots:
      if slot.primary_name not in names:
        names.append(slot.primary_name)
    if tuple(names) != _CONTACT_PRIMARIES:
      raise ValueError(
        "contact sensor primaries must be ordered left_ankle_roll_link then "
        "right_ankle_roll_link"
      )

  @staticmethod
  def _validate_shape(
    tensor: torch.Tensor, expected: tuple[int, ...], name: str
  ) -> None:
    if tensor.shape != expected:
      raise ValueError(f"{name} expected shape {expected}, got {tuple(tensor.shape)}")

  @classmethod
  def _validate_leg_shape(
    cls, tensor: torch.Tensor, num_envs: int, name: str
  ) -> None:
    cls._validate_shape(tensor, (num_envs, 2), name)

  @staticmethod
  def _validate_entity_pose_shape(
    tensor: torch.Tensor,
    num_envs: int,
    ids: list[int],
    name: str,
  ) -> None:
    if (
      tensor.ndim != 3
      or tensor.shape[0] != num_envs
      or tensor.shape[2] != 3
      or any(index < 0 or index >= tensor.shape[1] for index in ids)
    ):
      raise ValueError(
        f"{name} shape must provide ({num_envs}, entities, 3) and resolved IDs {ids}"
      )

  @staticmethod
  def _validate_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
      raise ValueError(f"{name} must be finite")

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
    step_length_command_name: str,
    step_length_levels: tuple[float, float, float],
    knee_forward_targets: tuple[float, float, float],
    command_name: str,
    command_threshold: float,
    nominal_swing_time_s: float,
    tracking_window: tuple[float, float],
    knee_body_cfg: SceneEntityCfg,
    foot_site_cfg: SceneEntityCfg,
  ) -> None:
    if (
      sensor_name != self._sensor_name
      or step_length_command_name != self._step_length_command_name
      or command_name != self._command_name
      or self._finite_float(command_threshold, "command_threshold")
      != self._command_threshold
      or self._positive_float(nominal_swing_time_s, "nominal_swing_time_s")
      != self._nominal_swing_time_s
      or self._validated_window(tracking_window) != self._tracking_window
      or _validated_levels(step_length_levels, "step_length_levels")
      != self._step_length_levels_tuple
      or _validated_levels(knee_forward_targets, "knee_forward_targets")
      != self._knee_forward_targets_tuple
      or self._entity_signature(
        knee_body_cfg,
        kind="knee bodies",
        ids_attr="body_ids",
        names_attr="body_names",
        expected_names=_KNEE_BODY_NAMES,
      )
      != self._knee_body_signature
      or self._entity_signature(
        foot_site_cfg,
        kind="foot sites",
        ids_attr="site_ids",
        names_attr="site_names",
        expected_names=_FOOT_SITE_NAMES,
      )
      != self._foot_site_signature
    ):
      raise ValueError(
        "commanded knee-forward reward parameters changed after initialization"
      )

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self._contact_seen[env_ids] = False
    self._reference_initialized[env_ids] = False
    self._liftoff_knee_forward[env_ids] = 0.0
    self._liftoff_foot_forward[env_ids] = 0.0
    self._peak_knee_forward[env_ids] = 0.0
    self._peak_phase[env_ids] = 0.0
    self._latched_targets[env_ids] = 0.0
    self._latched_level_indices[env_ids] = -1
    self._latched_active[env_ids] = False
    self._sampled_landing_reach[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    step_length_command_name: str,
    step_length_levels: tuple[float, float, float],
    knee_forward_targets: tuple[float, float, float],
    command_name: str,
    command_threshold: float,
    nominal_swing_time_s: float,
    tracking_window: tuple[float, float],
    knee_body_cfg: SceneEntityCfg,
    foot_site_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    self._flush_pending_metrics(env)
    self._validate_runtime_params(
      sensor_name,
      step_length_command_name,
      step_length_levels,
      knee_forward_targets,
      command_name,
      command_threshold,
      nominal_swing_time_s,
      tracking_window,
      knee_body_cfg,
      foot_site_cfg,
    )
    asset: Entity = env.scene[knee_body_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    root_pos = asset.data.root_link_pos_w
    root_quat = asset.data.root_link_quat_w
    all_knee_pos = asset.data.body_link_pos_w
    all_foot_pos = asset.data.site_pos_w
    self._validate_entity_pose_shape(
      all_knee_pos, env.num_envs, knee_body_cfg.body_ids, "knee position"
    )
    self._validate_entity_pose_shape(
      all_foot_pos, env.num_envs, foot_site_cfg.site_ids, "foot position"
    )
    self._validate_finite(all_knee_pos, "knee position")
    self._validate_finite(all_foot_pos, "foot position")
    knee_pos = all_knee_pos[:, knee_body_cfg.body_ids]
    foot_pos = all_foot_pos[:, foot_site_cfg.site_ids]
    air_time = sensor.data.current_air_time
    first_air = sensor.compute_first_air(env.step_dt)
    first_contact = sensor.compute_first_contact(env.step_dt)
    found = sensor.data.found

    self._validate_shape(root_pos, (env.num_envs, 3), "root position")
    self._validate_shape(root_quat, (env.num_envs, 4), "root quaternion")
    self._validate_shape(knee_pos, (env.num_envs, 2, 3), "knee position")
    self._validate_shape(foot_pos, (env.num_envs, 2, 3), "foot position")
    for name, tensor in (
      ("air time", air_time),
      ("first air", first_air),
      ("first contact", first_contact),
      ("contact found", found),
    ):
      self._validate_leg_shape(tensor, env.num_envs, name)
    for name, tensor in (
      ("root position", root_pos),
      ("root quaternion", root_quat),
      ("knee position", knee_pos),
      ("foot position", foot_pos),
      ("air time", air_time),
    ):
      self._validate_finite(tensor, name)

    step_length = env.command_manager.get_command(step_length_command_name)
    if step_length is None or step_length.shape != (env.num_envs, 1):
      raise ValueError(
        f"Command '{step_length_command_name}' must have shape ({env.num_envs}, 1)"
      )
    self._validate_finite(step_length, f"Command '{step_length_command_name}'")
    twist = env.command_manager.get_command(command_name)
    if twist is None or twist.shape != (env.num_envs, 3):
      raise ValueError(f"Command '{command_name}' must have shape ({env.num_envs}, 3)")
    self._validate_finite(twist, f"Command '{command_name}'")

    level_matches = torch.isclose(
      step_length,
      self._step_length_levels.to(step_length),
      rtol=0.0,
      atol=1e-6,
    )
    if not torch.all(level_matches.sum(dim=1) == 1):
      raise ValueError(
        "each step_length must match exactly one configured step-length level"
      )
    level_index = level_matches.to(torch.long).argmax(dim=1)

    knee_forward = _relative_forward(root_pos, root_quat, knee_pos)
    foot_forward = _relative_forward(root_pos, root_quat, foot_pos)
    liftoff = first_air.to(dtype=torch.bool) & self._contact_seen
    self._contact_seen |= found.to(dtype=torch.bool)
    self._liftoff_knee_forward = torch.where(
      liftoff, knee_forward, self._liftoff_knee_forward
    )
    self._liftoff_foot_forward = torch.where(
      liftoff, foot_forward, self._liftoff_foot_forward
    )
    self._peak_knee_forward = torch.where(
      liftoff, torch.zeros_like(self._peak_knee_forward), self._peak_knee_forward
    )
    self._peak_phase = torch.where(
      liftoff, torch.zeros_like(self._peak_phase), self._peak_phase
    )
    targets = self._knee_forward_targets_by_level.to(knee_forward)[level_index]
    self._latched_targets = torch.where(
      liftoff,
      targets.unsqueeze(1).expand_as(self._latched_targets),
      self._latched_targets,
    )
    self._latched_level_indices = torch.where(
      liftoff,
      level_index.unsqueeze(1).expand_as(self._latched_level_indices),
      self._latched_level_indices,
    )
    active = twist[:, 0] > self._command_threshold
    self._latched_active = torch.where(
      liftoff,
      active.unsqueeze(1).expand_as(self._latched_active),
      self._latched_active,
    )
    self._reference_initialized |= liftoff
    self._sampled_landing_reach = torch.where(
      liftoff,
      torch.zeros_like(self._sampled_landing_reach),
      self._sampled_landing_reach,
    )

    phase = torch.clamp(air_time / self._nominal_swing_time_s, 0.0, 1.0)
    knee_advance = knee_forward - self._liftoff_knee_forward
    in_window = (phase >= self._tracking_window[0]) & (
      phase <= self._tracking_window[1]
    )
    tracking = self._reference_initialized & self._latched_active & in_window
    better_peak = tracking & (knee_advance > self._peak_knee_forward)
    self._peak_knee_forward = torch.where(
      better_peak, knee_advance, self._peak_knee_forward
    )
    self._peak_phase = torch.where(better_peak, phase, self._peak_phase)

    landing = first_contact.to(dtype=torch.bool)
    scored_landing = landing & self._reference_initialized & self._latched_active
    landing_reach = foot_forward - self._liftoff_foot_forward
    self._sampled_landing_reach = torch.where(
      scored_landing, landing_reach, self._sampled_landing_reach
    )
    normalized_error = torch.square(
      self._peak_knee_forward
      / self._latched_targets.clamp_min(torch.finfo(knee_forward.dtype).eps)
      - 1.0
    )
    cost = torch.sum(normalized_error * scored_landing.float(), dim=1)
    self._capture_pending_metrics(scored_landing, normalized_error)
    self._clear_landed(landing)
    return cost

  def _capture_pending_metrics(
    self,
    scored_landing: torch.Tensor,
    normalized_error: torch.Tensor,
  ) -> None:
    for index, label in enumerate(_LEVEL_LABELS):
      level_landings = scored_landing & (self._latched_level_indices == index)
      if level_landings.any():
        self._pending_metrics[f"Metrics/knee_forward_peak_{label}_m"] = (
          self._peak_knee_forward[level_landings]
        )
        self._pending_metrics[f"Metrics/knee_forward_peak_phase_{label}"] = (
          self._peak_phase[level_landings]
        )
        self._pending_metrics[f"Metrics/landing_reach_{label}_m"] = (
          self._sampled_landing_reach[level_landings]
        )
    if scored_landing.any():
      self._pending_metrics["Metrics/knee_forward_normalized_error"] = (
        normalized_error[scored_landing]
      )

  def _flush_pending_metrics(self, env: ManagerBasedRlEnv) -> None:
    if not self._pending_metrics:
      return
    log = env.extras.setdefault("log", {})
    for key, samples in self._pending_metrics.items():
      if key in log:
        try:
          existing = torch.as_tensor(
            log[key], device=samples.device, dtype=samples.dtype
          ).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as error:
          raise ValueError(
            f"Existing metric '{key}' must be convertible to a tensor"
          ) from error
        if not torch.isfinite(existing).all():
          raise ValueError(f"Existing metric '{key}' must be finite")
        log[key] = torch.cat((existing, samples))
      else:
        log[key] = samples
    self._pending_metrics.clear()

  def _clear_landed(self, landing: torch.Tensor) -> None:
    self._reference_initialized &= ~landing
    self._liftoff_knee_forward = torch.where(
      landing,
      torch.zeros_like(self._liftoff_knee_forward),
      self._liftoff_knee_forward,
    )
    self._liftoff_foot_forward = torch.where(
      landing,
      torch.zeros_like(self._liftoff_foot_forward),
      self._liftoff_foot_forward,
    )
    self._peak_knee_forward = torch.where(
      landing, torch.zeros_like(self._peak_knee_forward), self._peak_knee_forward
    )
    self._peak_phase = torch.where(
      landing, torch.zeros_like(self._peak_phase), self._peak_phase
    )
    self._latched_targets = torch.where(
      landing, torch.zeros_like(self._latched_targets), self._latched_targets
    )
    self._latched_level_indices = torch.where(
      landing,
      torch.full_like(self._latched_level_indices, -1),
      self._latched_level_indices,
    )
    self._latched_active &= ~landing
    self._sampled_landing_reach = torch.where(
      landing,
      torch.zeros_like(self._sampled_landing_reach),
      self._sampled_landing_reach,
    )
