"""Deterministically evaluate full knee-lift swings at three command heights."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch


@dataclass(frozen=True)
class ThreeHeightEvaluationConfig:
  task_id: str
  checkpoint_file: str
  output_file: str
  baseline_result: str | None = None
  num_envs: int = 48
  device: str | None = None
  seed: int = 42
  vx: float = 0.5
  warmup_seconds: float = 5.0
  measurement_seconds: float = 30.0


def resolve_evaluation_paths(
  cfg: ThreeHeightEvaluationConfig,
) -> tuple[Path, Path, Path | None]:
  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(
      f"checkpoint_file is not a regular file: {checkpoint_path}"
    )
  output_path = Path(cfg.output_file).expanduser().resolve()
  baseline_path = (
    None
    if cfg.baseline_result is None
    else Path(cfg.baseline_result).expanduser().resolve()
  )
  if baseline_path is not None and not baseline_path.is_file():
    raise FileNotFoundError(
      f"baseline_result is not a regular file: {baseline_path}"
    )
  if output_path == checkpoint_path:
    raise ValueError("output_file must not resolve to checkpoint_file")
  if baseline_path is not None and output_path == baseline_path:
    raise ValueError("output_file must not resolve to baseline_result")
  if output_path.exists():
    raise FileExistsError(f"output_file already exists: {output_path}")
  return checkpoint_path, output_path, baseline_path



@dataclass(frozen=True)
class CompletedSwing:
  env_index: int
  leg_index: int
  knee_peak_m: float
  foot_peak_m: float
  knee_peak_phase: float
  mid_hip_pitch_rad: float
  mid_knee_flexion_rad: float


@dataclass(frozen=True)
class _AirborneSample:
  knee_relative_z: float
  foot_z: float
  hip_pitch: float
  knee_flexion: float


class SwingTrajectoryAccumulator:
  """Track complete contact-to-contact swings without reward-term state."""

  def __init__(self, num_envs: int, device: str) -> None:
    if num_envs < 1:
      raise ValueError("num_envs must be at least one")
    self.num_envs = num_envs
    self.device = torch.device(device)
    shape = (num_envs, 2)
    self._initialized = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._previous_contact = torch.zeros(
      shape, dtype=torch.bool, device=self.device
    )
    self._active = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._ground_z = torch.zeros(shape, device=self.device)
    self._liftoff_knee_relative_z = torch.zeros(shape, device=self.device)
    self._liftoff_ground_z = torch.zeros(shape, device=self.device)
    self._trajectories: list[list[list[_AirborneSample]]] = [
      [[], []] for _ in range(num_envs)
    ]
    self.first_contact = torch.zeros(
      shape, dtype=torch.bool, device=self.device
    )

  def discard(self, reset_environments: torch.Tensor) -> None:
    if reset_environments.shape != (self.num_envs,):
      raise ValueError(
        f"reset_environments must have shape ({self.num_envs},), "
        f"got {tuple(reset_environments.shape)}"
      )
    reset_environments = reset_environments.to(
      device=self.device, dtype=torch.bool
    )
    reset_ids = reset_environments.nonzero(as_tuple=False).flatten().tolist()
    self._initialized[reset_environments] = False
    self._active[reset_environments] = False
    self.first_contact[reset_environments] = False
    for env_index in reset_ids:
      self._trajectories[env_index][0].clear()
      self._trajectories[env_index][1].clear()

  def observe(
    self,
    *,
    in_contact: torch.Tensor,
    root_z: torch.Tensor,
    knee_z: torch.Tensor,
    foot_z: torch.Tensor,
    hip_pitch: torch.Tensor,
    knee_flexion: torch.Tensor,
    valid_environments: torch.Tensor | None = None,
  ) -> list[CompletedSwing]:
    self._validate_inputs(
      in_contact=in_contact,
      root_z=root_z,
      knee_z=knee_z,
      foot_z=foot_z,
      hip_pitch=hip_pitch,
      knee_flexion=knee_flexion,
    )
    if valid_environments is None:
      valid_environments = torch.ones(
        self.num_envs, dtype=torch.bool, device=self.device
      )
    elif valid_environments.shape != (self.num_envs,):
      raise ValueError(
        f"valid_environments must have shape ({self.num_envs},), "
        f"got {tuple(valid_environments.shape)}"
      )
    valid_environments = valid_environments.to(
      device=self.device, dtype=torch.bool
    )
    in_contact = in_contact.to(device=self.device, dtype=torch.bool)
    self.first_contact.zero_()
    completed: list[CompletedSwing] = []

    for env_index in range(self.num_envs):
      if not bool(valid_environments[env_index]):
        continue
      root_height = float(root_z[env_index].item())
      for leg_index in range(2):
        contact = bool(in_contact[env_index, leg_index])
        if not bool(self._initialized[env_index, leg_index]):
          self._initialized[env_index, leg_index] = True
          self._previous_contact[env_index, leg_index] = contact
          if contact:
            self._ground_z[env_index, leg_index] = foot_z[
              env_index, leg_index
            ]
          continue

        previous_contact = bool(
          self._previous_contact[env_index, leg_index]
        )
        liftoff = previous_contact and not contact
        landing = not previous_contact and contact

        if liftoff:
          self._active[env_index, leg_index] = True
          self._liftoff_knee_relative_z[env_index, leg_index] = (
            knee_z[env_index, leg_index] - root_z[env_index]
          )
          self._liftoff_ground_z[env_index, leg_index] = self._ground_z[
            env_index, leg_index
          ]
          self._trajectories[env_index][leg_index].clear()

        if bool(self._active[env_index, leg_index]) and not contact:
          self._trajectories[env_index][leg_index].append(
            _AirborneSample(
              knee_relative_z=float(
                knee_z[env_index, leg_index].item() - root_height
              ),
              foot_z=float(foot_z[env_index, leg_index].item()),
              hip_pitch=float(hip_pitch[env_index, leg_index].item()),
              knee_flexion=float(
                knee_flexion[env_index, leg_index].item()
              ),
            )
          )

        if landing:
          self.first_contact[env_index, leg_index] = True
          trajectory = self._trajectories[env_index][leg_index]
          if bool(self._active[env_index, leg_index]) and trajectory:
            completed.append(
              self._complete_swing(env_index, leg_index, trajectory)
            )
          self._active[env_index, leg_index] = False
          trajectory.clear()

        if contact:
          self._ground_z[env_index, leg_index] = foot_z[
            env_index, leg_index
          ]
        self._previous_contact[env_index, leg_index] = contact

    return completed

  def _complete_swing(
    self,
    env_index: int,
    leg_index: int,
    trajectory: list[_AirborneSample],
  ) -> CompletedSwing:
    knee_values = [sample.knee_relative_z for sample in trajectory]
    peak_index = max(range(len(knee_values)), key=knee_values.__getitem__)
    midpoint_index = round(0.5 * (len(trajectory) - 1))
    midpoint = trajectory[midpoint_index]
    liftoff_knee = float(
      self._liftoff_knee_relative_z[env_index, leg_index].item()
    )
    ground_z = float(self._liftoff_ground_z[env_index, leg_index].item())
    return CompletedSwing(
      env_index=env_index,
      leg_index=leg_index,
      knee_peak_m=knee_values[peak_index] - liftoff_knee,
      foot_peak_m=max(max(sample.foot_z for sample in trajectory) - ground_z, 0.0),
      knee_peak_phase=peak_index / max(len(trajectory) - 1, 1),
      mid_hip_pitch_rad=midpoint.hip_pitch,
      mid_knee_flexion_rad=midpoint.knee_flexion,
    )

  def _validate_inputs(self, **inputs: torch.Tensor) -> None:
    expected_shapes = {
      "in_contact": (self.num_envs, 2),
      "root_z": (self.num_envs,),
      "knee_z": (self.num_envs, 2),
      "foot_z": (self.num_envs, 2),
      "hip_pitch": (self.num_envs, 2),
      "knee_flexion": (self.num_envs, 2),
    }
    for name, value in inputs.items():
      if value.shape != expected_shapes[name]:
        raise ValueError(
          f"{name} must have shape {expected_shapes[name]}, "
          f"got {tuple(value.shape)}"
        )


def equal_level_slices(num_envs: int) -> dict[str, slice]:
  if num_envs < 3 or num_envs % 3 != 0:
    raise ValueError("num_envs must be divisible by 3 with nonempty groups")
  group_size = num_envs // 3
  return {
    "low": slice(0, group_size),
    "mid": slice(group_size, 2 * group_size),
    "high": slice(2 * group_size, 3 * group_size),
  }


def summarize(samples: list[float]) -> dict[str, float | int]:
  if not samples:
    raise ValueError("summarize requires at least one sample")
  converted: list[float] = []
  for index, sample in enumerate(samples):
    value = float(sample)
    if not math.isfinite(value):
      raise ValueError(f"samples[{index}] must be finite")
    converted.append(value)
  ordered = sorted(converted)
  count = len(ordered)
  mean = sum(ordered) / count
  middle = count // 2
  median = (
    ordered[middle]
    if count % 2
    else (ordered[middle - 1] + ordered[middle]) / 2.0
  )
  p95_position = 0.95 * (count - 1)
  p95_lower = math.floor(p95_position)
  p95_upper = math.ceil(p95_position)
  p95_fraction = p95_position - p95_lower
  p95 = ordered[p95_lower] + p95_fraction * (
    ordered[p95_upper] - ordered[p95_lower]
  )
  return {
    "count": count,
    "mean": mean,
    "median": median,
    "p95": p95,
    "min": ordered[0],
    "max": ordered[-1],
  }


_PROTOCOL_PATHS = (
  ("contract", "actor"),
  ("contract", "critic"),
  ("contract", "action"),
  ("evaluation", "protocol_version"),
  ("evaluation", "task_id"),
  ("evaluation", "seed"),
  ("evaluation", "num_envs"),
  ("evaluation", "device"),
  ("evaluation", "step_dt_seconds"),
  ("evaluation", "warmup_seconds"),
  ("evaluation", "warmup_steps"),
  ("evaluation", "measurement_seconds"),
  ("evaluation", "measurement_steps"),
  ("evaluation", "vx"),
  ("evaluation", "command", "twist_mps"),
  ("evaluation", "command", "swing_height_m", "low"),
  ("evaluation", "command", "swing_height_m", "mid"),
  ("evaluation", "command", "swing_height_m", "high"),
)


def _nested_value(result: dict[str, object], path: tuple[str, ...]) -> object:
  value: object = result
  for key in path:
    if not isinstance(value, dict) or key not in value:
      raise ValueError(f"result is missing {'.'.join(path)}")
    value = value[key]
  return value


def _require_finite_protocol_value(value: object, field: str) -> None:
  if type(value) in (int, float):
    if not math.isfinite(float(value)):
      raise ValueError(f"result field {field} must be finite")
    return
  if isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _require_finite_protocol_value(item, f"{field}[{index}]")


def _validate_comparable_protocol(
  baseline: dict[str, object], candidate: dict[str, object]
) -> None:
  for path in _PROTOCOL_PATHS:
    field = ".".join(path)
    baseline_value = _nested_value(baseline, path)
    candidate_value = _nested_value(candidate, path)
    _require_finite_protocol_value(baseline_value, field)
    _require_finite_protocol_value(candidate_value, field)
    if baseline_value != candidate_value:
      raise ValueError(
        f"evaluation protocol mismatch at {field}: "
        f"baseline={baseline_value!r}, candidate={candidate_value!r}"
      )


def _nested_number(result: dict[str, object], *path: str) -> float:
  value: object = result
  for key in path:
    if not isinstance(value, dict) or key not in value:
      raise ValueError(f"result is missing {'.'.join(path)}")
    value = value[key]
  if type(value) not in (int, float):
    raise TypeError(f"result field {'.'.join(path)} must be numeric")
  number = float(value)
  if not math.isfinite(number):
    raise ValueError(f"result field {'.'.join(path)} must be finite")
  return number


def _checkpoint_metadata(result: dict[str, object]) -> dict[str, str]:
  checkpoint = result.get("checkpoint")
  if not isinstance(checkpoint, dict):
    raise TypeError("result checkpoint must be an object")
  path = checkpoint.get("path")
  digest = checkpoint.get("sha256")
  if not isinstance(path, str) or not isinstance(digest, str):
    raise TypeError("checkpoint metadata must contain path and sha256 strings")
  return {"path": path, "sha256": digest}


def assess_candidate(
  baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
  _validate_comparable_protocol(baseline, candidate)
  baseline_velocity = _nested_number(
    baseline, "overall", "forward_velocity_abs_error_mps", "mean"
  )
  baseline_landing_force = _nested_number(
    baseline, "overall", "landing_force_n", "p95"
  )
  baseline_high_foot = _nested_number(
    baseline, "levels", "high", "foot_peak_m", "mean"
  )

  checks: dict[str, dict[str, object]] = {}

  def add_check(
    name: str,
    observed: float,
    comparison: str,
    threshold: float,
    passed: bool,
  ) -> None:
    checks[name] = {
      "observed": observed,
      "comparison": comparison,
      "threshold": threshold,
      "passed": bool(passed),
    }

  low_knee = _nested_number(candidate, "levels", "low", "knee_peak_m", "mean")
  mid_knee = _nested_number(candidate, "levels", "mid", "knee_peak_m", "mean")
  high_knee = _nested_number(
    candidate, "levels", "high", "knee_peak_m", "mean"
  )
  mid_phase = _nested_number(
    candidate, "levels", "mid", "knee_peak_phase", "mean"
  )
  mid_phase_p95 = _nested_number(
    candidate, "levels", "mid", "knee_peak_phase", "p95"
  )
  high_phase = _nested_number(
    candidate, "levels", "high", "knee_peak_phase", "mean"
  )
  high_phase_p95 = _nested_number(
    candidate, "levels", "high", "knee_peak_phase", "p95"
  )
  high_flexion = _nested_number(
    candidate, "levels", "high", "mid_knee_flexion_rad", "mean"
  )
  falls = _nested_number(candidate, "overall", "falls")
  velocity_error = _nested_number(
    candidate, "overall", "forward_velocity_abs_error_mps", "mean"
  )
  landing_force = _nested_number(
    candidate, "overall", "landing_force_n", "p95"
  )
  high_foot = _nested_number(
    candidate, "levels", "high", "foot_peak_m", "mean"
  )
  velocity_limit = baseline_velocity * 1.10
  landing_force_limit = baseline_landing_force * 1.15
  high_foot_improvement = high_foot - baseline_high_foot

  add_check("low_knee_peak_min_m", low_knee, ">=", 0.020, low_knee >= 0.020)
  add_check("low_knee_peak_max_m", low_knee, "<=", 0.032, low_knee <= 0.032)
  add_check("mid_knee_peak_min_m", mid_knee, ">=", 0.038, mid_knee >= 0.038)
  add_check(
    "high_knee_peak_min_m", high_knee, ">=", 0.050, high_knee >= 0.050
  )
  add_check(
    "mid_knee_peak_phase_mean_min",
    mid_phase,
    ">=",
    0.20,
    mid_phase >= 0.20,
  )
  add_check(
    "mid_knee_peak_phase_p95_max",
    mid_phase_p95,
    "<=",
    0.65,
    mid_phase_p95 <= 0.65,
  )
  add_check(
    "high_knee_peak_phase_mean_min",
    high_phase,
    ">=",
    0.20,
    high_phase >= 0.20,
  )
  add_check(
    "high_knee_peak_phase_p95_max",
    high_phase_p95,
    "<=",
    0.65,
    high_phase_p95 <= 0.65,
  )
  flexion_limit = math.radians(80.0)
  add_check(
    "high_mid_knee_flexion_max_rad",
    high_flexion,
    "<=",
    flexion_limit,
    high_flexion <= flexion_limit,
  )
  add_check("zero_falls", falls, "==", 0.0, falls == 0.0)
  add_check(
    "forward_velocity_error_max_mps",
    velocity_error,
    "<=",
    velocity_limit,
    velocity_error <= velocity_limit,
  )
  add_check(
    "landing_force_p95_max_n",
    landing_force,
    "<=",
    landing_force_limit,
    landing_force <= landing_force_limit,
  )
  add_check(
    "high_foot_peak_improvement_min_m",
    high_foot_improvement,
    ">=",
    0.01,
    high_foot_improvement >= 0.01,
  )

  mechanism_names = tuple(checks)[:-1]
  mechanism_passed = all(checks[name]["passed"] for name in mechanism_names)
  promotion_passed = bool(checks["high_foot_peak_improvement_min_m"]["passed"])
  checkpoint_status = (
    "rejected"
    if not mechanism_passed
    else "promoted"
    if promotion_passed
    else "not_promoted"
  )
  return {
    "mechanism_status": "passed" if mechanism_passed else "failed",
    "checkpoint_status": checkpoint_status,
    "checks": checks,
    "baseline_checkpoint": _checkpoint_metadata(baseline),
    "candidate_checkpoint": _checkpoint_metadata(candidate),
  }


def write_result_json(path: Path, result: dict[str, object]) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    raise FileExistsError(f"output_file already exists: {path}")
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent,
    prefix=f".{path.name}.",
    suffix=".tmp",
  )
  temporary_path = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
      temporary_file.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
      temporary_file.flush()
      os.fsync(temporary_file.fileno())
    try:
      os.link(temporary_path, path)
    except FileExistsError:
      raise FileExistsError(f"output_file already exists: {path}") from None
  finally:
    temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as checkpoint:
    for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _set_commands_and_observe(
  env: Any, level_slices: dict[str, slice], vx: float
) -> Any:
  from mjlab.tasks.velocity.mdp import UniformVelocityCommand

  from src.tasks.velocity.mdp import DiscreteSwingHeightCommand

  manager = env.unwrapped.command_manager
  twist = manager.get_term("twist")
  swing_height = manager.get_term("swing_height")
  if not isinstance(twist, UniformVelocityCommand):
    raise TypeError("twist command is not UniformVelocityCommand")
  if not isinstance(swing_height, DiscreteSwingHeightCommand):
    raise TypeError("swing_height command is not DiscreteSwingHeightCommand")
  twist.vel_command_b[:, 0] = vx
  twist.vel_command_b[:, 1:] = 0.0
  twist.is_heading_env.zero_()
  twist.is_standing_env.zero_()
  for level, target in zip(
    ("low", "mid", "high"), (0.05, 0.10, 0.15), strict=True
  ):
    swing_height.height_command[level_slices[level], 0] = target
  return env.get_observations()


def _resolve_pair(
  finder: Any, names: tuple[str, str], kind: str
) -> list[int]:
  ids, resolved_names = finder(names, preserve_order=True)
  if len(ids) != 2 or tuple(resolved_names) != names:
    raise RuntimeError(
      f"expected exactly ordered {kind} names {names}, got {resolved_names}"
    )
  return ids


def _empty_level_samples() -> dict[str, list[float]]:
  return {
    "knee_peak_m": [],
    "foot_peak_m": [],
    "knee_peak_phase": [],
    "mid_hip_pitch_rad": [],
    "mid_knee_flexion_rad": [],
    "forward_velocity_abs_error_mps": [],
    "landing_force_n": [],
  }


def run_evaluation(
  cfg: ThreeHeightEvaluationConfig,
) -> dict[str, object]:
  equal_level_slices(cfg.num_envs)
  if not math.isfinite(cfg.vx):
    raise ValueError("vx must be finite")
  if not math.isfinite(cfg.warmup_seconds) or cfg.warmup_seconds < 5.0:
    raise ValueError("warmup_seconds must be finite and at least 5")
  if (
    not math.isfinite(cfg.measurement_seconds)
    or cfg.measurement_seconds < 30.0
  ):
    raise ValueError("measurement_seconds must be finite and at least 30")

  checkpoint_path, output_path, baseline_path = resolve_evaluation_paths(cfg)
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  import src.tasks  # noqa: F401

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.episode_length_s = max(
    float(env_cfg.episode_length_s),
    cfg.warmup_seconds + cfg.measurement_seconds + 5.0,
  )

  twist_cfg = env_cfg.commands["twist"]
  twist_cfg.heading_command = False
  twist_cfg.ranges.heading = None
  twist_cfg.rel_heading_envs = 0.0
  twist_cfg.rel_standing_envs = 0.0
  twist_cfg.init_velocity_prob = 0.0
  twist_cfg.ranges.lin_vel_x = (cfg.vx, cfg.vx)
  twist_cfg.ranges.lin_vel_y = (0.0, 0.0)
  twist_cfg.ranges.ang_vel_z = (0.0, 0.0)
  twist_cfg.resampling_time_range = (1_000_000.0, 1_000_000.0)
  swing_cfg = env_cfg.commands["swing_height"]
  swing_cfg.resampling_time_range = (1_000_000.0, 1_000_000.0)
  if hasattr(swing_cfg, "log_transitions"):
    swing_cfg.log_transitions = False

  env: Any | None = None
  try:
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    contract_dims = unwrapped.observation_manager.group_obs_dim
    actor_dim = contract_dims.get("actor")
    critic_dim = contract_dims.get("critic")
    action_dim = unwrapped.action_manager.total_action_dim
    expected_contract = ((99,), (301,), 29)
    if (actor_dim, critic_dim, action_dim) != expected_contract:
      raise RuntimeError(
        "policy contract mismatch: expected actor=(99,), critic=(301,), "
        f"action=29; got actor={actor_dim}, critic={critic_dim}, "
        f"action={action_dim}"
      )

    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["feet_ground_contact"]
    knee_ids = _resolve_pair(
      robot.find_bodies,
      ("left_knee_link", "right_knee_link"),
      "knee body",
    )
    foot_ids = _resolve_pair(
      robot.find_sites, ("left_foot", "right_foot"), "foot site"
    )
    hip_ids = _resolve_pair(
      robot.find_joints,
      ("left_hip_pitch_joint", "right_hip_pitch_joint"),
      "hip joint",
    )
    knee_joint_ids = _resolve_pair(
      robot.find_joints,
      ("left_knee_joint", "right_knee_joint"),
      "knee joint",
    )

    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    level_slices = equal_level_slices(cfg.num_envs)
    env.reset()
    step_dt = float(unwrapped.step_dt)
    warmup_steps = math.ceil(cfg.warmup_seconds / step_dt)
    measurement_steps = math.ceil(cfg.measurement_seconds / step_dt)

    with torch.inference_mode():
      for _ in range(warmup_steps):
        observations = _set_commands_and_observe(env, level_slices, cfg.vx)
        actions = policy(observations)
        env.step(actions)

      tracker = SwingTrajectoryAccumulator(cfg.num_envs, device)
      found = contact_sensor.data.found
      force = contact_sensor.data.force
      if found is None or found.shape != (cfg.num_envs, 2):
        found_shape = None if found is None else tuple(found.shape)
        raise RuntimeError(
          f"contact found must have shape ({cfg.num_envs}, 2), got {found_shape}"
        )
      if force is None or force.shape != (cfg.num_envs, 2, 3):
        force_shape = None if force is None else tuple(force.shape)
        raise RuntimeError(
          "contact force must have shape "
          f"({cfg.num_envs}, 2, 3), got {force_shape}"
        )
      tracker.observe(
        in_contact=found > 0,
        root_z=robot.data.root_link_pos_w[:, 2],
        knee_z=robot.data.body_link_pos_w[:, knee_ids, 2],
        foot_z=robot.data.site_pos_w[:, foot_ids, 2],
        hip_pitch=robot.data.joint_pos[:, hip_ids],
        knee_flexion=robot.data.joint_pos[:, knee_joint_ids],
      )

      level_samples = {
        level: _empty_level_samples() for level in level_slices
      }
      fall_counts = {level: 0 for level in level_slices}

      for _ in range(measurement_steps):
        observations = _set_commands_and_observe(env, level_slices, cfg.vx)
        actions = policy(observations)
        env.step(actions)

        terminated = unwrapped.reset_terminated.to(dtype=torch.bool)
        tracker.discard(terminated)
        valid = ~terminated
        found = contact_sensor.data.found
        force = contact_sensor.data.force
        if found is None or found.shape != (cfg.num_envs, 2):
          found_shape = None if found is None else tuple(found.shape)
          raise RuntimeError(
            "contact found must have shape "
            f"({cfg.num_envs}, 2), got {found_shape}"
          )
        if force is None or force.shape != (cfg.num_envs, 2, 3):
          force_shape = None if force is None else tuple(force.shape)
          raise RuntimeError(
            "contact force must have shape "
            f"({cfg.num_envs}, 2, 3), got {force_shape}"
          )
        completed = tracker.observe(
          in_contact=found > 0,
          root_z=robot.data.root_link_pos_w[:, 2],
          knee_z=robot.data.body_link_pos_w[:, knee_ids, 2],
          foot_z=robot.data.site_pos_w[:, foot_ids, 2],
          hip_pitch=robot.data.joint_pos[:, hip_ids],
          knee_flexion=robot.data.joint_pos[:, knee_joint_ids],
          valid_environments=valid,
        )

        for level, level_slice in level_slices.items():
          fall_counts[level] += int(terminated[level_slice].sum().item())
          velocity_error = torch.abs(
            robot.data.root_link_lin_vel_b[level_slice, 0] - cfg.vx
          )
          level_samples[level][
            "forward_velocity_abs_error_mps"
          ].extend(velocity_error[valid[level_slice]].cpu().tolist())

        force_magnitude = torch.linalg.vector_norm(force, dim=-1)
        landing_force = force_magnitude[tracker.first_contact]
        landing_envs = tracker.first_contact.nonzero(as_tuple=False)[:, 0]
        for env_index, value in zip(
          landing_envs.tolist(), landing_force.cpu().tolist(), strict=True
        ):
          level = (
            "low"
            if env_index < level_slices["low"].stop
            else "mid"
            if env_index < level_slices["mid"].stop
            else "high"
          )
          level_samples[level]["landing_force_n"].append(value)

        for sample in completed:
          level = (
            "low"
            if sample.env_index < level_slices["low"].stop
            else "mid"
            if sample.env_index < level_slices["mid"].stop
            else "high"
          )
          level_samples[level]["knee_peak_m"].append(sample.knee_peak_m)
          level_samples[level]["foot_peak_m"].append(sample.foot_peak_m)
          level_samples[level]["knee_peak_phase"].append(
            sample.knee_peak_phase
          )
          level_samples[level]["mid_hip_pitch_rad"].append(
            sample.mid_hip_pitch_rad
          )
          level_samples[level]["mid_knee_flexion_rad"].append(
            sample.mid_knee_flexion_rad
          )

    for level, samples in level_samples.items():
      if not samples["knee_peak_m"] or not samples["foot_peak_m"]:
        raise RuntimeError(
          f"height level {level} has zero scored knee or foot landings"
        )
      if not samples["landing_force_n"]:
        raise RuntimeError(f"height level {level} has zero landing force samples")

    levels: dict[str, object] = {}
    for level, samples in level_samples.items():
      levels[level] = {
        **{name: summarize(values) for name, values in samples.items()},
        "falls": fall_counts[level],
      }
    velocity_samples = [
      value
      for level in ("low", "mid", "high")
      for value in level_samples[level]["forward_velocity_abs_error_mps"]
    ]
    landing_force_samples = [
      value
      for level in ("low", "mid", "high")
      for value in level_samples[level]["landing_force_n"]
    ]
    result: dict[str, object] = {
      "checkpoint": {
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
      },
      "contract": {"actor": 99, "critic": 301, "action": 29},
      "evaluation": {
        "protocol_version": "g1-knee-lift-v2",
        **asdict(cfg),
        "checkpoint_file": str(checkpoint_path),
        "output_file": str(output_path),
        "device": device,
        "step_dt_seconds": step_dt,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "command": {
          "twist_mps": [cfg.vx, 0.0, 0.0],
          "swing_height_m": {"low": 0.05, "mid": 0.10, "high": 0.15},
        },
      },
      "levels": levels,
      "overall": {
        "forward_velocity_abs_error_mps": summarize(velocity_samples),
        "landing_force_n": summarize(landing_force_samples),
        "falls": sum(fall_counts.values()),
      },
    }
    if baseline_path is not None:
      with baseline_path.open(encoding="utf-8") as baseline_file:
        baseline = json.load(baseline_file)
      if not isinstance(baseline, dict):
        raise ValueError("baseline result must contain a JSON object")
      result["decision"] = assess_candidate(
        cast(dict[str, object], baseline), result
      )
    write_result_json(output_path, result)
    return result
  finally:
    if env is not None:
      env.close()


@dataclass(frozen=True)
class _CliConfig:
  checkpoint_file: str
  output_file: str
  baseline_result: str | None = None
  num_envs: int = 48
  device: str | None = None
  seed: int = 42
  vx: float = 0.5
  warmup_seconds: float = 5.0
  measurement_seconds: float = 30.0


def main() -> None:
  import mjlab
  import mjlab.tasks
  import tyro
  from mjlab.tasks.registry import list_tasks

  import src.tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    _CliConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_evaluation(ThreeHeightEvaluationConfig(task_id=task_id, **asdict(args)))


if __name__ == "__main__":
  main()
