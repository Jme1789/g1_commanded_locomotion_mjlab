"""Evaluate commanded G1 gait over nine height/step-length combinations."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from scripts.evaluate_three_height_gait import (
  _checkpoint_metadata,
  _nested_number,
  _resolve_pair,
  _sha256,
  summarize,
  write_result_json,
)
from scripts.evaluate_three_height_gait import (
  resolve_evaluation_paths as _resolve_evaluation_paths,
)

HEIGHTS = ("low", "mid", "high")
LENGTHS = ("short", "medium", "long")
HEIGHT_TARGETS = (0.05, 0.10, 0.15)
LENGTH_TARGETS = (0.20, 0.30, 0.40)
METRICS = (
  "knee_forward_peak_m",
  "knee_forward_peak_phase",
  "knee_lift_m",
  "foot_clearance_m",
  "landing_reach_m",
)


@dataclass(frozen=True)
class StepLengthEvaluationConfig:
  task_id: str
  checkpoint_file: str
  output_file: str
  baseline_result: str | None = None
  num_envs: int = 54
  device: str | None = None
  seed: int = 42
  vx: float = 0.5
  warmup_seconds: float = 5.0
  measurement_seconds: float = 30.0


def equal_combination_slices(num_envs: int) -> dict[str, dict[str, slice]]:
  if num_envs < 9 or num_envs % 9 != 0:
    raise ValueError("num_envs must be divisible by 9 with nonempty groups")
  size = num_envs // 9
  return {
    height: {
      length: slice(index * size, (index + 1) * size)
      for index, length in enumerate(LENGTHS, start=height_index * 3)
    }
    for height_index, height in enumerate(HEIGHTS)
  }


def resolve_evaluation_paths(
  cfg: StepLengthEvaluationConfig,
) -> tuple[Path, Path, Path | None]:
  return _resolve_evaluation_paths(cfg)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CompletedStride:
  env_index: int
  leg_index: int
  knee_forward_peak_m: float
  knee_forward_peak_phase: float
  knee_lift_m: float
  foot_clearance_m: float
  landing_reach_m: float


@dataclass(frozen=True)
class _AirborneSample:
  knee_forward_m: float
  knee_relative_z: float
  foot_z: float


class StrideTrajectoryAccumulator:
  """Track complete contact-to-contact strides in the heading frame."""

  def __init__(self, num_envs: int, device: str) -> None:
    if num_envs < 1:
      raise ValueError("num_envs must be at least one")
    self.num_envs = num_envs
    self.device = torch.device(device)
    shape = (num_envs, 2)
    self._initialized = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._previous_contact = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._active = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._ground_z = torch.zeros(shape, dtype=torch.float64, device=self.device)
    self._liftoff_knee_forward = torch.zeros(shape, dtype=torch.float64, device=self.device)
    self._liftoff_knee_z = torch.zeros(shape, dtype=torch.float64, device=self.device)
    self._liftoff_foot_forward = torch.zeros(shape, dtype=torch.float64, device=self.device)
    self._liftoff_ground_z = torch.zeros(shape, dtype=torch.float64, device=self.device)
    self._trajectories: list[list[list[_AirborneSample]]] = [
      [[], []] for _ in range(num_envs)
    ]
    self.first_contact = torch.zeros(shape, dtype=torch.bool, device=self.device)

  def discard(self, reset_environments: torch.Tensor) -> None:
    if reset_environments.shape != (self.num_envs,):
      raise ValueError(
        f"reset_environments must have shape ({self.num_envs},), "
        f"got {tuple(reset_environments.shape)}"
      )
    resets = reset_environments.to(device=self.device, dtype=torch.bool)
    self._initialized[resets] = False
    self._active[resets] = False
    self.first_contact[resets] = False
    for env_index in resets.nonzero(as_tuple=False).flatten().tolist():
      self._trajectories[env_index][0].clear()
      self._trajectories[env_index][1].clear()

  @staticmethod
  def _forward(
    root_position_w: torch.Tensor,
    root_quaternion_w: torch.Tensor,
    point_position_w: torch.Tensor,
  ) -> torch.Tensor:
    relative = point_position_w - root_position_w.unsqueeze(1)
    w, x, y, z = root_quaternion_w.unbind(dim=-1)
    heading_x = 1.0 - 2.0 * (y * y + z * z)
    heading_y = 2.0 * (x * y + w * z)
    return relative[..., 0] * heading_x.unsqueeze(1) + relative[..., 1] * heading_y.unsqueeze(1)

  def observe(
    self,
    *,
    in_contact: torch.Tensor,
    root_position_w: torch.Tensor,
    root_quaternion_w: torch.Tensor,
    knee_position_w: torch.Tensor,
    foot_position_w: torch.Tensor,
    valid_environments: torch.Tensor | None = None,
  ) -> list[CompletedStride]:
    expected = {
      "in_contact": (self.num_envs, 2),
      "root_position_w": (self.num_envs, 3),
      "root_quaternion_w": (self.num_envs, 4),
      "knee_position_w": (self.num_envs, 2, 3),
      "foot_position_w": (self.num_envs, 2, 3),
    }
    for name, tensor in (
      ("in_contact", in_contact),
      ("root_position_w", root_position_w),
      ("root_quaternion_w", root_quaternion_w),
      ("knee_position_w", knee_position_w),
      ("foot_position_w", foot_position_w),
    ):
      if tensor.shape != expected[name]:
        raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}")
    if valid_environments is None:
      valid_environments = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    elif valid_environments.shape != (self.num_envs,):
      raise ValueError(
        f"valid_environments must have shape ({self.num_envs},), "
        f"got {tuple(valid_environments.shape)}"
      )

    contact = in_contact.to(device=self.device, dtype=torch.bool)
    valid = valid_environments.to(device=self.device, dtype=torch.bool)
    root = root_position_w.to(self.device)
    knee = knee_position_w.to(self.device)
    foot = foot_position_w.to(self.device)
    knee_forward = self._forward(root, root_quaternion_w.to(self.device), knee)
    foot_forward = self._forward(root, root_quaternion_w.to(self.device), foot)
    knee_relative_z = knee[..., 2] - root[:, None, 2]
    self.first_contact.zero_()
    completed: list[CompletedStride] = []

    for env_index in range(self.num_envs):
      if not bool(valid[env_index]):
        continue
      for leg_index in range(2):
        current_contact = bool(contact[env_index, leg_index])
        if not bool(self._initialized[env_index, leg_index]):
          self._initialized[env_index, leg_index] = True
          self._previous_contact[env_index, leg_index] = current_contact
          if current_contact:
            self._ground_z[env_index, leg_index] = foot[env_index, leg_index, 2]
          continue
        previous_contact = bool(self._previous_contact[env_index, leg_index])
        liftoff = previous_contact and not current_contact
        landing = not previous_contact and current_contact
        trajectory = self._trajectories[env_index][leg_index]
        if liftoff:
          self._active[env_index, leg_index] = True
          self._liftoff_knee_forward[env_index, leg_index] = knee_forward[env_index, leg_index]
          self._liftoff_knee_z[env_index, leg_index] = knee_relative_z[env_index, leg_index]
          self._liftoff_foot_forward[env_index, leg_index] = foot_forward[env_index, leg_index]
          self._liftoff_ground_z[env_index, leg_index] = self._ground_z[env_index, leg_index]
          trajectory.clear()
        if bool(self._active[env_index, leg_index]) and not current_contact:
          trajectory.append(
            _AirborneSample(
              knee_forward_m=float(knee_forward[env_index, leg_index].item()),
              knee_relative_z=float(knee_relative_z[env_index, leg_index].item()),
              foot_z=float(foot[env_index, leg_index, 2].item()),
            )
          )
        if landing:
          self.first_contact[env_index, leg_index] = True
          if bool(self._active[env_index, leg_index]) and trajectory:
            completed.append(self._complete(env_index, leg_index, trajectory, foot_forward))
          self._active[env_index, leg_index] = False
          trajectory.clear()
        if current_contact:
          self._ground_z[env_index, leg_index] = foot[env_index, leg_index, 2]
        self._previous_contact[env_index, leg_index] = current_contact
    return completed

  def _complete(
    self,
    env_index: int,
    leg_index: int,
    trajectory: list[_AirborneSample],
    foot_forward: torch.Tensor,
  ) -> CompletedStride:
    liftoff_knee_forward = float(self._liftoff_knee_forward[env_index, leg_index].item())
    advances = [sample.knee_forward_m - liftoff_knee_forward for sample in trajectory]
    peak_index = max(range(len(advances)), key=advances.__getitem__)
    knee_lift = max(sample.knee_relative_z for sample in trajectory) - float(
      self._liftoff_knee_z[env_index, leg_index].item()
    )
    clearance = max(sample.foot_z for sample in trajectory) - float(
      self._liftoff_ground_z[env_index, leg_index].item()
    )
    landing_reach = float(foot_forward[env_index, leg_index].item()) - float(
      self._liftoff_foot_forward[env_index, leg_index].item()
    )
    return CompletedStride(
      env_index=env_index,
      leg_index=leg_index,
      knee_forward_peak_m=advances[peak_index],
      knee_forward_peak_phase=peak_index / max(len(trajectory) - 1, 1),
      knee_lift_m=max(knee_lift, 0.0),
      foot_clearance_m=max(clearance, 0.0),
      landing_reach_m=landing_reach,
    )


_PROTOCOL_PATHS = (
  ("contract", "actor"), ("contract", "critic"), ("contract", "action"),
  ("evaluation", "protocol_version"), ("evaluation", "task_id"),
  ("evaluation", "seed"), ("evaluation", "num_envs"), ("evaluation", "device"),
  ("evaluation", "step_dt_seconds"), ("evaluation", "warmup_seconds"),
  ("evaluation", "warmup_steps"), ("evaluation", "measurement_seconds"),
  ("evaluation", "measurement_steps"), ("evaluation", "vx"),
  ("evaluation", "command", "twist_mps"),
  *(("evaluation", "command", "swing_height_m", height) for height in HEIGHTS),
  *(("evaluation", "command", "step_length_m", length) for length in LENGTHS),
)


def _nested_value(result: dict[str, object], path: tuple[str, ...]) -> object:
  value: object = result
  for key in path:
    if not isinstance(value, dict) or key not in value:
      raise ValueError(f"result is missing {'.'.join(path)}")
    value = value[key]
  return value


def _require_finite(value: object, field: str) -> None:
  if type(value) in (int, float):
    if not math.isfinite(float(value)):
      raise ValueError(f"result field {field} must be finite")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _require_finite(item, f"{field}[{index}]")


def _validate_protocol(baseline: dict[str, object], candidate: dict[str, object]) -> None:
  for path in _PROTOCOL_PATHS:
    field = ".".join(path)
    baseline_value = _nested_value(baseline, path)
    candidate_value = _nested_value(candidate, path)
    _require_finite(baseline_value, field)
    _require_finite(candidate_value, field)
    if baseline_value != candidate_value:
      raise ValueError(
        f"evaluation protocol mismatch at {field}: "
        f"baseline={baseline_value!r}, candidate={candidate_value!r}"
      )


def _stat_number(result: dict[str, object], *path: str) -> float:
  count = _nested_number(result, *path[:-1], "count")
  if not count.is_integer() or count < 1:
    raise ValueError(f"result field {'.'.join((*path[:-1], 'count'))} must be a positive count")
  return _nested_number(result, *path)

def _validate_combination_summaries(result: dict[str, object], result_name: str) -> None:
  expected_metrics = set(METRICS)
  expected_stats = {"count", "mean", "median", "p95", "min", "max"}
  for height in HEIGHTS:
    for length in LENGTHS:
      combination_path = ("combinations", height, length)
      combination = _nested_value(result, combination_path)
      if not isinstance(combination, dict) or set(combination) != expected_metrics:
        raise ValueError(
          f"{result_name} result {'.'.join(combination_path)} must contain exactly "
          f"{sorted(expected_metrics)}"
        )
      for metric in METRICS:
        summary_path = (*combination_path, metric)
        summary = combination[metric]
        if not isinstance(summary, dict) or set(summary) != expected_stats:
          raise ValueError(
            f"{result_name} result {'.'.join(summary_path)} must contain exactly "
            f"{sorted(expected_stats)}"
          )
        count = _nested_number(result, *summary_path, "count")
        if not count.is_integer() or count < 1:
          raise ValueError(
            f"{result_name} result {'.'.join((*summary_path, 'count'))} "
            "must be a positive count"
          )
        for statistic in ("mean", "median", "p95", "min", "max"):
          _nested_number(result, *summary_path, statistic)



def assess_step_length_candidate(
  baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
  _validate_protocol(baseline, candidate)
  _validate_combination_summaries(baseline, "baseline")
  _validate_combination_summaries(candidate, "candidate")
  checks: dict[str, dict[str, object]] = {}

  def add(name: str, observed: float, comparison: str, threshold: float, passed: bool) -> None:
    checks[name] = {
      "observed": observed,
      "comparison": comparison,
      "threshold": threshold,
      "passed": bool(passed),
    }

  for height in HEIGHTS:
    knee = [
      _stat_number(candidate, "combinations", height, length, "knee_forward_peak_m", "mean")
      for length in LENGTHS
    ]
    reach = [
      _stat_number(candidate, "combinations", height, length, "landing_reach_m", "mean")
      for length in LENGTHS
    ]
    retention = min(
      _stat_number(candidate, "combinations", height, length, "knee_lift_m", "mean")
      - _stat_number(baseline, "combinations", height, length, "knee_lift_m", "mean")
      for length in LENGTHS
    )
    adjacent = min(knee[1] - knee[0], knee[2] - knee[1])
    reach_span = reach[2] - reach[0]
    add(f"{height}_knee_forward_monotonic", min(knee[1] - knee[0], knee[2] - knee[1]), ">", 0.0, knee[0] < knee[1] < knee[2])
    add(f"{height}_knee_forward_adjacent_separation_min_m", adjacent, ">=", 0.025, adjacent >= 0.025)
    add(f"{height}_landing_reach_monotonic", min(reach[1] - reach[0], reach[2] - reach[1]), ">", 0.0, reach[0] < reach[1] < reach[2])
    add(f"{height}_landing_reach_long_short_separation_min_m", reach_span, ">=", 0.05, reach_span >= 0.05)
    add(f"{height}_knee_lift_retention_min_m", retention, ">=", -0.005, retention >= -0.005)

  baseline_velocity = _stat_number(baseline, "overall", "forward_velocity_abs_error_mps", "mean")
  baseline_force = _stat_number(baseline, "overall", "landing_force_n", "p95")
  baseline_falls = _nested_number(baseline, "overall", "falls")
  velocity = _stat_number(candidate, "overall", "forward_velocity_abs_error_mps", "mean")
  force = _stat_number(candidate, "overall", "landing_force_n", "p95")
  falls = _nested_number(candidate, "overall", "falls")
  for result_name, value in (("baseline", baseline_falls), ("candidate", falls)):
    if not value.is_integer() or value < 0:
      raise ValueError(
        f"{result_name} result field overall.falls must be a non-negative integer"
      )
  velocity_limit = baseline_velocity * 1.10
  force_limit = baseline_force * 1.15
  add("forward_velocity_error_max_mps", velocity, "<=", velocity_limit, velocity <= velocity_limit)
  add("falls_max", falls, "<=", baseline_falls, falls <= baseline_falls)
  add("landing_force_p95_max_n", force, "<=", force_limit, force <= force_limit)
  return {
    "status": "passed" if all(check["passed"] for check in checks.values()) else "failed",
    "checks": checks,
    "baseline_checkpoint": _checkpoint_metadata(baseline),
    "candidate_checkpoint": _checkpoint_metadata(candidate),
  }


def _set_commands_and_observe(env: Any, slices: dict[str, dict[str, slice]], vx: float) -> Any:
  from mjlab.tasks.velocity.mdp import UniformVelocityCommand

  from src.tasks.velocity.mdp import (
    DiscreteStepLengthCommand,
    DiscreteSwingHeightCommand,
  )

  manager = env.unwrapped.command_manager
  twist = manager.get_term("twist")
  height_term = manager.get_term("swing_height")
  length_term = manager.get_term("step_length")
  if not isinstance(twist, UniformVelocityCommand):
    raise TypeError("twist command is not UniformVelocityCommand")
  if not isinstance(height_term, DiscreteSwingHeightCommand):
    raise TypeError("swing_height command is not DiscreteSwingHeightCommand")
  if not isinstance(length_term, DiscreteStepLengthCommand):
    raise TypeError("step_length command is not DiscreteStepLengthCommand")
  twist.vel_command_b[:, 0] = vx
  twist.vel_command_b[:, 1:] = 0.0
  twist.is_heading_env.zero_()
  twist.is_standing_env.zero_()
  for height, height_target in zip(HEIGHTS, HEIGHT_TARGETS, strict=True):
    for length, length_target in zip(LENGTHS, LENGTH_TARGETS, strict=True):
      group = slices[height][length]
      height_term.height_command[group, 0] = height_target
      length_term.length_command[group, 0] = length_target
  return env.get_observations()


def _empty_samples() -> dict[str, list[float]]:
  return {metric: [] for metric in METRICS}


def _combination_for_env(
  env_index: int, slices: dict[str, dict[str, slice]]
) -> tuple[str, str]:
  for height in HEIGHTS:
    for length in LENGTHS:
      group = slices[height][length]
      if group.start <= env_index < group.stop:
        return height, length
  raise RuntimeError(f"environment index {env_index} is not assigned")


def run_evaluation(cfg: StepLengthEvaluationConfig) -> dict[str, object]:
  slices = equal_combination_slices(cfg.num_envs)
  if not math.isfinite(cfg.vx):
    raise ValueError("vx must be finite")
  if not math.isfinite(cfg.warmup_seconds) or cfg.warmup_seconds < 5.0:
    raise ValueError("warmup_seconds must be finite and at least 5")
  if not math.isfinite(cfg.measurement_seconds) or cfg.measurement_seconds < 30.0:
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
  env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), cfg.warmup_seconds + cfg.measurement_seconds + 5.0)
  twist_cfg = env_cfg.commands["twist"]
  twist_cfg.heading_command = False
  twist_cfg.ranges.heading = None
  twist_cfg.rel_heading_envs = 0.0
  twist_cfg.rel_standing_envs = 0.0
  twist_cfg.init_velocity_prob = 0.0
  twist_cfg.ranges.lin_vel_x = (cfg.vx, cfg.vx)
  twist_cfg.ranges.lin_vel_y = (0.0, 0.0)
  twist_cfg.ranges.ang_vel_z = (0.0, 0.0)
  for command_name in ("twist", "swing_height", "step_length"):
    env_cfg.commands[command_name].resampling_time_range = (1_000_000.0, 1_000_000.0)
  if hasattr(env_cfg.commands["swing_height"], "log_transitions"):
    env_cfg.commands["swing_height"].log_transitions = False

  env: Any | None = None
  try:
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    actor_dim = unwrapped.observation_manager.group_obs_dim.get("actor")
    critic_dim = unwrapped.observation_manager.group_obs_dim.get("critic")
    action_dim = unwrapped.action_manager.total_action_dim
    expected = ((100,), (302,), 29)
    if (actor_dim, critic_dim, action_dim) != expected:
      raise RuntimeError(
        "policy contract mismatch: expected actor=(100,), critic=(302,), "
        f"action=29; got actor={actor_dim}, critic={critic_dim}, action={action_dim}"
      )
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["feet_ground_contact"]
    knee_ids = _resolve_pair(robot.find_bodies, ("left_knee_link", "right_knee_link"), "knee body")
    foot_ids = _resolve_pair(robot.find_sites, ("left_foot", "right_foot"), "foot site")
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    env.reset()
    step_dt = float(unwrapped.step_dt)
    warmup_steps = math.ceil(cfg.warmup_seconds / step_dt)
    measurement_steps = math.ceil(cfg.measurement_seconds / step_dt)

    with torch.inference_mode():
      for _ in range(warmup_steps):
        env.step(policy(_set_commands_and_observe(env, slices, cfg.vx)))
      tracker = StrideTrajectoryAccumulator(cfg.num_envs, device)
      found = contact_sensor.data.found
      if found is None or found.shape != (cfg.num_envs, 2):
        raise RuntimeError(f"contact found must have shape ({cfg.num_envs}, 2)")
      tracker.observe(
        in_contact=found > 0,
        root_position_w=robot.data.root_link_pos_w,
        root_quaternion_w=robot.data.root_link_quat_w,
        knee_position_w=robot.data.body_link_pos_w[:, knee_ids],
        foot_position_w=robot.data.site_pos_w[:, foot_ids],
      )
      samples = {height: {length: _empty_samples() for length in LENGTHS} for height in HEIGHTS}
      velocity_samples: list[float] = []
      landing_force_samples: list[float] = []
      falls = 0
      for _ in range(measurement_steps):
        env.step(policy(_set_commands_and_observe(env, slices, cfg.vx)))
        terminated = unwrapped.reset_terminated.to(dtype=torch.bool)
        tracker.discard(terminated)
        valid = ~terminated
        falls += int(terminated.sum().item())
        velocity_samples.extend(
          torch.abs(robot.data.root_link_lin_vel_b[valid, 0] - cfg.vx).cpu().tolist()
        )
        found = contact_sensor.data.found
        force = contact_sensor.data.force
        if found is None or found.shape != (cfg.num_envs, 2):
          raise RuntimeError(f"contact found must have shape ({cfg.num_envs}, 2)")
        if force is None or force.shape != (cfg.num_envs, 2, 3):
          raise RuntimeError(f"contact force must have shape ({cfg.num_envs}, 2, 3)")
        completed = tracker.observe(
          in_contact=found > 0,
          root_position_w=robot.data.root_link_pos_w,
          root_quaternion_w=robot.data.root_link_quat_w,
          knee_position_w=robot.data.body_link_pos_w[:, knee_ids],
          foot_position_w=robot.data.site_pos_w[:, foot_ids],
          valid_environments=valid,
        )
        landing_force_samples.extend(
          torch.linalg.vector_norm(force, dim=-1)[tracker.first_contact].cpu().tolist()
        )
        for stride in completed:
          height, length = _combination_for_env(stride.env_index, slices)
          for metric in METRICS:
            samples[height][length][metric].append(float(getattr(stride, metric)))

    combinations = {
      height: {
        length: {metric: summarize(samples[height][length][metric]) for metric in METRICS}
        for length in LENGTHS
      }
      for height in HEIGHTS
    }
    result: dict[str, object] = {
      "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
      "contract": {"actor": 100, "critic": 302, "action": 29},
      "evaluation": {
        "protocol_version": "g1-step-length-v1",
        "task_id": cfg.task_id,
        "seed": cfg.seed,
        "num_envs": cfg.num_envs,
        "device": device,
        "step_dt_seconds": step_dt,
        "warmup_seconds": cfg.warmup_seconds,
        "warmup_steps": warmup_steps,
        "measurement_seconds": cfg.measurement_seconds,
        "measurement_steps": measurement_steps,
        "vx": cfg.vx,
        "command": {
          "twist_mps": [cfg.vx, 0.0, 0.0],
          "swing_height_m": dict(zip(HEIGHTS, HEIGHT_TARGETS, strict=True)),
          "step_length_m": dict(zip(LENGTHS, LENGTH_TARGETS, strict=True)),
        },
      },
      "combinations": combinations,
      "overall": {
        "forward_velocity_abs_error_mps": summarize(velocity_samples),
        "landing_force_n": summarize(landing_force_samples),
        "falls": falls,
      },
    }
    if baseline_path is not None:
      with baseline_path.open(encoding="utf-8") as baseline_file:
        baseline = json.load(baseline_file)
      if not isinstance(baseline, dict):
        raise ValueError("baseline result must contain a JSON object")
      result["decision"] = assess_step_length_candidate(cast(dict[str, object], baseline), result)
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
  num_envs: int = 54
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
  run_evaluation(StepLengthEvaluationConfig(task_id=task_id, **asdict(args)))


if __name__ == "__main__":
  main()
