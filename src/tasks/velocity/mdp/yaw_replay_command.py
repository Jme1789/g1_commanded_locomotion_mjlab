from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class SymmetricYawReplayVelocityCommand(UniformVelocityCommand):
  """Replay standing, both yaw directions, and forward walking exclusively."""

  cfg: SymmetricYawReplayVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    samples = torch.rand(len(env_ids), device=self.device)
    stand_end = self.cfg.standing_fraction
    positive_yaw_end = stand_end + self.cfg.yaw_fraction_each
    negative_yaw_end = positive_yaw_end + self.cfg.yaw_fraction_each

    standing = samples < stand_end
    positive_yaw = (samples >= stand_end) & (samples < positive_yaw_end)
    negative_yaw = (samples >= positive_yaw_end) & (samples < negative_yaw_end)
    forward = samples >= negative_yaw_end

    self.vel_command_b[env_ids] = 0.0
    self.vel_command_b[env_ids[positive_yaw], 2] = self.cfg.yaw_speed_rad_s
    self.vel_command_b[env_ids[negative_yaw], 2] = -self.cfg.yaw_speed_rad_s
    self.vel_command_b[env_ids[forward], 0] = self.cfg.forward_velocity_mps
    self.is_standing_env[env_ids] = standing
    self.is_heading_env[env_ids] = False
    self.heading_target[env_ids] = 0.0


@dataclass(kw_only=True)
class SymmetricYawReplayVelocityCommandCfg(UniformVelocityCommandCfg):
  """Four-mode replay distribution for restoring symmetric in-place yaw."""

  forward_velocity_mps: float = 0.5
  yaw_speed_rad_s: float = 0.5
  standing_fraction: float = 0.20
  yaw_fraction_each: float = 0.10
  ranges: UniformVelocityCommandCfg.Ranges = field(
    default_factory=lambda: UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.5),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.5, 0.5),
      heading=None,
    )
  )

  def __post_init__(self) -> None:
    super().__post_init__()
    values = (self.forward_velocity_mps, self.yaw_speed_rad_s)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
      raise ValueError("forward velocity and yaw speed must be finite and positive")
    fractions = (self.standing_fraction, self.yaw_fraction_each)
    if not all(math.isfinite(value) and value >= 0.0 for value in fractions):
      raise ValueError("replay fractions must be finite and non-negative")
    if self.standing_fraction + 2.0 * self.yaw_fraction_each >= 1.0:
      raise ValueError("replay fractions must leave a positive forward fraction")
    if self.heading_command or self.ranges.heading is not None:
      raise ValueError("symmetric yaw replay does not use heading commands")
    if self.init_velocity_prob != 0.0:
      raise ValueError("symmetric yaw replay requires init_velocity_prob=0")

  def build(self, env: ManagerBasedRlEnv) -> SymmetricYawReplayVelocityCommand:
    return SymmetricYawReplayVelocityCommand(self, env)
