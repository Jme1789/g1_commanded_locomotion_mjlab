from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class StandingReplayScalarCommand(CommandTerm):
  """Switch one scalar command using the velocity term's standing mask."""

  cfg: StandingReplayScalarCommandCfg

  def __init__(
    self, cfg: StandingReplayScalarCommandCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self.scalar_command = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.scalar_command

  def _standing_mask(self) -> torch.Tensor:
    velocity_term = self._env.command_manager.get_term(
      self.cfg.velocity_command_name
    )
    return velocity_term.is_standing_env

  def _turning_mask(self) -> torch.Tensor:
    velocity_term = self._env.command_manager.get_term(
      self.cfg.velocity_command_name
    )
    return (
      torch.abs(velocity_term.command[:, 2])
      > self.cfg.turning_command_threshold
    )

  def _write_values(self, env_ids: torch.Tensor) -> None:
    standing = self._standing_mask()[env_ids]
    values = torch.where(
      standing,
      self.cfg.standing_value_m,
      self.cfg.moving_value_m,
    )
    if self.cfg.turning_value_m is not None:
      turning = self._turning_mask()[env_ids]
      values = torch.where(turning, self.cfg.turning_value_m, values)
    self.scalar_command[env_ids, 0] = values

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._write_values(env_ids)

  def _update_command(self) -> None:
    self._write_values(torch.arange(self.num_envs, device=self.device))

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class StandingReplayScalarCommandCfg(CommandTermCfg):
  """Scalar values paired with moving and standing velocity commands."""

  moving_value_m: float
  standing_value_m: float
  turning_value_m: float | None = None
  turning_command_threshold: float = 0.1
  velocity_command_name: str = "twist"

  def __post_init__(self) -> None:
    if self.turning_value_m is not None and not (
      math.isfinite(self.turning_value_m) and self.turning_value_m > 0.0
    ):
      raise ValueError("turning_value_m must be finite and positive")
    if not (
      math.isfinite(self.turning_command_threshold)
      and self.turning_command_threshold >= 0.0
    ):
      raise ValueError(
        "turning_command_threshold must be finite and non-negative"
      )

  def build(self, env: ManagerBasedRlEnv) -> StandingReplayScalarCommand:
    return StandingReplayScalarCommand(self, env)
