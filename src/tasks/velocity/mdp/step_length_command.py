from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class DiscreteStepLengthCommand(CommandTerm):
  """Sample one scalar step length from a fixed set of levels."""

  cfg: DiscreteStepLengthCommandCfg

  def __init__(self, cfg: DiscreteStepLengthCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.length_command = torch.zeros(self.num_envs, 1, device=self.device)
    self._levels = torch.tensor(cfg.levels_m, device=self.device, dtype=torch.float32)
    self._probabilities = torch.tensor(cfg.probabilities, device=self.device, dtype=torch.float32)
    self._ping_pong_indices = torch.tensor(
      (
        *range(len(cfg.levels_m)),
        *range(len(cfg.levels_m) - 2, 0, -1),
      ),
      device=self.device,
      dtype=torch.long,
    )
    self._last_logged_level_index: int | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.length_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "ping_pong":
      positions = self.command_counter[env_ids] % len(self._ping_pong_indices)
      level_indices = self._ping_pong_indices[positions]
    else:
      level_indices = torch.multinomial(
        self._probabilities, len(env_ids), replacement=True
      )
    self.length_command[env_ids, 0] = self._levels[level_indices]
    self._log_transition(env_ids, level_indices)

  def _log_transition(
    self, env_ids: torch.Tensor, level_indices: torch.Tensor
  ) -> None:
    if not self.cfg.log_transitions:
      return
    env_zero_positions = (env_ids == 0).nonzero().flatten()
    if len(env_zero_positions) == 0:
      return
    level_index = int(level_indices[env_zero_positions[0]].item())
    if level_index == self._last_logged_level_index:
      return

    self._last_logged_level_index = level_index
    labels = ("SHORT", "MEDIUM", "LONG")
    label = (
      labels[level_index]
      if len(self.cfg.levels_m) == 3
      else f"LEVEL_{level_index + 1}"
    )
    target_m = float(self._levels[level_index].item())
    print(
      f"[STEP_LENGTH] level={label:<6} target={target_m:.3f} m",
      flush=True,
    )

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class DiscreteStepLengthCommandCfg(CommandTermCfg):
  """Configuration for a discrete physical step-length command."""

  levels_m: tuple[float, ...]
  probabilities: tuple[float, ...]
  sampling_mode: Literal["random", "ping_pong"] = "random"
  log_transitions: bool = False

  def __post_init__(self) -> None:
    if self.sampling_mode not in ("random", "ping_pong"):
      raise ValueError("sampling_mode must be 'random' or 'ping_pong'")
    if not self.levels_m:
      raise ValueError("levels_m must contain at least one positive level")
    if not all(math.isfinite(value) and value > 0.0 for value in self.levels_m):
      raise ValueError("levels_m values must be finite and positive")
    if any(current >= following for current, following in zip(self.levels_m, self.levels_m[1:], strict=False)):
      raise ValueError("levels_m must be strictly increasing")
    if len(self.probabilities) != len(self.levels_m):
      raise ValueError("probabilities must have the same length as levels_m")
    if not all(math.isfinite(value) and value >= 0.0 for value in self.probabilities):
      raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1.0e-6):
      raise ValueError("probabilities must sum to 1")
    lower, upper = self.resampling_time_range
    if not (math.isfinite(lower) and math.isfinite(upper) and lower > 0.0 and upper >= lower):
      raise ValueError("resampling_time_range must be finite, positive, and ordered")

  def build(self, env: ManagerBasedRlEnv) -> DiscreteStepLengthCommand:
    return DiscreteStepLengthCommand(self, env)
