from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from src.tasks.velocity.mdp.step_length_command import (
  DiscreteStepLengthCommand,
  DiscreteStepLengthCommandCfg,
)


def _cfg(
  *,
  levels_m: tuple[float, ...] = (0.20, 0.30, 0.40),
  probabilities: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3),
  resampling_time_range: tuple[float, float] = (3.0, 8.0),
  sampling_mode: Literal["random", "ping_pong"] = "random",
  log_transitions: bool = False,
) -> DiscreteStepLengthCommandCfg:
  return DiscreteStepLengthCommandCfg(
    levels_m=levels_m,
    probabilities=probabilities,
    resampling_time_range=resampling_time_range,
    sampling_mode=sampling_mode,
    log_transitions=log_transitions,
  )


def test_step_length_command_samples_only_three_levels() -> None:
  env = SimpleNamespace(num_envs=900, device="cpu")
  term = DiscreteStepLengthCommand(_cfg(), env)
  torch.manual_seed(13)

  term._resample_command(torch.arange(env.num_envs))

  assert term.command.shape == (900, 1)
  allowed = torch.tensor((0.20, 0.30, 0.40))
  assert torch.all(torch.isclose(term.command, allowed.view(1, -1)).any(dim=1))
  assert len(torch.unique(term.command)) == 3


@pytest.mark.parametrize(
  ("levels_m", "probabilities", "resampling_time_range", "message"),
  [
    ((), (), (3.0, 8.0), "at least one"),
    ((0.20, float("nan")), (0.5, 0.5), (3.0, 8.0), "finite"),
    ((0.20, float("inf")), (0.5, 0.5), (3.0, 8.0), "finite"),
    ((0.0, 0.30), (0.5, 0.5), (3.0, 8.0), "positive"),
    ((0.30, 0.20), (0.5, 0.5), (3.0, 8.0), "strictly increasing"),
    ((0.20, 0.30), (0.5,), (3.0, 8.0), "same length"),
    ((0.20, 0.30), (-0.1, 1.1), (3.0, 8.0), "non-negative"),
    ((0.20, 0.30), (0.6, 0.3), (3.0, 8.0), "sum to 1"),
    ((0.20, 0.30), (0.5, 0.5), (float("nan"), 8.0), "finite"),
    ((0.20, 0.30), (0.5, 0.5), (0.0, 8.0), "positive"),
    ((0.20, 0.30), (0.5, 0.5), (8.0, 3.0), "ordered"),
  ],
)
def test_step_length_command_rejects_invalid_configuration(
  levels_m: tuple[float, ...],
  probabilities: tuple[float, ...],
  resampling_time_range: tuple[float, float],
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    _cfg(
      levels_m=levels_m,
      probabilities=probabilities,
      resampling_time_range=resampling_time_range,
    )


def test_step_length_command_rejects_invalid_sampling_mode() -> None:
  with pytest.raises(ValueError, match="sampling_mode"):
    _cfg(sampling_mode="unsupported")  # type: ignore[arg-type]


def test_step_length_command_resampling_does_not_change_twist() -> None:
  env = SimpleNamespace(num_envs=4, device="cpu")
  term = DiscreteStepLengthCommand(_cfg(), env)
  twist_before = term.command.clone()

  term.reset(torch.arange(env.num_envs))

  assert torch.equal(term.command, term.length_command)
  assert not hasattr(term, "twist")
  assert torch.equal(twist_before, torch.zeros(4, 1))


def test_ping_pong_step_length_starts_short_and_returns_through_medium() -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteStepLengthCommand(_cfg(sampling_mode="ping_pong"), env)
  env_ids = torch.tensor((0,))

  observed = []
  for _ in range(5):
    command._resample(env_ids)
    observed.append(command.command[0, 0].item())

  assert observed == pytest.approx((0.20, 0.30, 0.40, 0.30, 0.20))


def test_ping_pong_step_length_logs_only_level_changes(
  capsys: pytest.CaptureFixture[str],
) -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteStepLengthCommand(
    _cfg(sampling_mode="ping_pong", log_transitions=True), env
  )
  env_ids = torch.tensor((0,))

  for _ in range(4):
    command._resample(env_ids)

  assert capsys.readouterr().out.splitlines() == [
    "[STEP_LENGTH] level=SHORT  target=0.200 m",
    "[STEP_LENGTH] level=MEDIUM target=0.300 m",
    "[STEP_LENGTH] level=LONG   target=0.400 m",
    "[STEP_LENGTH] level=MEDIUM target=0.300 m",
  ]


def test_ping_pong_step_length_log_does_not_repeat_same_level_after_reset(
  capsys: pytest.CaptureFixture[str],
) -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteStepLengthCommand(
    _cfg(sampling_mode="ping_pong", log_transitions=True), env
  )
  env_ids = torch.tensor((0,))

  command._resample(env_ids)
  command.reset(env_ids)

  assert capsys.readouterr().out.splitlines() == [
    "[STEP_LENGTH] level=SHORT  target=0.200 m"
  ]


def test_random_step_length_is_silent_by_default(
  capsys: pytest.CaptureFixture[str],
) -> None:
  env = SimpleNamespace(num_envs=4, device="cpu")
  command = DiscreteStepLengthCommand(_cfg(), env)

  command._resample(torch.arange(env.num_envs))

  assert capsys.readouterr().out == ""
