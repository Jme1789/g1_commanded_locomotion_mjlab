from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import Literal

import mjlab.terrains as terrain_gen
import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks

from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_three_height_peak_env_cfg,
  unitree_g1_stairs_env_cfg,
  unitree_g1_stairs_three_height_env_cfg,
)
from src.tasks.velocity.config.g1.rl_cfg import (
  unitree_g1_flat_three_height_peak_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
  unitree_g1_three_height_ppo_runner_cfg,
)
from src.tasks.velocity.mdp.rewards import (
  CommandedFeetClearance,
  CommandedFeetPeakClearance,
)
from src.tasks.velocity.mdp.swing_height_command import (
  DiscreteSwingHeightCommand,
  DiscreteSwingHeightCommandCfg,
)


def _height_command_cfg(
  *,
  levels_m: tuple[float, ...] = (0.05, 0.10, 0.15),
  probabilities: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3),
  sampling_mode: Literal["random", "ping_pong"] = "random",
  log_transitions: bool = False,
) -> DiscreteSwingHeightCommandCfg:
  return DiscreteSwingHeightCommandCfg(
    levels_m=levels_m,
    probabilities=probabilities,
    resampling_time_range=(3.0, 8.0),
    sampling_mode=sampling_mode,
    log_transitions=log_transitions,
  )


def test_discrete_height_command_returns_one_configured_level_per_env() -> None:
  env = SimpleNamespace(num_envs=256, device="cpu")
  command = DiscreteSwingHeightCommand(_height_command_cfg(), env)
  env_ids = torch.arange(env.num_envs)

  torch.manual_seed(7)
  command._resample_command(env_ids)

  assert command.command.shape == (env.num_envs, 1)
  levels = torch.tensor((0.05, 0.10, 0.15))
  matches_level = torch.isclose(command.command, levels.view(1, -1)).any(dim=1)
  assert torch.all(matches_level)
  assert len(torch.unique(command.command)) == 3


def test_ping_pong_height_command_starts_low_and_returns_through_middle() -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteSwingHeightCommand(
    _height_command_cfg(sampling_mode="ping_pong"), env
  )
  env_ids = torch.tensor((0,))

  observed = []
  for _ in range(5):
    command._resample(env_ids)
    observed.append(command.command[0, 0].item())

  assert observed == pytest.approx((0.05, 0.10, 0.15, 0.10, 0.05))


def test_ping_pong_height_command_restarts_low_after_reset() -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteSwingHeightCommand(
    _height_command_cfg(sampling_mode="ping_pong"), env
  )
  env_ids = torch.tensor((0,))
  command._resample(env_ids)
  command._resample(env_ids)

  command.reset(env_ids)

  assert command.command[0, 0].item() == pytest.approx(0.05)


def test_ping_pong_transition_log_reports_level_changes(
  capsys: pytest.CaptureFixture[str],
) -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteSwingHeightCommand(
    _height_command_cfg(sampling_mode="ping_pong", log_transitions=True), env
  )
  env_ids = torch.tensor((0,))

  for _ in range(5):
    command._resample(env_ids)

  assert capsys.readouterr().out.splitlines() == [
    "[SWING_HEIGHT] level=LOW  target=0.050 m",
    "[SWING_HEIGHT] level=MID  target=0.100 m",
    "[SWING_HEIGHT] level=HIGH target=0.150 m",
    "[SWING_HEIGHT] level=MID  target=0.100 m",
    "[SWING_HEIGHT] level=LOW  target=0.050 m",
  ]


def test_transition_log_does_not_repeat_the_same_level(
  capsys: pytest.CaptureFixture[str],
) -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteSwingHeightCommand(
    _height_command_cfg(sampling_mode="ping_pong", log_transitions=True), env
  )
  env_ids = torch.tensor((0,))

  command._resample(env_ids)
  command.reset(env_ids)

  assert capsys.readouterr().out.splitlines() == [
    "[SWING_HEIGHT] level=LOW  target=0.050 m"
  ]


def test_transition_log_reports_only_environment_zero(
  capsys: pytest.CaptureFixture[str],
) -> None:
  env = SimpleNamespace(num_envs=2, device="cpu")
  command = DiscreteSwingHeightCommand(
    _height_command_cfg(sampling_mode="ping_pong", log_transitions=True), env
  )

  command._resample(torch.tensor((1,)))

  assert capsys.readouterr().out == ""


def test_transition_log_is_silent_by_default(
  capsys: pytest.CaptureFixture[str],
) -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = DiscreteSwingHeightCommand(
    _height_command_cfg(sampling_mode="ping_pong"), env
  )

  command._resample(torch.tensor((0,)))

  assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
  ("levels_m", "probabilities", "message"),
  [
    ((0.10, 0.05), (0.5, 0.5), "strictly increasing"),
    ((0.0, 0.10), (0.5, 0.5), "positive"),
    ((0.05, 0.10), (1.0,), "same length"),
    ((0.05, 0.10), (0.6, -0.1), "non-negative"),
    ((0.05, 0.10), (0.6, 0.3), "sum to 1"),
  ],
)
def test_discrete_height_command_rejects_invalid_levels_or_probabilities(
  levels_m: tuple[float, ...],
  probabilities: tuple[float, ...],
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    _height_command_cfg(levels_m=levels_m, probabilities=probabilities)


def test_height_command_rejects_unknown_sampling_mode() -> None:
  with pytest.raises(ValueError, match="sampling_mode"):
    _height_command_cfg(sampling_mode="cyclic")  # type: ignore[arg-type]


class _CommandManager:
  def __init__(self, commands: dict[str, torch.Tensor]):
    self.commands = commands

  def get_command(self, name: str) -> torch.Tensor:
    return self.commands[name]


def _clearance_fixture() -> tuple[SimpleNamespace, CommandedFeetClearance]:
  num_envs = 2
  ground_z = torch.tensor((0.0, 1.0)).view(num_envs, 1)
  foot_pos = torch.zeros(num_envs, 2, 3)
  foot_pos[:, :, 2] = ground_z
  foot_vel = torch.zeros(num_envs, 2, 3)
  foot_vel[:, :, 0] = 1.0
  robot = SimpleNamespace(
    data=SimpleNamespace(site_pos_w=foot_pos, site_lin_vel_w=foot_vel)
  )
  contact = SimpleNamespace(data=SimpleNamespace(found=torch.ones(num_envs, 2)))
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={"robot": robot, "feet_ground_contact": contact},
    command_manager=_CommandManager(
      {
        "twist": torch.tensor(((0.5, 0.0, 0.0), (0.5, 0.0, 0.0))),
        "swing_height": torch.tensor(((0.10,), (0.10,))),
      }
    ),
    episode_length_buf=torch.ones(num_envs, dtype=torch.long),
  )
  cfg = RewardTermCfg(
    func=CommandedFeetClearance,
    weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_command_name": "swing_height",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )
  return env, CommandedFeetClearance(cfg, env)


def _evaluate_clearance(
  term: CommandedFeetClearance, env: SimpleNamespace
) -> torch.Tensor:
  return term(
    env,
    sensor_name="feet_ground_contact",
    height_command_name="swing_height",
    command_name="twist",
    command_threshold=0.1,
    asset_cfg=SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
  )


def test_commanded_clearance_is_relative_to_each_foots_last_contact_height() -> None:
  env, term = _clearance_fixture()
  torch.testing.assert_close(_evaluate_clearance(term, env), torch.zeros(2))

  env.scene["feet_ground_contact"].data.found.zero_()
  env.scene["robot"].data.site_pos_w[:, :, 2] += 0.10
  cost_at_target = _evaluate_clearance(term, env)

  torch.testing.assert_close(cost_at_target, torch.zeros(2))
  env.scene["robot"].data.site_pos_w[:, :, 2] -= 0.05
  cost_below_target = _evaluate_clearance(term, env)
  assert cost_below_target[0] > 0.0
  torch.testing.assert_close(cost_below_target[0], cost_below_target[1])


def test_height_command_alone_does_not_activate_clearance_objective() -> None:
  env, term = _clearance_fixture()
  _evaluate_clearance(term, env)
  env.scene["feet_ground_contact"].data.found.zero_()
  env.scene["robot"].data.site_pos_w[:, :, 2] += 0.03
  env.command_manager.commands["twist"].zero_()

  torch.testing.assert_close(_evaluate_clearance(term, env), torch.zeros(2))


class _PeakContactSensor:
  def __init__(self, num_envs: int) -> None:
    self.data = SimpleNamespace(found=torch.ones(num_envs, 2, dtype=torch.bool))
    self.first_air = torch.zeros(num_envs, 2, dtype=torch.bool)
    self.first_contact = torch.zeros(num_envs, 2, dtype=torch.bool)

  def compute_first_air(self, dt: float) -> torch.Tensor:
    del dt
    return self.first_air

  def compute_first_contact(self, dt: float) -> torch.Tensor:
    del dt
    return self.first_contact

  def clear_events(self) -> None:
    self.first_air.zero_()
    self.first_contact.zero_()


def _peak_fixture(
  *,
  height_commands: tuple[float, ...] = (0.10, 0.10),
  ground_heights: tuple[float, ...] = (0.0, 1.0),
) -> tuple[SimpleNamespace, CommandedFeetPeakClearance]:
  num_envs = len(height_commands)
  assert len(ground_heights) == num_envs
  ground_z = torch.tensor(ground_heights).view(num_envs, 1)
  foot_pos = torch.zeros(num_envs, 2, 3)
  foot_pos[:, :, 2] = ground_z
  robot = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=foot_pos,
      site_lin_vel_w=torch.zeros(num_envs, 2, 3),
    )
  )
  contact = _PeakContactSensor(num_envs)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    scene={"robot": robot, "feet_ground_contact": contact},
    command_manager=_CommandManager(
      {
        "twist": torch.tensor(((0.5, 0.0, 0.0),) * num_envs),
        "swing_height": torch.tensor(height_commands).view(num_envs, 1),
      }
    ),
    extras={"log": {}},
  )
  cfg = RewardTermCfg(
    func=CommandedFeetPeakClearance,
    weight=-10.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_command_name": "swing_height",
      "height_levels": (0.05, 0.10, 0.15),
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )
  return env, CommandedFeetPeakClearance(cfg, env)


def _evaluate_peak(
  term: CommandedFeetPeakClearance, env: SimpleNamespace
) -> torch.Tensor:
  return term(
    env,
    sensor_name="feet_ground_contact",
    height_command_name="swing_height",
    height_levels=(0.05, 0.10, 0.15),
    command_name="twist",
    command_threshold=0.1,
    asset_cfg=SceneEntityCfg(
      "robot", site_names=("left_foot", "right_foot")
    ),
  )


def _start_left_swing(
  term: CommandedFeetPeakClearance,
  env: SimpleNamespace,
  peak_heights: tuple[float, ...],
) -> torch.Tensor:
  contact = env.scene["feet_ground_contact"]
  ground = env.scene["robot"].data.site_pos_w[:, 1, 2].clone()
  contact.clear_events()
  contact.data.found[:, 0] = False
  contact.first_air[:, 0] = True
  env.scene["robot"].data.site_pos_w[:, 0, 2] = ground + torch.tensor(
    peak_heights
  )
  return _evaluate_peak(term, env)


def _land_left_foot(
  term: CommandedFeetPeakClearance, env: SimpleNamespace
) -> torch.Tensor:
  contact = env.scene["feet_ground_contact"]
  ground = env.scene["robot"].data.site_pos_w[:, 1, 2].clone()
  contact.clear_events()
  contact.data.found[:, 0] = True
  contact.first_contact[:, 0] = True
  env.scene["robot"].data.site_pos_w[:, 0, 2] = ground
  return _evaluate_peak(term, env)


def test_peak_reward_uses_relative_peak_with_zero_foot_xy_speed() -> None:
  env, term = _peak_fixture()
  torch.testing.assert_close(_evaluate_peak(term, env), torch.zeros(2))

  torch.testing.assert_close(
    _start_left_swing(term, env, (0.10, 0.10)), torch.zeros(2)
  )
  torch.testing.assert_close(_land_left_foot(term, env), torch.zeros(2))


def test_peak_reward_latches_target_and_motion_state_at_liftoff() -> None:
  env, term = _peak_fixture(height_commands=(0.10,), ground_heights=(0.0,))
  _evaluate_peak(term, env)

  _start_left_swing(term, env, (0.10,))
  env.command_manager.commands["swing_height"].fill_(0.15)
  env.command_manager.commands["twist"].zero_()
  env.scene["feet_ground_contact"].clear_events()
  torch.testing.assert_close(_evaluate_peak(term, env), torch.zeros(1))
  torch.testing.assert_close(_land_left_foot(term, env), torch.zeros(1))

  _start_left_swing(term, env, (0.05,))
  env.command_manager.commands["twist"][:, 0] = 0.5
  env.scene["feet_ground_contact"].clear_events()
  env.scene["robot"].data.site_pos_w[:, 0, 2] = 0.15
  _evaluate_peak(term, env)
  torch.testing.assert_close(_land_left_foot(term, env), torch.zeros(1))


def test_peak_reward_penalizes_only_landing_peak_error() -> None:
  env, term = _peak_fixture(height_commands=(0.10,), ground_heights=(0.0,))
  _evaluate_peak(term, env)

  torch.testing.assert_close(
    _start_left_swing(term, env, (0.05,)), torch.zeros(1)
  )
  env.scene["feet_ground_contact"].clear_events()
  torch.testing.assert_close(_evaluate_peak(term, env), torch.zeros(1))

  torch.testing.assert_close(_land_left_foot(term, env), torch.tensor((0.25,)))


def test_peak_reward_reset_discards_selected_in_progress_swing() -> None:
  env, term = _peak_fixture()
  _evaluate_peak(term, env)
  _start_left_swing(term, env, (0.05, 0.05))

  term.reset(torch.tensor((0,)))

  torch.testing.assert_close(
    _land_left_foot(term, env), torch.tensor((0.0, 0.25))
  )


def test_peak_reward_logs_real_landing_samples_by_latched_level() -> None:
  env, term = _peak_fixture(
    height_commands=(0.05, 0.10, 0.15),
    ground_heights=(0.0, 1.0, 2.0),
  )
  _evaluate_peak(term, env)
  _start_left_swing(term, env, (0.04, 0.09, 0.14))
  assert env.extras["log"]["Metrics/swing_peak_low_m"].numel() == 0
  assert env.extras["log"]["Metrics/swing_peak_mid_m"].numel() == 0
  assert env.extras["log"]["Metrics/swing_peak_high_m"].numel() == 0

  _land_left_foot(term, env)

  assert env.extras["log"]["Metrics/swing_peak_low_m"].tolist() == pytest.approx(
    (0.04,)
  )
  assert env.extras["log"]["Metrics/swing_peak_mid_m"].tolist() == pytest.approx(
    (0.09,)
  )
  assert env.extras["log"]["Metrics/swing_peak_high_m"].tolist() == pytest.approx(
    (0.14,)
  )
  assert env.extras["log"][
    "Metrics/swing_peak_normalized_error"
  ].tolist() == pytest.approx(
    (0.04, 0.01, (0.14 / 0.15 - 1.0) ** 2), abs=1e-6
  )


@pytest.mark.parametrize(
  "height_levels",
  [(), (0.05, 0.10), (0.05, 0.10, float("nan")), (0.05, 0.0, 0.15)],
)
def test_peak_reward_rejects_invalid_height_levels(
  height_levels: tuple[float, ...],
) -> None:
  env, _ = _peak_fixture()
  cfg = RewardTermCfg(
    func=CommandedFeetPeakClearance,
    weight=-10.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_command_name": "swing_height",
      "height_levels": height_levels,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )

  with pytest.raises(ValueError, match="height_levels"):
    CommandedFeetPeakClearance(cfg, env)

def test_three_height_task_is_additive_and_keeps_twist_unchanged() -> None:
  baseline = unitree_g1_stairs_env_cfg()
  candidate = unitree_g1_stairs_three_height_env_cfg()

  assert asdict(candidate.commands["twist"]) == asdict(baseline.commands["twist"])
  assert candidate.commands["swing_height"].levels_m == (0.05, 0.10, 0.15)
  assert candidate.commands["swing_height"].probabilities == (
    1 / 3,
    1 / 3,
    1 / 3,
  )
  expected_actor_terms = [
    name
    for name in baseline.observations["actor"].terms
    if name != "height_scan"
  ] + ["swing_height_command"]
  assert list(candidate.observations["actor"].terms) == expected_actor_terms
  assert list(candidate.observations["critic"].terms) == [
    *baseline.observations["critic"].terms,
    "swing_height_command",
  ]
  assert candidate.observations["actor"].terms[
    "swing_height_command"
  ].params == {"command_name": "swing_height"}
  assert candidate.observations["critic"].terms[
    "swing_height_command"
  ].params == {"command_name": "swing_height"}


def test_three_height_play_uses_ten_second_ping_pong_without_changing_training() -> None:
  training = unitree_g1_stairs_three_height_env_cfg(play=False)
  play = unitree_g1_stairs_three_height_env_cfg(play=True)

  assert training.commands["swing_height"].sampling_mode == "random"
  assert training.commands["swing_height"].resampling_time_range == (3.0, 8.0)
  assert training.commands["swing_height"].log_transitions is False
  assert play.commands["swing_height"].sampling_mode == "ping_pong"
  assert play.commands["swing_height"].resampling_time_range == (10.0, 10.0)
  assert play.commands["swing_height"].log_transitions is True


def test_three_height_task_changes_only_candidate_clearance_behavior() -> None:
  baseline = unitree_g1_stairs_env_cfg()
  candidate = unitree_g1_stairs_three_height_env_cfg()
  baseline_clearance = baseline.rewards["foot_clearance"]
  candidate_clearance = candidate.rewards["foot_clearance"]

  assert candidate_clearance.func is CommandedFeetClearance
  assert baseline_clearance.weight == -1.0
  assert candidate_clearance.weight == -5.0
  assert candidate_clearance.params == {
    "sensor_name": "feet_ground_contact",
    "height_command_name": "swing_height",
    "command_name": "twist",
    "command_threshold": baseline_clearance.params["command_threshold"],
    "asset_cfg": baseline_clearance.params["asset_cfg"],
  }
  assert "target_height" in baseline_clearance.params
  assert "height_scan" in baseline.observations["actor"].terms


def test_three_height_runner_keeps_ppo_and_uses_separate_log_namespace() -> None:
  baseline = unitree_g1_ppo_runner_cfg()
  candidate = unitree_g1_three_height_ppo_runner_cfg()

  assert asdict(candidate.actor) == asdict(baseline.actor)
  assert asdict(candidate.critic) == asdict(baseline.critic)
  assert asdict(candidate.algorithm) == asdict(baseline.algorithm)
  assert candidate.num_steps_per_env == baseline.num_steps_per_env
  assert candidate.max_iterations == baseline.max_iterations
  assert candidate.experiment_name == "g1_velocity_stairs_three_height"


def test_three_height_task_is_registered_without_replacing_baseline() -> None:
  tasks = set(list_tasks())
  assert "Unitree-G1-Stairs-ThreeHeight" in tasks
  assert "Unitree-G1-Stairs" in tasks


def test_flat_peak_task_keeps_three_height_policy_io_contract() -> None:
  stairs = unitree_g1_stairs_three_height_env_cfg()
  flat = unitree_g1_flat_three_height_peak_env_cfg()

  assert asdict(flat.commands["twist"]) == asdict(stairs.commands["twist"])
  assert asdict(flat.commands["swing_height"]) == asdict(
    stairs.commands["swing_height"]
  )
  assert list(flat.observations["actor"].terms) == list(
    stairs.observations["actor"].terms
  )
  assert list(flat.observations["critic"].terms) == list(
    stairs.observations["critic"].terms
  )
  assert asdict(flat.actions["joint_pos"]) == asdict(stairs.actions["joint_pos"])


def test_flat_peak_task_uses_only_generator_backed_flat_terrain() -> None:
  cfg = unitree_g1_flat_three_height_peak_env_cfg()
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None

  assert generator.curriculum is False
  assert generator.num_rows == 1
  assert generator.num_cols == 1
  assert cfg.scene.terrain.max_init_terrain_level == 0
  assert list(generator.sub_terrains) == ["flat"]
  assert isinstance(generator.sub_terrains["flat"], terrain_gen.BoxFlatTerrainCfg)
  assert generator.sub_terrains["flat"].proportion == 1.0
  assert "terrain_levels" not in cfg.curriculum
  assert "command_vel" in cfg.curriculum


def test_flat_peak_task_changes_only_clearance_reward_and_terrain() -> None:
  stairs = unitree_g1_stairs_three_height_env_cfg()
  flat = unitree_g1_flat_three_height_peak_env_cfg()

  assert stairs.rewards["foot_clearance"].func is CommandedFeetClearance
  assert stairs.rewards["foot_clearance"].weight == -5.0
  assert flat.rewards["foot_clearance"].func is CommandedFeetPeakClearance
  assert flat.rewards["foot_clearance"].weight == -10.0
  assert flat.rewards["foot_clearance"].params == {
    "sensor_name": "feet_ground_contact",
    "height_command_name": "swing_height",
    "height_levels": (0.05, 0.10, 0.15),
    "command_name": "twist",
    "command_threshold": stairs.rewards["foot_clearance"].params[
      "command_threshold"
    ],
    "asset_cfg": stairs.rewards["foot_clearance"].params["asset_cfg"],
  }
  for name, reward in stairs.rewards.items():
    if name != "foot_clearance":
      assert asdict(flat.rewards[name]) == asdict(reward)


def test_flat_peak_play_keeps_deterministic_height_cycle() -> None:
  play = unitree_g1_flat_three_height_peak_env_cfg(play=True)

  assert play.commands["swing_height"].sampling_mode == "ping_pong"
  assert play.commands["swing_height"].resampling_time_range == (10.0, 10.0)
  assert play.commands["swing_height"].log_transitions is True


def test_flat_peak_runner_reuses_ppo_in_separate_namespace() -> None:
  baseline = unitree_g1_three_height_ppo_runner_cfg()
  candidate = unitree_g1_flat_three_height_peak_ppo_runner_cfg()

  assert asdict(candidate.actor) == asdict(baseline.actor)
  assert asdict(candidate.critic) == asdict(baseline.critic)
  assert asdict(candidate.algorithm) == asdict(baseline.algorithm)
  assert candidate.num_steps_per_env == baseline.num_steps_per_env
  assert candidate.max_iterations == baseline.max_iterations
  assert candidate.experiment_name == "g1_velocity_flat_three_height_peak"


def test_flat_peak_task_is_registered_without_replacing_existing_tasks() -> None:
  tasks = set(list_tasks())
  assert "Unitree-G1-Flat-ThreeHeight" in tasks
  assert "Unitree-G1-Stairs-ThreeHeight" in tasks
