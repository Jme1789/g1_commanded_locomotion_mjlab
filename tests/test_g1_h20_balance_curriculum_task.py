from __future__ import annotations

from dataclasses import asdict

import mjlab.terrains as terrain_gen
from mjlab.tasks.registry import list_tasks

from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_h20_longer_step_stand_yaw_replay_env_cfg,
  unitree_g1_h20_balance_curriculum_env_cfg,
)
from src.tasks.velocity.config.g1.rl_cfg import (
  unitree_g1_flat_h20_longer_step_stand_yaw_replay_ppo_runner_cfg,
  unitree_g1_h20_balance_curriculum_ppo_runner_cfg,
)


def _scene_without_terrain(cfg) -> dict[str, object]:
  scene = asdict(cfg.scene)
  scene.pop("terrain")
  return scene


def test_balance_curriculum_changes_only_terrain_and_curriculum() -> None:
  baseline = unitree_g1_flat_h20_longer_step_stand_yaw_replay_env_cfg()
  candidate = unitree_g1_h20_balance_curriculum_env_cfg()

  candidate_scene = _scene_without_terrain(candidate)
  baseline_scene = _scene_without_terrain(baseline)
  assert candidate_scene.pop("num_envs") == 2048
  assert baseline_scene.pop("num_envs") == 1
  assert candidate_scene == baseline_scene
  assert candidate.commands == baseline.commands
  assert candidate.observations == baseline.observations
  assert candidate.actions == baseline.actions
  assert candidate.rewards == baseline.rewards
  assert candidate.events == baseline.events
  assert candidate.terminations == baseline.terminations
  assert candidate.metrics == baseline.metrics
  assert set(candidate.curriculum) == {"terrain_levels"}
  assert baseline.curriculum == {}


def test_balance_curriculum_uses_the_approved_training_terrain_mix() -> None:
  cfg = unitree_g1_h20_balance_curriculum_env_cfg()
  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None

  assert generator.seed == 42
  assert generator.size == (6.4, 6.4)
  assert generator.border_width == 1.0
  assert generator.num_rows == 4
  assert generator.num_cols == 10
  assert generator.difficulty_range == (0.0, 1.0)
  assert generator.curriculum is True
  assert terrain.max_init_terrain_level == 0
  assert list(generator.sub_terrains) == [
    "flat",
    "step_up_stairs",
    "random_rough",
  ]

  flat = generator.sub_terrains["flat"]
  assert isinstance(flat, terrain_gen.BoxFlatTerrainCfg)
  assert flat.proportion == 0.30

  stairs = generator.sub_terrains["step_up_stairs"]
  assert isinstance(stairs, terrain_gen.BoxInvertedPyramidStairsTerrainCfg)
  assert stairs.proportion == 0.50
  assert stairs.step_height_range == (0.01, 0.08)
  assert stairs.step_width == 0.30
  assert stairs.platform_width == 3.0
  assert stairs.border_width == 1.0

  rough = generator.sub_terrains["random_rough"]
  assert isinstance(rough, terrain_gen.HfRandomUniformTerrainCfg)
  assert rough.proportion == 0.20
  assert rough.noise_range == (0.005, 0.03)
  assert rough.noise_step == 0.005
  assert rough.border_width == 0.25


def test_balance_curriculum_play_uses_one_fixed_eight_centimetre_stair() -> None:
  cfg = unitree_g1_h20_balance_curriculum_env_cfg(play=True)
  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None

  assert generator.num_rows == 1
  assert generator.num_cols == 1
  assert generator.curriculum is False
  assert "terrain_levels" not in cfg.curriculum
  assert list(generator.sub_terrains) == ["step_up_stairs"]
  stairs = generator.sub_terrains["step_up_stairs"]
  assert isinstance(stairs, terrain_gen.BoxInvertedPyramidStairsTerrainCfg)
  assert stairs.proportion == 1.0
  assert stairs.step_height_range == (0.08, 0.08)
  assert stairs.step_width == 0.30


def test_balance_curriculum_preserves_policy_contract_and_registers_task() -> None:
  baseline = unitree_g1_flat_h20_longer_step_stand_yaw_replay_ppo_runner_cfg()
  candidate = unitree_g1_h20_balance_curriculum_ppo_runner_cfg()
  expected = asdict(baseline)
  expected.update(
    experiment_name="g1_velocity_h20_balance_curriculum",
    max_iterations=5001,
    run_name="h20-balance-curriculum-r0",
    logger="tensorboard",
  )

  assert asdict(candidate) == expected
  assert 3 + 3 + 3 + 2 + 29 + 29 + 29 + 1 + 1 == 100
  assert 100 + 187 + 3 + 2 + 2 + 2 + 6 == 302
  assert "Unitree-G1-H20-BalanceCurriculum" in set(list_tasks())


def test_balance_curriculum_owns_the_release_training_defaults() -> None:
  env_cfg = unitree_g1_h20_balance_curriculum_env_cfg()
  agent_cfg = unitree_g1_h20_balance_curriculum_ppo_runner_cfg()

  assert env_cfg.scene.num_envs == 2048
  assert agent_cfg.max_iterations == 5001
  assert agent_cfg.run_name == "h20-balance-curriculum-r0"
  assert agent_cfg.logger == "tensorboard"
