"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_g1_three_height_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the G1 PPO baseline under a separate experiment namespace."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_stairs_three_height"
  return cfg


def unitree_g1_flat_three_height_peak_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the three-height PPO config under the flat peak-reward namespace."""
  cfg = unitree_g1_three_height_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_three_height_peak"
  return cfg


def unitree_g1_flat_three_height_knee_lift_ppo_runner_cfg(
) -> RslRlOnPolicyRunnerCfg:
  """Reuse the validated PPO config under the knee-lift namespace."""
  cfg = unitree_g1_flat_three_height_peak_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_three_height_knee_lift"
  return cfg


def unitree_g1_flat_three_height_step_length_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the knee-lift PPO config under the step-length namespace."""
  cfg = unitree_g1_flat_three_height_knee_lift_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_three_height_step_length"
  return cfg


def unitree_g1_flat_h20_high_lift_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the validated PPO config under the flat H20 namespace."""
  cfg = unitree_g1_flat_three_height_step_length_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_h20_high_lift"
  return cfg


def unitree_g1_flat_h20_longer_step_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the validated H20 PPO config under the longer-step namespace."""
  cfg = unitree_g1_flat_h20_high_lift_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_h20_longer_step"
  return cfg


def unitree_g1_flat_h20_longer_step_stand_replay_ppo_runner_cfg(
) -> RslRlOnPolicyRunnerCfg:
  """Reuse the LongerStep PPO config under the stand-replay namespace."""
  cfg = unitree_g1_flat_h20_longer_step_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_h20_longer_step_stand_replay"
  return cfg


def unitree_g1_flat_h20_longer_step_stand_yaw_replay_ppo_runner_cfg(
) -> RslRlOnPolicyRunnerCfg:
  """Reuse the StandReplay PPO config under the symmetric-yaw namespace."""
  cfg = unitree_g1_flat_h20_longer_step_stand_replay_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_h20_longer_step_stand_yaw_replay"
  return cfg


def unitree_g1_h20_balance_curriculum_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the stand/yaw PPO config under the balance curriculum namespace."""
  cfg = unitree_g1_flat_h20_longer_step_stand_yaw_replay_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_h20_balance_curriculum"
  cfg.max_iterations = 5001
  cfg.run_name = "h20-balance-curriculum-r0"
  cfg.logger = "tensorboard"
  return cfg


def unitree_g1_step_up_scene_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the validated step-length PPO config for the scene experiment."""
  cfg = unitree_g1_flat_three_height_step_length_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_step_up_scene_experiment"
  return cfg
