from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_h20_high_lift_env_cfg,
  unitree_g1_flat_h20_longer_step_env_cfg,
  unitree_g1_flat_h20_longer_step_stand_replay_env_cfg,
  unitree_g1_flat_h20_longer_step_stand_yaw_replay_env_cfg,
  unitree_g1_flat_three_height_knee_lift_env_cfg,
  unitree_g1_flat_three_height_peak_env_cfg,
  unitree_g1_flat_three_height_step_length_env_cfg,
  unitree_g1_h20_balance_curriculum_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_stairs_env_cfg,
  unitree_g1_stairs_three_height_env_cfg,
  unitree_g1_step_up_scene_env_cfg,
)
from .rl_cfg import (
  unitree_g1_flat_h20_high_lift_ppo_runner_cfg,
  unitree_g1_flat_h20_longer_step_ppo_runner_cfg,
  unitree_g1_flat_h20_longer_step_stand_replay_ppo_runner_cfg,
  unitree_g1_flat_h20_longer_step_stand_yaw_replay_ppo_runner_cfg,
  unitree_g1_flat_three_height_knee_lift_ppo_runner_cfg,
  unitree_g1_flat_three_height_peak_ppo_runner_cfg,
  unitree_g1_flat_three_height_step_length_ppo_runner_cfg,
  unitree_g1_h20_balance_curriculum_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
  unitree_g1_step_up_scene_ppo_runner_cfg,
  unitree_g1_three_height_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Stairs",
  env_cfg=unitree_g1_stairs_env_cfg(),
  play_env_cfg=unitree_g1_stairs_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Stairs-ThreeHeight",
  env_cfg=unitree_g1_stairs_three_height_env_cfg(),
  play_env_cfg=unitree_g1_stairs_three_height_env_cfg(play=True),
  rl_cfg=unitree_g1_three_height_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


register_mjlab_task(
  task_id="Unitree-G1-Flat-ThreeHeight",
  env_cfg=unitree_g1_flat_three_height_peak_env_cfg(),
  play_env_cfg=unitree_g1_flat_three_height_peak_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_three_height_peak_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-ThreeHeight-KneeLift",
  env_cfg=unitree_g1_flat_three_height_knee_lift_env_cfg(),
  play_env_cfg=unitree_g1_flat_three_height_knee_lift_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_three_height_knee_lift_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-G1-Flat-ThreeHeight-StepLength",
  env_cfg=unitree_g1_flat_three_height_step_length_env_cfg(),
  play_env_cfg=unitree_g1_flat_three_height_step_length_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_three_height_step_length_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-H20-HighLift",
  env_cfg=unitree_g1_flat_h20_high_lift_env_cfg(),
  play_env_cfg=unitree_g1_flat_h20_high_lift_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_h20_high_lift_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-H20-LongerStep",
  env_cfg=unitree_g1_flat_h20_longer_step_env_cfg(),
  play_env_cfg=unitree_g1_flat_h20_longer_step_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_h20_longer_step_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-H20-LongerStep-StandReplay",
  env_cfg=unitree_g1_flat_h20_longer_step_stand_replay_env_cfg(),
  play_env_cfg=unitree_g1_flat_h20_longer_step_stand_replay_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_h20_longer_step_stand_replay_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-H20-LongerStep-StandYawReplay",
  env_cfg=unitree_g1_flat_h20_longer_step_stand_yaw_replay_env_cfg(),
  play_env_cfg=unitree_g1_flat_h20_longer_step_stand_yaw_replay_env_cfg(play=True),
  rl_cfg=unitree_g1_flat_h20_longer_step_stand_yaw_replay_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-H20-BalanceCurriculum",
  env_cfg=unitree_g1_h20_balance_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_h20_balance_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_h20_balance_curriculum_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-StepUp-Scene-Experiment",
  env_cfg=unitree_g1_step_up_scene_env_cfg(),
  play_env_cfg=unitree_g1_step_up_scene_env_cfg(play=True),
  rl_cfg=unitree_g1_step_up_scene_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
