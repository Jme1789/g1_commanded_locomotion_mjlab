"""Unitree G1 velocity environment configurations."""

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.tasks.velocity.mdp import (
  CommandedFeetClearance,
  CommandedFeetPeakClearance,
  CommandedKneeForward,
  CommandedKneeLift,
  DiscreteStepLengthCommandCfg,
  DiscreteSwingHeightCommandCfg,
  StandingReplayScalarCommandCfg,
  SymmetricYawReplayVelocityCommandCfg,
)
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if (
      cfg.scene.terrain is not None
      and cfg.scene.terrain.terrain_generator is not None
    ):
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_g1_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 stairs-focused velocity configuration.

  Builds on the rough-terrain config but reshapes the terrain generator to
  emphasize climbing/descending stairs at realistic step heights, while keeping
  enough flat/rough patches so the policy still walks naturally on flat ground.

  Key differences vs. `unitree_g1_rough_env_cfg`:
    - Stairs (up + down pyramids) make up 60% of terrain, with step height up to
      0.18 m (real-world stair height) instead of 0.10 m. The terrain curriculum
      interpolates step height from 0 -> max by difficulty, so the robot starts
      on shallow steps and progresses to tall ones.
    - Higher foot-clearance target (0.15 m) so feet clear taller step edges.
    - Slightly wider steps (0.32 m run) for a more human-like stair gait.
  """
  cfg = unitree_g1_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  tg = cfg.scene.terrain.terrain_generator

  # Reshape terrain mix to emphasize stairs while retaining flat + rough patches.
  # Step height is interpolated 0 -> max by the terrain-level curriculum.
  tg.sub_terrains = {
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.15),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.30,
      step_height_range=(0.0, 0.18),
      step_width=0.32,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.30,
      step_height_range=(0.0, 0.18),
      step_width=0.32,
      platform_width=3.0,
      border_width=1.0,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.15,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.10,
      slope_range=(0.0, 1.0),
      platform_width=2.0,
      border_width=0.25,
    ),
  }

  # Taller steps need higher foot clearance so the swing foot clears the edge.
  cfg.rewards["foot_clearance"].params["target_height"] = 0.15

  # Cap commanded speed for stairs. Climbing 0.18m steps at 2 m/s is physically
  # infeasible, and the base velocity curriculum (which ramps lin_vel_x up to
  # 2.0) would otherwise make the robot fail the terrain-level distance check and
  # get demoted to easier terrain. Keep the top speed at 1.0 m/s so terrain
  # levels can rise instead of collapsing.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)

  # Override the velocity curriculum so stage 2 does not push speed back to 2.0.
  cmd_vel_cur = cfg.curriculum.get("command_vel")
  if cmd_vel_cur is not None:
    cmd_vel_cur.params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
      {"step": 5000 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5)},
    ]

  return cfg


def unitree_g1_stairs_three_height_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a deployable G1 stairs task with three swing-height commands."""
  cfg = unitree_g1_stairs_env_cfg(play=play)
  swing_height_cmd = DiscreteSwingHeightCommandCfg(
    levels_m=(0.05, 0.10, 0.15),
    probabilities=(1 / 3, 1 / 3, 1 / 3),
    resampling_time_range=(3.0, 8.0),
  )
  if play:
    swing_height_cmd.sampling_mode = "ping_pong"
    swing_height_cmd.resampling_time_range = (10.0, 10.0)
    swing_height_cmd.log_transitions = True
  cfg.commands["swing_height"] = swing_height_cmd

  del cfg.observations["actor"].terms["height_scan"]
  cfg.observations["actor"].terms["swing_height_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "swing_height"},
  )
  cfg.observations["critic"].terms["swing_height_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "swing_height"},
  )

  baseline_clearance = cfg.rewards["foot_clearance"]
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=CommandedFeetClearance,
    weight=-5.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_command_name": "swing_height",
      "command_name": "twist",
      "command_threshold": baseline_clearance.params["command_threshold"],
      "asset_cfg": baseline_clearance.params["asset_cfg"],
    },
  )
  return cfg


def unitree_g1_flat_three_height_peak_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the checkpoint-compatible three-height task on flat terrain."""
  cfg = unitree_g1_stairs_three_height_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  terrain_generator = cfg.scene.terrain.terrain_generator
  terrain_generator.curriculum = False
  terrain_generator.num_rows = 1
  terrain_generator.num_cols = 1
  terrain_generator.sub_terrains = {
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0)
  }
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.curriculum.pop("terrain_levels", None)

  previous_clearance = cfg.rewards["foot_clearance"]
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=CommandedFeetPeakClearance,
    weight=-10.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_command_name": "swing_height",
      "height_levels": (0.05, 0.10, 0.15),
      "command_name": "twist",
      "command_threshold": previous_clearance.params["command_threshold"],
      "asset_cfg": previous_clearance.params["asset_cfg"],
    },
  )
  return cfg


def unitree_g1_flat_three_height_knee_lift_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Add commanded knee lift without changing the validated policy contract."""
  cfg = unitree_g1_flat_three_height_peak_env_cfg(play=play)
  cfg.rewards["commanded_knee_lift"] = RewardTermCfg(
    func=CommandedKneeLift,
    weight=-5.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_command_name": "swing_height",
      "height_levels": (0.05, 0.10, 0.15),
      "knee_lift_targets": (0.025, 0.040, 0.055),
      "command_name": "twist",
      "command_threshold": 0.1,
      "nominal_swing_time_s": 0.28,
      "tracking_window": (0.20, 0.65),
      "knee_body_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_knee_link", "right_knee_link"),
        preserve_order=True,
      ),
      "hip_joint_cfg": SceneEntityCfg(
        "robot",
        joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
        preserve_order=True,
      ),
      "knee_joint_cfg": SceneEntityCfg(
        "robot",
        joint_names=("left_knee_joint", "right_knee_joint"),
        preserve_order=True,
      ),
    },
  )
  return cfg


def unitree_g1_flat_three_height_step_length_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Add a step-length command to the validated knee-lift task."""
  cfg = unitree_g1_flat_three_height_knee_lift_env_cfg(play=play)
  step_length_command = DiscreteStepLengthCommandCfg(
    levels_m=(0.20, 0.30, 0.40),
    probabilities=(1 / 3, 1 / 3, 1 / 3),
    resampling_time_range=(3.0, 8.0),
  )
  if play:
    cfg.commands["swing_height"] = DiscreteSwingHeightCommandCfg(
      levels_m=(0.10,),
      probabilities=(1.0,),
      resampling_time_range=(10.0, 10.0),
      sampling_mode="ping_pong",
      log_transitions=True,
    )
    step_length_command.resampling_time_range = (10.0, 10.0)
    step_length_command.sampling_mode = "ping_pong"
    step_length_command.log_transitions = True
  cfg.commands["step_length"] = step_length_command
  for group in ("actor", "critic"):
    cfg.observations[group].terms["step_length_command"] = ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "step_length"},
    )
  cfg.rewards["commanded_knee_forward"] = RewardTermCfg(
    func=CommandedKneeForward,
    weight=-5.0,
    params={
      "sensor_name": "feet_ground_contact",
      "step_length_command_name": "step_length",
      "step_length_levels": (0.20, 0.30, 0.40),
      "knee_forward_targets": (0.10, 0.15, 0.20),
      "command_name": "twist",
      "command_threshold": 0.1,
      "nominal_swing_time_s": 0.28,
      "tracking_window": (0.30, 0.80),
      "knee_body_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_knee_link", "right_knee_link"),
        preserve_order=True,
      ),
      "foot_site_cfg": SceneEntityCfg(
        "robot",
        site_names=("left_foot", "right_foot"),
        preserve_order=True,
      ),
    },
  )
  return cfg


def unitree_g1_flat_h20_high_lift_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Raise only the fixed HIGH swing-height target on validated flat terrain."""
  cfg = unitree_g1_flat_three_height_step_length_env_cfg(play=play)

  twist_command = cfg.commands["twist"]
  assert isinstance(twist_command, UniformVelocityCommandCfg)
  twist_command.rel_standing_envs = 0.0
  twist_command.heading_command = False
  twist_command.ranges.lin_vel_x = (0.5, 0.5)
  twist_command.ranges.lin_vel_y = (0.0, 0.0)
  twist_command.ranges.ang_vel_z = (0.0, 0.0)
  twist_command.ranges.heading = None

  cfg.commands["swing_height"] = DiscreteSwingHeightCommandCfg(
    levels_m=(0.20,),
    probabilities=(1.0,),
    resampling_time_range=(3.0, 8.0),
  )
  cfg.commands["step_length"] = DiscreteStepLengthCommandCfg(
    levels_m=(0.40,),
    probabilities=(1.0,),
    resampling_time_range=(3.0, 8.0),
  )
  cfg.curriculum.pop("command_vel", None)

  cfg.rewards["foot_clearance"].params["height_levels"] = (0.05, 0.10, 0.20)
  knee_lift = cfg.rewards["commanded_knee_lift"]
  knee_lift.params["height_levels"] = (0.05, 0.10, 0.20)
  knee_lift.params["knee_lift_targets"] = (0.025, 0.040, 0.075)
  return cfg


def unitree_g1_flat_h20_longer_step_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Increase only the LONG knee-forward target over the validated H20 task."""
  cfg = unitree_g1_flat_h20_high_lift_env_cfg(play=play)
  knee_forward = cfg.rewards["commanded_knee_forward"]
  knee_forward.params["knee_forward_targets"] = (0.10, 0.15, 0.25)
  return cfg


def unitree_g1_flat_h20_longer_step_stand_replay_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Replay the deployed no-input tuple without changing the moving task."""
  cfg = unitree_g1_flat_h20_longer_step_env_cfg(play=play)

  twist_command = cfg.commands["twist"]
  assert isinstance(twist_command, UniformVelocityCommandCfg)
  twist_command.rel_standing_envs = 0.20

  cfg.commands["swing_height"] = StandingReplayScalarCommandCfg(
    moving_value_m=0.20,
    standing_value_m=0.10,
    velocity_command_name="twist",
    resampling_time_range=(3.0, 8.0),
  )
  cfg.commands["step_length"] = StandingReplayScalarCommandCfg(
    moving_value_m=0.40,
    standing_value_m=0.30,
    velocity_command_name="twist",
    resampling_time_range=(3.0, 8.0),
  )
  return cfg


def unitree_g1_flat_h20_longer_step_stand_yaw_replay_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Replay standing, symmetric in-place yaw, and forward walking."""
  cfg = unitree_g1_flat_h20_longer_step_stand_replay_env_cfg(play=play)

  previous_twist = cfg.commands["twist"]
  assert isinstance(previous_twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = SymmetricYawReplayVelocityCommandCfg(
    entity_name=previous_twist.entity_name,
    forward_velocity_mps=0.5,
    yaw_speed_rad_s=0.5,
    standing_fraction=0.20,
    yaw_fraction_each=0.10,
    resampling_time_range=previous_twist.resampling_time_range,
    debug_vis=previous_twist.debug_vis,
    viz=previous_twist.viz,
  )

  swing_height = cfg.commands["swing_height"]
  assert isinstance(swing_height, StandingReplayScalarCommandCfg)
  swing_height.turning_value_m = 0.10

  step_length = cfg.commands["step_length"]
  assert isinstance(step_length, StandingReplayScalarCommandCfg)
  step_length.turning_value_m = 0.30
  return cfg


def unitree_g1_h20_balance_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Restore stair and rough-terrain balance without changing policy inputs."""
  cfg = unitree_g1_flat_h20_longer_step_stand_yaw_replay_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = 2048

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  terrain_generator = cfg.scene.terrain.terrain_generator
  terrain_generator.seed = 42
  terrain_generator.size = (6.4, 6.4)
  terrain_generator.border_width = 1.0
  terrain_generator.difficulty_range = (0.0, 1.0)
  cfg.scene.terrain.max_init_terrain_level = 0

  if play:
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.curriculum = False
    terrain_generator.sub_terrains = {
      "step_up_stairs": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.08, 0.08),
        step_width=0.30,
        platform_width=3.0,
        border_width=1.0,
      )
    }
    cfg.curriculum.pop("terrain_levels", None)
    return cfg

  terrain_generator.num_rows = 4
  terrain_generator.num_cols = 10
  terrain_generator.curriculum = True
  terrain_generator.sub_terrains = {
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.30),
    "step_up_stairs": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.50,
      step_height_range=(0.01, 0.08),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.20,
      noise_range=(0.005, 0.03),
      noise_step=0.005,
      border_width=0.25,
    ),
  }
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_vel,
    params={"command_name": "twist"},
  )
  return cfg


def unitree_g1_step_up_scene_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Specialize the validated height/length policy for ascending stairs."""
  cfg = unitree_g1_flat_three_height_step_length_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  terrain_generator = cfg.scene.terrain.terrain_generator
  terrain_generator.seed = 42
  terrain_generator.size = (6.4, 6.4)
  terrain_generator.border_width = 1.0
  terrain_generator.num_rows = 1 if play else 4
  terrain_generator.num_cols = 1
  terrain_generator.difficulty_range = (0.0, 1.0)
  terrain_generator.curriculum = not play
  step_height_range = (0.08, 0.08) if play else (0.01, 0.09)
  terrain_generator.sub_terrains = {
    "step_up_stairs": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      size=terrain_generator.size,
      step_height_range=step_height_range,
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    )
  }
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.curriculum.pop("command_vel", None)
  if play:
    cfg.curriculum.pop("terrain_levels", None)
  else:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    )

  twist_command = cfg.commands["twist"]
  assert isinstance(twist_command, UniformVelocityCommandCfg)
  twist_command.rel_standing_envs = 0.0
  twist_command.heading_command = False
  twist_command.ranges.lin_vel_x = (0.5, 0.5)
  twist_command.ranges.lin_vel_y = (0.0, 0.0)
  twist_command.ranges.ang_vel_z = (0.0, 0.0)
  twist_command.ranges.heading = None
  cfg.commands["swing_height"] = DiscreteSwingHeightCommandCfg(
    levels_m=(0.15,),
    probabilities=(1.0,),
    resampling_time_range=(3.0, 8.0),
  )
  cfg.commands["step_length"] = DiscreteStepLengthCommandCfg(
    levels_m=(0.40,),
    probabilities=(1.0,),
    resampling_time_range=(3.0, 8.0),
  )
  return cfg
