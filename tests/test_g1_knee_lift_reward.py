from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp.knee_lift_reward import CommandedKneeLift


class FakeKneeContactSensor:
  def __init__(self, num_envs: int) -> None:
    self.data = SimpleNamespace(
      found=torch.ones(num_envs, 2, dtype=torch.bool),
      current_air_time=torch.zeros(num_envs, 2),
    )
    self.first_air = torch.zeros(num_envs, 2, dtype=torch.bool)
    self.first_contact = torch.zeros(num_envs, 2, dtype=torch.bool)
    self._slots = [
      SimpleNamespace(primary_name=name)
      for name in (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
      )
    ]

  def compute_first_air(self, dt: float) -> torch.Tensor:
    del dt
    return self.first_air

  def compute_first_contact(self, dt: float) -> torch.Tensor:
    del dt
    return self.first_contact

  def clear_events(self) -> None:
    self.first_air.zero_()
    self.first_contact.zero_()


HEIGHT_LEVELS = (0.05, 0.10, 0.15)
KNEE_TARGETS = (0.025, 0.040, 0.055)


class FakeCommandManager:
  def __init__(self, commands: dict[str, torch.Tensor]) -> None:
    self.commands = commands

  def get_command(self, name: str) -> torch.Tensor:
    return self.commands[name]


def entity_cfgs() -> tuple[SceneEntityCfg, SceneEntityCfg, SceneEntityCfg]:
  return (
    SceneEntityCfg(
      "robot",
      body_names=("left_knee_link", "right_knee_link"),
      body_ids=[0, 1],
      preserve_order=True,
    ),
    SceneEntityCfg(
      "robot",
      joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
      joint_ids=[0, 2],
      preserve_order=True,
    ),
    SceneEntityCfg(
      "robot",
      joint_names=("left_knee_joint", "right_knee_joint"),
      joint_ids=[1, 3],
      preserve_order=True,
    ),
  )


def reward_params(**overrides: object) -> dict[str, object]:
  knee_body_cfg, hip_joint_cfg, knee_joint_cfg = entity_cfgs()
  params: dict[str, object] = {
    "sensor_name": "feet_ground_contact",
    "height_command_name": "swing_height",
    "height_levels": HEIGHT_LEVELS,
    "knee_lift_targets": KNEE_TARGETS,
    "command_name": "twist",
    "command_threshold": 0.1,
    "nominal_swing_time_s": 0.28,
    "tracking_window": (0.20, 0.65),
    "knee_body_cfg": knee_body_cfg,
    "hip_joint_cfg": hip_joint_cfg,
    "knee_joint_cfg": knee_joint_cfg,
  }
  params.update(overrides)
  return params


def knee_fixture(
  *,
  height_commands: tuple[float, ...] = (0.10,),
  config_overrides: dict[str, object] | None = None,
) -> tuple[SimpleNamespace, CommandedKneeLift]:
  num_envs = len(height_commands)
  root_pos = torch.zeros(num_envs, 3)
  root_pos[:, 2] = 1.0
  knee_pos = torch.zeros(num_envs, 2, 3)
  knee_pos[:, :, 2] = 0.6
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=root_pos,
      body_link_pos_w=knee_pos,
      joint_pos=torch.zeros(num_envs, 4),
    )
  )
  contact = FakeKneeContactSensor(num_envs)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    scene={"robot": robot, "feet_ground_contact": contact},
    command_manager=FakeCommandManager(
      {
        "twist": torch.tensor(((0.5, 0.0, 0.0),) * num_envs),
        "swing_height": torch.tensor(height_commands).view(num_envs, 1),
      }
    ),
    extras={"log": {}},
  )
  params = reward_params(**(config_overrides or {}))
  cfg = RewardTermCfg(func=CommandedKneeLift, weight=-5.0, params=params)
  return env, CommandedKneeLift(cfg, env)


def evaluate(
  term: CommandedKneeLift, env: SimpleNamespace, **overrides: object
) -> torch.Tensor:
  return term(env, **reward_params(**overrides))


def start_left_swing(
  term: CommandedKneeLift,
  env: SimpleNamespace,
  *,
  air_time: float = 0.02,
) -> None:
  contact = env.scene["feet_ground_contact"]
  contact.clear_events()
  contact.data.found[:, 0] = True
  contact.data.current_air_time[:, 0] = 0.0
  torch.testing.assert_close(evaluate(term, env), torch.zeros(env.num_envs))
  contact.clear_events()
  contact.data.found[:, 0] = False
  contact.data.current_air_time[:, 0] = air_time
  contact.first_air[:, 0] = True
  torch.testing.assert_close(evaluate(term, env), torch.zeros(env.num_envs))
  contact.clear_events()


def set_left_relative_knee_lift(
  env: SimpleNamespace,
  lift: float | torch.Tensor,
  *,
  root_z: float = 1.0,
) -> None:
  lift_tensor = torch.as_tensor(lift, dtype=torch.float32).flatten()
  if lift_tensor.numel() == 1:
    lift_tensor = lift_tensor.expand(env.num_envs)
  assert lift_tensor.shape == (env.num_envs,)
  robot = env.scene["robot"]
  robot.data.root_link_pos_w[:, 2] = root_z
  robot.data.body_link_pos_w[:, 0, 2] = root_z - 0.4 + lift_tensor
  robot.data.body_link_pos_w[:, 1, 2] = root_z - 0.4


def set_air_time(env: SimpleNamespace, value: float) -> None:
  env.scene["feet_ground_contact"].data.current_air_time[:, 0] = value


def land_left(env: SimpleNamespace) -> None:
  contact = env.scene["feet_ground_contact"]
  contact.clear_events()
  contact.data.found[:, 0] = True
  contact.data.current_air_time[:, 0] = 0.0
  contact.first_contact[:, 0] = True


def test_knee_lift_tracks_root_relative_peak_only_inside_window_and_scores_once():
  env, term = knee_fixture(height_commands=(0.10,))
  start_left_swing(term, env, air_time=0.02)

  set_left_relative_knee_lift(env, 0.08, root_z=1.20)
  set_air_time(env, 0.04)
  assert evaluate(term, env).item() == 0.0

  set_left_relative_knee_lift(env, 0.04, root_z=0.95)
  set_air_time(env, 0.14)
  env.scene["robot"].data.joint_pos[0, [0, 1]] = torch.tensor((-0.4, 1.2))
  assert evaluate(term, env).item() == 0.0

  set_left_relative_knee_lift(env, 0.05, root_z=1.10)
  set_air_time(env, 0.21)
  assert evaluate(term, env).item() == 0.0

  land_left(env)
  assert evaluate(term, env).item() == pytest.approx(0.0)
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == pytest.approx(0.0)


def test_knee_lift_latches_level_and_requires_forward_command_at_liftoff():
  env, term = knee_fixture(height_commands=(0.10, 0.15))
  env.command_manager.commands["twist"] = torch.tensor(
    ((0.5, 0.3, 0.2), (0.0, 0.5, 0.5))
  )
  start_left_swing(term, env, air_time=0.02)
  env.command_manager.commands["swing_height"][:] = 0.05
  set_left_relative_knee_lift(env, 0.02)
  set_air_time(env, 0.14)
  evaluate(term, env)
  land_left(env)
  cost = evaluate(term, env)
  assert cost.tolist() == pytest.approx((0.25, 0.0), abs=1e-6)


def test_knee_lift_logs_peak_phase_and_midpoint_angles_by_latched_level():
  env, term = knee_fixture(height_commands=(0.05, 0.10, 0.15))
  start_left_swing(term, env, air_time=0.02)
  set_left_relative_knee_lift(env, torch.tensor((0.025, 0.040, 0.055)))
  set_air_time(env, 0.14)
  env.scene["robot"].data.joint_pos[:, 0] = torch.tensor((-0.2, -0.3, -0.4))
  env.scene["robot"].data.joint_pos[:, 1] = torch.tensor((0.8, 1.0, 1.2))
  evaluate(term, env)
  land_left(env)
  evaluate(term, env)

  log = env.extras["log"]
  assert log["Metrics/knee_lift_peak_low_m"].tolist() == pytest.approx((0.025,))
  assert log["Metrics/knee_lift_peak_mid_m"].tolist() == pytest.approx((0.040,))
  assert log["Metrics/knee_lift_peak_high_m"].tolist() == pytest.approx((0.055,))
  assert log["Metrics/knee_lift_peak_phase_low"].tolist() == pytest.approx((0.5,))
  assert log["Metrics/knee_mid_swing_hip_pitch_high_rad"].tolist() == pytest.approx((-0.4,))
  assert log["Metrics/knee_mid_swing_flexion_high_rad"].tolist() == pytest.approx((1.2,))
  assert log["Metrics/knee_lift_normalized_error"].tolist() == pytest.approx((0.0, 0.0, 0.0))


def test_knee_lift_reset_discards_only_selected_in_progress_swings():
  env, term = knee_fixture(height_commands=(0.10, 0.10))
  start_left_swing(term, env)
  set_left_relative_knee_lift(env, 0.02)
  set_air_time(env, 0.14)
  evaluate(term, env)

  term.reset(torch.tensor((0,)))
  land_left(env)

  assert evaluate(term, env).tolist() == pytest.approx((0.0, 0.25), abs=1e-6)


def test_knee_lift_requires_episode_contact_before_scoring_a_swing() -> None:
  env, term = knee_fixture()
  contact = env.scene["feet_ground_contact"]
  term.reset(slice(None))

  contact.data.found[:, 0] = False
  contact.data.current_air_time[:, 0] = 0.02
  contact.first_air[:, 0] = True
  assert evaluate(term, env).item() == 0.0
  contact.clear_events()
  set_left_relative_knee_lift(env, 0.02)
  set_air_time(env, 0.14)
  assert evaluate(term, env).item() == 0.0
  land_left(env)
  assert evaluate(term, env).item() == 0.0
  assert all(value.numel() == 0 for value in env.extras["log"].values())

  contact.clear_events()
  set_left_relative_knee_lift(env, 0.0)
  start_left_swing(term, env)
  set_left_relative_knee_lift(env, 0.02)
  set_air_time(env, 0.14)
  assert evaluate(term, env).item() == 0.0
  land_left(env)
  assert evaluate(term, env).item() == pytest.approx(0.25, abs=1e-6)
  assert env.extras["log"]["Metrics/knee_lift_peak_mid_m"].tolist() == pytest.approx(
    (0.02,)
  )
  contact.clear_events()
  assert evaluate(term, env).item() == 0.0



def test_knee_lift_landing_clears_all_swing_state():
  env, term = knee_fixture()
  start_left_swing(term, env)
  set_left_relative_knee_lift(env, 0.02)
  set_air_time(env, 0.14)
  env.scene["robot"].data.joint_pos[0, [0, 1]] = torch.tensor((-0.4, 1.2))
  evaluate(term, env)
  land_left(env)
  evaluate(term, env)

  assert term._liftoff_relative_knee_z.tolist() == [[0.0, 0.0]]
  assert term._peak_knee_lift.tolist() == [[0.0, 0.0]]
  assert term._peak_phase.tolist() == [[0.0, 0.0]]
  assert term._latched_targets.tolist() == [[0.0, 0.0]]
  assert term._latched_level_indices.tolist() == [[-1, -1]]
  assert term._latched_active.tolist() == [[False, False]]
  assert term._reference_initialized.tolist() == [[False, False]]
  assert term._midpoint_sampled.tolist() == [[False, False]]
  assert term._mid_hip_pitch.tolist() == [[0.0, 0.0]]
  assert term._mid_knee_flexion.tolist() == [[0.0, 0.0]]
def test_knee_lift_rejects_non_unique_runtime_height_level():
  env, term = knee_fixture(
    height_commands=(0.0500005,),
    config_overrides={
      "height_levels": (0.05, 0.050001, 0.15),
      "knee_lift_targets": KNEE_TARGETS,
    },
  )
  with pytest.raises(ValueError, match="exactly one configured height level"):
    evaluate(
      term,
      env,
      height_levels=(0.05, 0.050001, 0.15),
      knee_lift_targets=KNEE_TARGETS,
    )


def test_knee_lift_rejects_mismatched_leg_shapes():
  env, term = knee_fixture()
  env.scene["feet_ground_contact"].data.current_air_time = torch.zeros(1, 1)
  with pytest.raises(ValueError, match=r"expected.*\(1, 2\)"):
    evaluate(term, env)


def test_knee_lift_rejects_reversed_contact_sensor_primaries() -> None:
  env, _ = knee_fixture()
  env.scene["feet_ground_contact"]._slots = [
    SimpleNamespace(primary_name=name)
    for name in ("right_ankle_roll_link", "left_ankle_roll_link")
  ]
  with pytest.raises(ValueError, match="left.*right"):
    CommandedKneeLift(
      RewardTermCfg(
        func=CommandedKneeLift,
        weight=-5.0,
        params=reward_params(),
      ),
      env,
    )


def test_knee_lift_rejects_reversed_knee_body_names() -> None:
  with pytest.raises(ValueError, match="left_knee_link"):
    knee_fixture(config_overrides={
      "knee_body_cfg": SceneEntityCfg(
        "robot",
        body_names=("right_knee_link", "left_knee_link"),
        body_ids=[0, 1],
        preserve_order=True,
      )
    })


def test_knee_lift_rejects_reversed_hip_joint_names() -> None:
  with pytest.raises(ValueError, match="left_hip_pitch_joint"):
    knee_fixture(config_overrides={
      "hip_joint_cfg": SceneEntityCfg(
        "robot",
        joint_names=("right_hip_pitch_joint", "left_hip_pitch_joint"),
        joint_ids=[0, 2],
        preserve_order=True,
      )
    })

def test_knee_lift_rejects_reversed_knee_joint_names() -> None:
  with pytest.raises(ValueError, match="left_knee_joint"):
    knee_fixture(config_overrides={
      "knee_joint_cfg": SceneEntityCfg(
        "robot",
        joint_names=("right_knee_joint", "left_knee_joint"),
        joint_ids=[1, 3],
        preserve_order=True,
      )
    })
INVALID_CONFIGS = (
  {"height_levels": (0.05, 0.10)},
  {"height_levels": (0.05, 0.05, 0.15)},
  {"knee_lift_targets": (0.025, float("nan"), 0.055)},
  {"knee_lift_targets": (0.040, 0.025, 0.055)},
  {"nominal_swing_time_s": 0.0},
  {"tracking_window": (0.65, 0.20)},
  {"tracking_window": (-0.01, 0.65)},
)


@pytest.mark.parametrize("overrides", INVALID_CONFIGS)
def test_knee_lift_rejects_invalid_configuration(
  overrides: dict[str, object],
) -> None:
  with pytest.raises(ValueError):
    knee_fixture(config_overrides=overrides)


@pytest.mark.parametrize(
  ("command_name", "bad_value", "message"),
  (
    ("swing_height", torch.tensor(((0.10, 0.10),)), "shape"),
    ("swing_height", torch.tensor(((float("nan"),),)), "finite"),
    ("twist", torch.tensor(((0.5, 0.0),)), "shape"),
    ("twist", torch.tensor(((float("inf"), 0.0, 0.0),)), "finite"),
  ),
)
def test_knee_lift_rejects_invalid_runtime_commands(
  command_name: str,
  bad_value: torch.Tensor,
  message: str,
) -> None:
  env, term = knee_fixture()
  env.command_manager.commands[command_name] = bad_value
  with pytest.raises(ValueError, match=message):
    evaluate(term, env)
