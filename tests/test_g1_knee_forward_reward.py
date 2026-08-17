import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp import CommandedKneeForward

STEP_LENGTH_LEVELS = (0.20, 0.30, 0.40)
KNEE_FORWARD_TARGETS = (0.10, 0.15, 0.20)


class FakeContactSensor:
  def __init__(self, num_envs: int) -> None:
    self.data = SimpleNamespace(
      found=torch.ones(num_envs, 2, dtype=torch.bool),
      current_air_time=torch.zeros(num_envs, 2),
    )
    self.first_air = torch.zeros(num_envs, 2, dtype=torch.bool)
    self.first_contact = torch.zeros(num_envs, 2, dtype=torch.bool)
    self._slots = [
      SimpleNamespace(primary_name=name)
      for name in ("left_ankle_roll_link", "right_ankle_roll_link")
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


class FakeCommandManager:
  def __init__(self, commands: dict[str, torch.Tensor]) -> None:
    self.commands = commands

  def get_command(self, name: str) -> torch.Tensor:
    return self.commands[name]


def entity_cfgs() -> tuple[SceneEntityCfg, SceneEntityCfg]:
  return (
    SceneEntityCfg(
      "robot",
      body_names=("left_knee_link", "right_knee_link"),
      body_ids=[0, 1],
      preserve_order=True,
    ),
    SceneEntityCfg(
      "robot",
      site_names=("left_foot", "right_foot"),
      site_ids=[0, 1],
      preserve_order=True,
    ),
  )


def reward_params(**overrides: object) -> dict[str, object]:
  knee_body_cfg, foot_site_cfg = entity_cfgs()
  params: dict[str, object] = {
    "sensor_name": "feet_ground_contact",
    "step_length_command_name": "step_length",
    "step_length_levels": STEP_LENGTH_LEVELS,
    "knee_forward_targets": KNEE_FORWARD_TARGETS,
    "command_name": "twist",
    "command_threshold": 0.1,
    "nominal_swing_time_s": 1.0,
    "tracking_window": (0.30, 0.80),
    "knee_body_cfg": knee_body_cfg,
    "foot_site_cfg": foot_site_cfg,
  }
  params.update(overrides)
  return params


def fixture(
  *,
  step_lengths: tuple[float, ...] = (0.20,),
  config_overrides: dict[str, object] | None = None,
) -> tuple[SimpleNamespace, CommandedKneeForward]:
  num_envs = len(step_lengths)
  root_pos = torch.zeros(num_envs, 3)
  root_quat = torch.zeros(num_envs, 4)
  root_quat[:, 0] = 1.0
  knee_pos = torch.zeros(num_envs, 2, 3)
  knee_pos[:, :, 2] = 0.5
  foot_pos = torch.zeros(num_envs, 2, 3)
  foot_pos[:, :, 2] = 0.05
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=root_pos,
      root_link_quat_w=root_quat,
      body_link_pos_w=knee_pos,
      site_pos_w=foot_pos,
    )
  )
  contact = FakeContactSensor(num_envs)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    reset_ids=torch.arange(num_envs),
    scene={"robot": robot, "feet_ground_contact": contact},
    command_manager=FakeCommandManager(
      {
        "twist": torch.tensor(((0.5, 0.0, 0.0),) * num_envs),
        "step_length": torch.tensor(step_lengths).view(num_envs, 1),
      }
    ),
    extras={"log": {}},
  )
  params = reward_params(**(config_overrides or {}))
  cfg = RewardTermCfg(func=CommandedKneeForward, weight=-5.0, params=params)
  return env, CommandedKneeForward(cfg, env)


def evaluate(
  term: CommandedKneeForward, env: SimpleNamespace, **overrides: object
) -> torch.Tensor:
  return term(env, **reward_params(**overrides))


def establish_contact(term: CommandedKneeForward, env: SimpleNamespace) -> None:
  contact = env.scene["feet_ground_contact"]
  contact.clear_events()
  contact.data.found.fill_(True)
  contact.data.current_air_time.zero_()
  torch.testing.assert_close(evaluate(term, env), torch.zeros(env.num_envs))


def liftoff(
  term: CommandedKneeForward,
  env: SimpleNamespace,
  *,
  legs: tuple[int, ...] = (0,),
) -> None:
  contact = env.scene["feet_ground_contact"]
  contact.clear_events()
  contact.data.current_air_time.zero_()
  for leg in legs:
    contact.data.found[:, leg] = False
    contact.data.current_air_time[:, leg] = 0.02
    contact.first_air[:, leg] = True
  torch.testing.assert_close(evaluate(term, env), torch.zeros(env.num_envs))
  contact.clear_events()


def set_progress(env: SimpleNamespace, progress: float, *, leg: int = 0) -> None:
  env.scene["feet_ground_contact"].data.current_air_time[:, leg] = progress


def set_heading_pose(
  env: SimpleNamespace,
  *,
  knee_forward: float | tuple[float, ...] | None = None,
  foot_forward: float | tuple[float, ...] | None = None,
  leg: int = 0,
  yaw: float | tuple[float, ...] = 0.0,
  root_xy: tuple[float, float] = (0.0, 0.0),
) -> None:
  num_envs = env.num_envs
  yaw_values = torch.as_tensor(yaw, dtype=torch.float32).flatten()
  if yaw_values.numel() == 1:
    yaw_values = yaw_values.expand(num_envs)
  assert yaw_values.shape == (num_envs,)

  robot = env.scene["robot"]
  robot.data.root_link_pos_w[:, :2] = torch.tensor(root_xy)
  robot.data.root_link_quat_w.zero_()
  robot.data.root_link_quat_w[:, 0] = torch.cos(yaw_values / 2.0)
  robot.data.root_link_quat_w[:, 3] = torch.sin(yaw_values / 2.0)

  def set_point(field: str, value: float | tuple[float, ...] | None) -> None:
    if value is None:
      return
    forward = torch.as_tensor(value, dtype=torch.float32).flatten()
    if forward.numel() == 1:
      forward = forward.expand(num_envs)
    assert forward.shape == (num_envs,)
    points = getattr(robot.data, field)
    points[:, leg, 0] = robot.data.root_link_pos_w[:, 0] + forward * torch.cos(
      yaw_values
    )
    points[:, leg, 1] = robot.data.root_link_pos_w[:, 1] + forward * torch.sin(
      yaw_values
    )

  set_point("body_link_pos_w", knee_forward)
  set_point("site_pos_w", foot_forward)


def land(
  env: SimpleNamespace,
  *,
  legs: tuple[int, ...] = (0,),
) -> None:
  contact = env.scene["feet_ground_contact"]
  contact.clear_events()
  for leg in legs:
    contact.data.found[:, leg] = True
    contact.data.current_air_time[:, leg] = 0.0
    contact.first_contact[:, leg] = True


def record_peak(
  term: CommandedKneeForward,
  env: SimpleNamespace,
  values: tuple[float, ...] | float,
  *,
  progress: float = 0.5,
  leg: int = 0,
) -> None:
  set_heading_pose(env, knee_forward=values, leg=leg)
  set_progress(env, progress, leg=leg)
  torch.testing.assert_close(evaluate(term, env), torch.zeros(env.num_envs))


def test_requires_episode_contact_before_liftoff_can_score() -> None:
  env, term = fixture()
  term.reset(slice(None))
  contact = env.scene["feet_ground_contact"]
  contact.data.found[:, 0] = False
  contact.data.current_air_time[:, 0] = 0.02
  contact.first_air[:, 0] = True
  assert evaluate(term, env).item() == 0.0

  contact.clear_events()
  record_peak(term, env, 0.10)
  set_heading_pose(env, foot_forward=0.20)
  land(env)
  assert evaluate(term, env).item() == 0.0
  assert all(value.numel() == 0 for value in env.extras["log"].values())


def test_forward_threshold_is_the_only_activation_gate() -> None:
  env, term = fixture(step_lengths=(0.20, 0.40))
  env.command_manager.commands["twist"] = torch.tensor(
    ((0.1, 2.0, 2.0), (-0.5, 2.0, 2.0))
  )
  establish_contact(term, env)
  liftoff(term, env)
  record_peak(term, env, (0.10, 0.20))
  land(env)
  assert evaluate(term, env).tolist() == [0.0, 0.0]
  assert all(value.numel() == 0 for value in env.extras["log"].values())


def test_heading_frame_removes_root_translation_and_common_yaw() -> None:
  env, term = fixture(step_lengths=(0.20, 0.40))
  establish_contact(term, env)
  liftoff(term, env)
  set_heading_pose(
    env,
    knee_forward=(0.10, 0.10),
    yaw=math.pi / 2,
    root_xy=(4.0, -3.0),
  )
  set_progress(env, 0.5)
  assert evaluate(term, env).tolist() == [0.0, 0.0]
  set_heading_pose(
    env,
    foot_forward=(0.20, -0.10),
    yaw=math.pi / 2,
    root_xy=(4.0, -3.0),
  )
  land(env)
  torch.testing.assert_close(evaluate(term, env), torch.tensor((0.0, 0.25)))
  env.scene["feet_ground_contact"].clear_events()
  torch.testing.assert_close(evaluate(term, env), torch.zeros(env.num_envs))
  assert env.extras["log"]["Metrics/landing_reach_short_m"].tolist() == pytest.approx(
    (0.20,)
  )
  assert env.extras["log"]["Metrics/landing_reach_long_m"].tolist() == pytest.approx(
    (-0.10,)
  )


def test_step_length_target_is_latched_at_liftoff() -> None:
  env, term = fixture(step_lengths=(0.20, 0.40))
  establish_contact(term, env)
  liftoff(term, env)
  env.command_manager.commands["step_length"][:] = torch.tensor(((0.40,), (0.20,)))
  record_peak(term, env, (0.10, 0.10))
  land(env)
  assert evaluate(term, env).tolist() == pytest.approx((0.0, 0.25), abs=1e-6)


def test_only_tracking_window_including_boundaries_updates_peak() -> None:
  env, term = fixture(step_lengths=(0.30,))
  establish_contact(term, env)
  liftoff(term, env)

  record_peak(term, env, 0.25, progress=0.29)
  record_peak(term, env, 0.10, progress=0.30)
  record_peak(term, env, 0.08, progress=0.50)
  record_peak(term, env, 0.15, progress=0.80)
  record_peak(term, env, 0.30, progress=0.81)
  land(env)

  assert evaluate(term, env).item() == pytest.approx(0.0, abs=1e-6)
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0
  log = env.extras["log"]
  assert log["Metrics/knee_forward_peak_medium_m"].tolist() == pytest.approx((0.15,))
  assert log["Metrics/knee_forward_peak_phase_medium"].tolist() == pytest.approx((0.80,))


@pytest.mark.parametrize(
  ("peaks", "expected"),
  (
    ((0.10, 0.15, 0.20), (0.0, 0.0, 0.0)),
    ((0.05, 0.30, 0.30), (0.25, 1.0, 0.25)),
  ),
)
def test_normalized_square_error_uses_target_for_each_level(
  peaks: tuple[float, ...], expected: tuple[float, ...]
) -> None:
  env, term = fixture(step_lengths=STEP_LENGTH_LEVELS)
  establish_contact(term, env)
  liftoff(term, env)
  record_peak(term, env, peaks)
  land(env)
  assert evaluate(term, env).tolist() == pytest.approx(expected, abs=1e-6)
  env.scene["feet_ground_contact"].clear_events()
  torch.testing.assert_close(evaluate(term, env), torch.zeros(env.num_envs))
  assert env.extras["log"]["Metrics/knee_forward_normalized_error"].tolist() == (
    pytest.approx(expected, abs=1e-6)
  )


def test_landing_scores_once_logs_diagnostics_and_clears_only_landed_leg() -> None:
  env, term = fixture(step_lengths=(0.20,))
  establish_contact(term, env)
  liftoff(term, env, legs=(0, 1))
  record_peak(term, env, 0.05, progress=0.5, leg=0)
  record_peak(term, env, 0.10, progress=0.6, leg=1)
  set_heading_pose(env, foot_forward=-0.04, leg=0)
  set_heading_pose(env, foot_forward=0.12, leg=1)

  land(env, legs=(0,))
  assert evaluate(term, env).item() == pytest.approx(0.25, abs=1e-6)
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0
  log = env.extras["log"]
  assert log["Metrics/knee_forward_peak_short_m"].tolist() == pytest.approx((0.05,))
  assert log["Metrics/knee_forward_peak_phase_short"].tolist() == pytest.approx((0.5,))
  assert log["Metrics/landing_reach_short_m"].tolist() == pytest.approx((-0.04,))
  for state in (
    term._liftoff_knee_forward,
    term._liftoff_foot_forward,
    term._peak_knee_forward,
    term._peak_phase,
    term._latched_targets,
    term._sampled_landing_reach,
  ):
    assert state[0, 0].item() == 0.0
  assert term._latched_level_indices.tolist() == [[-1, 0]]
  assert term._latched_active.tolist() == [[False, True]]
  assert term._reference_initialized.tolist() == [[False, True]]
  assert term._peak_knee_forward[0, 1].item() == pytest.approx(0.10)
  assert term._peak_phase[0, 1].item() == pytest.approx(0.60)
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0

  land(env, legs=(1,))
  assert evaluate(term, env).item() == pytest.approx(0.0, abs=1e-6)
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0
  assert log["Metrics/knee_forward_peak_short_m"].tolist() == pytest.approx(
    (0.05, 0.10)
  )
  assert log["Metrics/knee_forward_peak_phase_short"].tolist() == pytest.approx(
    (0.50, 0.60)
  )
  assert log["Metrics/landing_reach_short_m"].tolist() == pytest.approx(
    (-0.04, 0.12)
  )
  for state in (
    term._liftoff_knee_forward,
    term._liftoff_foot_forward,
    term._peak_knee_forward,
    term._peak_phase,
    term._latched_targets,
    term._sampled_landing_reach,
  ):
    torch.testing.assert_close(state, torch.zeros_like(state))
  assert term._latched_level_indices.tolist() == [[-1, -1]]
  assert term._latched_active.tolist() == [[False, False]]
  assert term._reference_initialized.tolist() == [[False, False]]
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0


def test_scored_metrics_survive_reset_and_append_without_empty_overwrite() -> None:
  env, term = fixture(step_lengths=(0.40,))
  establish_contact(term, env)
  liftoff(term, env)
  record_peak(term, env, 0.20, progress=0.5)
  set_heading_pose(env, foot_forward=0.30)
  land(env)
  assert evaluate(term, env).item() == pytest.approx(0.0, abs=1e-6)

  env.extras["log"] = {}
  term.reset(env.reset_ids)
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0

  expected_keys = {
    "Metrics/knee_forward_peak_long_m",
    "Metrics/knee_forward_peak_phase_long",
    "Metrics/landing_reach_long_m",
    "Metrics/knee_forward_normalized_error",
  }
  log = env.extras["log"]
  assert set(log) == expected_keys
  assert log["Metrics/knee_forward_peak_long_m"].tolist() == pytest.approx((0.20,))
  assert log["Metrics/knee_forward_peak_phase_long"].tolist() == pytest.approx((0.50,))
  assert log["Metrics/landing_reach_long_m"].tolist() == pytest.approx((0.30,))
  assert log["Metrics/knee_forward_normalized_error"].tolist() == pytest.approx((0.0,))
  assert all(value.numel() > 0 and torch.isfinite(value).all() for value in log.values())

  snapshot = {key: value.clone() for key, value in log.items()}
  assert evaluate(term, env).item() == 0.0
  assert set(log) == expected_keys
  for key, value in snapshot.items():
    torch.testing.assert_close(log[key], value)

  set_heading_pose(env, knee_forward=0.0, foot_forward=0.0)
  establish_contact(term, env)
  liftoff(term, env)
  record_peak(term, env, 0.10, progress=0.6)
  set_heading_pose(env, foot_forward=-0.20)
  land(env)
  assert evaluate(term, env).item() == pytest.approx(0.25, abs=1e-6)
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0

  assert log["Metrics/knee_forward_peak_long_m"].tolist() == pytest.approx(
    (0.20, 0.10)
  )
  assert log["Metrics/knee_forward_peak_phase_long"].tolist() == pytest.approx(
    (0.50, 0.60)
  )
  assert log["Metrics/landing_reach_long_m"].tolist() == pytest.approx((0.30, -0.20))
  assert log["Metrics/knee_forward_normalized_error"].tolist() == pytest.approx(
    (0.0, 0.25)
  )
  assert all(value.numel() > 0 and torch.isfinite(value).all() for value in log.values())


@pytest.mark.parametrize(
  "existing",
  (1.0, [1.0], torch.tensor(1.0, dtype=torch.float64)),
  ids=("python-float", "python-list", "scalar-tensor"),
)
def test_pending_metric_flush_normalizes_existing_values(
  existing: float | list[float] | torch.Tensor,
) -> None:
  env, term = fixture(step_lengths=(0.20,))
  establish_contact(term, env)
  liftoff(term, env)
  record_peak(term, env, 0.05)
  land(env)
  assert evaluate(term, env).item() == pytest.approx(0.25, abs=1e-6)

  key = "Metrics/knee_forward_normalized_error"
  env.extras["log"][key] = existing
  env.scene["feet_ground_contact"].clear_events()
  assert evaluate(term, env).item() == 0.0

  samples = env.extras["log"][key]
  assert isinstance(samples, torch.Tensor)
  assert samples.ndim == 1
  assert samples.device == term._peak_knee_forward.device
  assert samples.dtype == term._peak_knee_forward.dtype
  assert torch.isfinite(samples).all()
  torch.testing.assert_close(samples, torch.tensor((1.0, 0.25)))

def test_reset_discards_only_selected_incomplete_swings() -> None:
  env, term = fixture(step_lengths=(0.20, 0.20))
  establish_contact(term, env)
  liftoff(term, env)
  record_peak(term, env, (0.05, 0.05))

  term.reset(env.reset_ids[:1])
  land(env)
  assert evaluate(term, env).tolist() == pytest.approx((0.0, 0.25), abs=1e-6)


@pytest.mark.parametrize(
  ("command_name", "bad_value", "message"),
  (
    ("step_length", torch.tensor(((0.20, 0.20),)), "shape"),
    ("step_length", torch.tensor(((float("nan"),),)), "finite"),
    ("twist", torch.tensor(((0.5, 0.0),)), "shape"),
    ("twist", torch.tensor(((float("inf"), 0.0, 0.0),)), "finite"),
  ),
)
def test_invalid_commands_fail_closed(
  command_name: str, bad_value: torch.Tensor, message: str
) -> None:
  env, term = fixture()
  env.command_manager.commands[command_name] = bad_value
  with pytest.raises(ValueError, match=message):
    evaluate(term, env)


@pytest.mark.parametrize(
  ("field", "bad_value"),
  (
    ("root_link_pos_w", torch.tensor(((float("nan"), 0.0, 0.0),))),
    ("root_link_quat_w", torch.tensor(((1.0, 0.0, 0.0, float("inf")),))),
    (
      "body_link_pos_w",
      torch.tensor((((float("nan"), 0.0, 0.5), (0.0, 0.0, 0.5)),)),
    ),
    (
      "site_pos_w",
      torch.tensor((((0.0, 0.0, 0.05), (float("inf"), 0.0, 0.05)),)),
    ),
  ),
)
def test_nonfinite_poses_fail_closed(field: str, bad_value: torch.Tensor) -> None:
  env, term = fixture()
  setattr(env.scene["robot"].data, field, bad_value)
  with pytest.raises(ValueError, match="finite"):
    evaluate(term, env)


@pytest.mark.parametrize(
  ("field", "bad_value"),
  (
    (
      "body_link_pos_w",
      torch.tensor(
        (((0.0, 0.0, 0.5), (0.0, 0.0, 0.5), (float("nan"), 0.0, 0.5)),)
      ),
    ),
    (
      "site_pos_w",
      torch.tensor(
        (((0.0, 0.0, 0.05), (0.0, 0.0, 0.05), (float("inf"), 0.0, 0.05)),)
      ),
    ),
  ),
)
def test_full_pose_tensor_is_finite_before_entity_indexing(
  field: str, bad_value: torch.Tensor
) -> None:
  env, term = fixture()
  setattr(env.scene["robot"].data, field, bad_value)
  with pytest.raises(ValueError, match="finite"):
    evaluate(term, env)

@pytest.mark.parametrize(
  ("field", "bad_value"),
  (
    ("root_link_pos_w", torch.zeros(1, 2)),
    ("root_link_quat_w", torch.zeros(1, 3)),
    ("body_link_pos_w", torch.zeros(1, 1, 3)),
    ("site_pos_w", torch.zeros(1, 1, 3)),
  ),
)
def test_wrong_pose_shapes_fail_closed(field: str, bad_value: torch.Tensor) -> None:
  env, term = fixture()
  setattr(env.scene["robot"].data, field, bad_value)
  with pytest.raises(ValueError, match="shape"):
    evaluate(term, env)


def test_wrong_contact_shape_fails_closed() -> None:
  env, term = fixture()
  env.scene["feet_ground_contact"].data.current_air_time = torch.zeros(1, 1)
  with pytest.raises(ValueError, match=r"expected.*\(1, 2\)"):
    evaluate(term, env)


def test_reversed_contact_primaries_fail_closed() -> None:
  env, _ = fixture()
  env.scene["feet_ground_contact"]._slots.reverse()
  with pytest.raises(ValueError, match="left.*right"):
    CommandedKneeForward(
      RewardTermCfg(func=CommandedKneeForward, weight=-5.0, params=reward_params()),
      env,
    )


@pytest.mark.parametrize(
  ("config_name", "bad_cfg", "message"),
  (
    (
      "knee_body_cfg",
      SceneEntityCfg(
        "robot",
        body_names=("right_knee_link", "left_knee_link"),
        body_ids=[0, 1],
        preserve_order=True,
      ),
      "left_knee_link",
    ),
    (
      "foot_site_cfg",
      SceneEntityCfg(
        "robot",
        site_names=("right_foot", "left_foot"),
        site_ids=[0, 1],
        preserve_order=True,
      ),
      "left_foot",
    ),
  ),
)
def test_reversed_entity_order_fails_closed(
  config_name: str, bad_cfg: SceneEntityCfg, message: str
) -> None:
  with pytest.raises(ValueError, match=message):
    fixture(config_overrides={config_name: bad_cfg})


def test_ambiguous_runtime_level_fails_closed() -> None:
  levels = (0.20, 0.200001, 0.40)
  env, term = fixture(
    step_lengths=(0.2000005,),
    config_overrides={"step_length_levels": levels},
  )
  with pytest.raises(ValueError, match="exactly one configured step-length level"):
    evaluate(term, env, step_length_levels=levels)


@pytest.mark.parametrize(
  "overrides",
  (
    {"step_length_levels": (0.20, 0.30)},
    {"step_length_levels": (0.20, 0.20, 0.40)},
    {"knee_forward_targets": (0.10, float("nan"), 0.20)},
    {"knee_forward_targets": (0.15, 0.10, 0.20)},
    {"nominal_swing_time_s": 0.0},
    {"tracking_window": (0.80, 0.30)},
    {"tracking_window": (-0.01, 0.80)},
  ),
)
def test_invalid_configuration_fails_closed(overrides: dict[str, object]) -> None:
  with pytest.raises(ValueError):
    fixture(config_overrides=overrides)


@pytest.mark.parametrize(
  "runtime_overrides",
  (
    {"sensor_name": "other"},
    {"step_length_command_name": "other"},
    {"step_length_levels": (0.20, 0.31, 0.40)},
    {"knee_forward_targets": (0.10, 0.16, 0.20)},
    {"command_name": "other"},
    {"command_threshold": 0.2},
    {"nominal_swing_time_s": 0.9},
    {"tracking_window": (0.31, 0.80)},
    {
      "knee_body_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_knee_link", "right_knee_link"),
        body_ids=[1, 0],
        preserve_order=True,
      )
    },
    {
      "foot_site_cfg": SceneEntityCfg(
        "robot",
        site_names=("left_foot", "right_foot"),
        site_ids=[1, 0],
        preserve_order=True,
      )
    },
  ),
)
def test_runtime_parameter_changes_fail_closed(
  runtime_overrides: dict[str, object],
) -> None:
  env, term = fixture()
  with pytest.raises((KeyError, ValueError)):
    evaluate(term, env, **runtime_overrides)
