from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
G1_CONFIG = ROOT / "deploy/robots/g1/config/config.yaml"
VELOCITY_CONFIG = (
  ROOT / "deploy/robots/g1/config/policy/velocity/v1/params/deploy.yaml"
)


def test_start_enters_fixstand_and_lt_boosts_shoulder_yaw() -> None:
  g1 = yaml.safe_load(G1_CONFIG.read_text())
  velocity = yaml.safe_load(VELOCITY_CONFIG.read_text())

  assert g1["FSM"]["Passive"]["transitions"] == {
    "FixStand": "start.on_pressed",
    "GetUp": "A",
  }

  yaw_range = velocity["commands"]["base_velocity"]["ranges"]["ang_vel_z"]
  mapping = velocity["external_control"]["motion_mapping"]
  assert yaw_range == [-1.5, 1.5]
  assert mapping["shoulder_yaw_speed"] == 1.0
  assert mapping["yaw_boost_multiplier"] == 1.5
  assert mapping["shoulder_yaw_speed"] * mapping["yaw_boost_multiplier"] == 1.5
