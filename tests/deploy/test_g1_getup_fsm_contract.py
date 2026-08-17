from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
G1_CONFIG = ROOT / "deploy/robots/g1/config/config.yaml"


def test_g1_getup_fsm_trigger_exit_and_failure_contract() -> None:
  fsm = yaml.safe_load(G1_CONFIG.read_text())["FSM"]

  assert fsm["_"]["GetUp"] == {"id": 4}
  assert fsm["_"]["Fallen"] == {"id": 6}
  assert fsm["Passive"]["transitions"]["GetUp"] == "A"
  assert fsm["FixStand"]["transitions"]["GetUp"] == "A"
  assert "GetUp" not in fsm["Velocity"]["transitions"]
  assert "GetUp" not in fsm["Mimic_Dance1_subject2"]["transitions"]

  fallen = fsm["Fallen"]
  assert fallen == {
    "transitions": {
      "Passive": "LT + B.on_pressed",
      "GetUp": "A",
    },
    "detection": {
      "fallen_tilt_min_rad": 1.0,
      "confirm_duration_s": 0.2,
      "max_update_gap_s": 0.1,
    },
  }

  getup = fsm["GetUp"]
  assert getup["transitions"] == {"Passive": "LT + B.on_pressed"}
  assert getup["policy_dir"] == "config/policy/getup/amp_reference"
  assert getup["success_state"] == "Velocity"
  assert getup["failure_state"] == "Fallen"
  assert getup["trigger"] == {
    "hold_s": 1.0,
    "fallen_tilt_min_rad": 1.0,
    "max_update_gap_s": 0.1,
  }
  assert getup["recovery"] == {
    "upright_tilt_max_rad": 0.35,
    "angular_speed_max_rad_s": 0.5,
    "joint_speed_max_rad_s": 1.0,
    "stable_duration_s": 1.0,
    "timeout_s": 8.0,
    "max_consecutive_inference_failures": 3,
    "max_sample_gap_s": 0.1,
  }
