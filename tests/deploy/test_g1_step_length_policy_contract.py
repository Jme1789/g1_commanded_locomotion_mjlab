from __future__ import annotations

import hashlib
from pathlib import Path

import onnx
import yaml

ROOT = Path(__file__).resolve().parents[2]
G1_CONFIG = ROOT / "deploy/robots/g1/config/config.yaml"
POLICY_ROOT = ROOT / "deploy/robots/g1/config/policy/velocity"
V1_YAML = POLICY_ROOT / "v1/params/deploy.yaml"
V1_ONNX = POLICY_ROOT / "v1/exported/policy.onnx"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(model: onnx.ModelProto) -> dict[str, str]:
  return {entry.key: entry.value for entry in model.metadata_props}


def _csv_floats(value: str) -> list[float]:
  return [float(item) for item in value.split(",")]


def _shape(value_info: onnx.ValueInfoProto) -> list[int]:
  return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


def test_g1_velocity_explicitly_selects_v1() -> None:
  config = yaml.safe_load(G1_CONFIG.read_text())
  assert config["FSM"]["Velocity"]["policy_dir"] == "config/policy/velocity/v1"


def test_v1_yaml_matches_the_exported_policy_contract() -> None:
  assert _sha256(V1_ONNX) == (
    "e3d16f4dbc67fc9e78d5eaceaa7370ef5df12e4114bddacf518bc9cdeb0ddb89"
  )
  model = onnx.load(V1_ONNX, load_external_data=False)
  metadata = _metadata(model)
  assert [(value.name, _shape(value)) for value in model.graph.input] == [
    ("obs", [1, 100])
  ]
  assert [(value.name, _shape(value)) for value in model.graph.output] == [
    ("actions", [1, 29])
  ]
  assert metadata["command_names"].split(",") == [
    "twist",
    "swing_height",
    "step_length",
  ]
  assert metadata["observation_names"].split(",") == [
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
    "swing_height_command",
    "step_length_command",
  ]

  config = yaml.safe_load(V1_YAML.read_text())
  assert config["stiffness"] == _csv_floats(metadata["joint_stiffness"])
  assert config["damping"] == _csv_floats(metadata["joint_damping"])
  assert config["default_joint_pos"] == _csv_floats(
    metadata["default_joint_pos"]
  )
  action = config["actions"]["JointPositionAction"]
  assert action["scale"] == _csv_floats(metadata["action_scale"])
  assert action["offset"] == _csv_floats(metadata["default_joint_pos"])

  assert list(config["observations"]) == [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "gait_phase",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "swing_height_command",
    "step_length_command",
  ]
  assert config["observations"]["swing_height_command"] == {
    "params": {},
    "clip": None,
    "scale": [1.0],
    "history_length": 1,
  }
  assert config["observations"]["step_length_command"] == {
    "params": {},
    "clip": None,
    "scale": [1.0],
    "history_length": 1,
  }

  external = config["external_control"]
  assert external["swing_height"]["joint_overlay_enabled"] is False
  assert external["swing_height"]["height_command_m"] == {
    "low": 0.05,
    "medium": 0.10,
    "high": 0.20,
  }
  assert external["motion_mapping"]["step_length_m"] == {
    "short": 0.20,
    "medium": 0.30,
    "long": 0.40,
  }
