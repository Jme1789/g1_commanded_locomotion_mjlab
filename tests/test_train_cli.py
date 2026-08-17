from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import warp as wp
import yaml

from scripts.train import (
  TrainConfig,
  TuningError,
  configure_warp_cuda_compilation,
  prepare_train_config,
  render_effective_config,
  write_training_params,
)


def register_tasks() -> None:
  importlib.import_module("mjlab.tasks")
  importlib.import_module("src.tasks")


def test_training_runtime_forces_warp_cubin_output(monkeypatch) -> None:
  monkeypatch.setattr(wp.config, "cuda_output", None)

  configure_warp_cuda_compilation()

  assert wp.config.cuda_output == "cubin"


def test_cli_value_wins_over_central_tuning(monkeypatch) -> None:
  register_tasks()
  monkeypatch.setattr(
    "scripts.train.active_rl_tuning_overrides",
    lambda: {"agent.algorithm.learning_rate": 5.0e-4},
  )
  prepared = prepare_train_config(
    "Unitree-G1-Stairs",
    ["--agent.algorithm.learning-rate", "0.00025"],
  )
  assert prepared.cfg.agent.algorithm.learning_rate == 2.5e-4
  assert (
    prepared.tuning_diff["agent.algorithm.learning_rate"]["tuning"] == 5.0e-4
  )
  assert prepared.cli_diff["agent.algorithm.learning_rate"]["cli"] == 2.5e-4


def test_rendered_config_contains_default_tuning_cli_and_final(monkeypatch) -> None:
  register_tasks()
  monkeypatch.setattr(
    "scripts.train.active_rl_tuning_overrides",
    lambda: {"agent.algorithm.learning_rate": 5.0e-4},
  )
  prepared = prepare_train_config("Unitree-G1-Stairs", [])
  rendered = render_effective_config(prepared)
  assert rendered["schema_version"] == 2
  assert rendered["profile_path"] is None
  assert rendered["profile_revision"] is None
  item = rendered["parameters"]["agent.algorithm.learning_rate"]
  assert item["default"] == 1.0e-3
  assert item["tuning"] == 5.0e-4
  assert item["profile"] == "KEEP_TASK_DEFAULT"
  assert item["cli"] == "KEEP_TASK_DEFAULT"
  assert item["final"] == 5.0e-4
  assert item["definition"]


def test_dump_effective_config_is_safe_yaml(
  tmp_path: Path, monkeypatch
) -> None:
  register_tasks()
  output = tmp_path / "effective.yaml"
  prepared = prepare_train_config(
    "Unitree-G1-Stairs",
    ["--dump-effective-config", str(output)],
  )
  prepared.dump_if_requested()
  loaded = yaml.safe_load(output.read_text())
  assert loaded["task_id"] == "Unitree-G1-Stairs"
  assert loaded["baseline_commit"].startswith("3ac7da5")
  assert "env.rewards.track_linear_velocity.weight" in loaded["parameters"]


def test_print_effective_config_accepts_bare_flag() -> None:
  register_tasks()
  prepared = prepare_train_config("Unitree-G1-Stairs", ["--print-effective-config"])
  assert prepared.cfg.print_effective_config is True


def test_invalid_cli_value_uses_the_same_validator() -> None:
  register_tasks()
  with pytest.raises(TuningError, match="learning_rate"):
    prepare_train_config(
      "Unitree-G1-Stairs",
      ["--agent.algorithm.learning-rate", "-0.001"],
    )


def test_non_target_task_keeps_default_preparation_compatible() -> None:
  register_tasks()
  prepared = prepare_train_config("Mjlab-Lift-Cube-Yam", [])
  assert prepared.cfg.agent.algorithm.learning_rate == 1.0e-3
  assert prepared.tuning_diff == {}
  assert prepared.cli_diff == {}


def test_training_cli_has_no_reverse_dependency_on_console_profiles() -> None:
  register_tasks()
  with pytest.raises(SystemExit):
    prepare_train_config(
      "Unitree-G1-Stairs",
      ["--config", "personal-console-profile.yaml"],
    )


def test_training_params_include_tuning_manifest(tmp_path: Path) -> None:
  register_tasks()
  cfg = TrainConfig.from_task("Unitree-G1-Stairs")
  manifest = {"schema_version": 1, "task_id": "Unitree-G1-Stairs"}
  write_training_params(tmp_path, cfg, manifest)
  assert (tmp_path / "params" / "env.yaml").exists()
  assert (tmp_path / "params" / "agent.yaml").exists()
  loaded = yaml.safe_load(
    (tmp_path / "params" / "tuning.yaml").read_text()
  )
  assert loaded == manifest
