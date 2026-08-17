from pathlib import Path

import tyro

from scripts.play_forward import ForwardPlayConfig


def test_play_forward_uses_the_selected_release_artifact_by_default() -> None:
  cfg = tyro.cli(ForwardPlayConfig, args=[])

  assert cfg.checkpoint_file == (
    "artifacts/g1-commanded-locomotion-v1/model_37300.pt"
  )
  assert Path(cfg.checkpoint_file).is_file()
  assert cfg.num_envs == 1
  assert cfg.device is None
  assert cfg.vx == 0.5
  assert cfg.vy == 0.0
  assert cfg.wz == 0.0
