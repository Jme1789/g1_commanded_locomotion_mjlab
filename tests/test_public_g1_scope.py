from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_public_training_repository_excludes_local_tuning_console() -> None:
  excluded = (
    ROOT / "src" / "training_console",
    ROOT / "tests" / "training_console",
    ROOT / "configs" / "training_console",
    ROOT / "scripts" / "tuning_console.py",
    ROOT / "scripts" / "tuning_console_worker.py",
  )

  assert all(not path.exists() for path in excluded)
  assert "src.training_console" not in (ROOT / "scripts" / "train.py").read_text()
  assert "src.training_console" not in (ROOT / "scripts" / "play.py").read_text()


def test_public_task_catalog_does_not_advertise_non_g1_unitree_tasks() -> None:
  """The G1 release must not register robot packages it no longer ships."""
  result = subprocess.run(
    [sys.executable, "-m", "scripts.list_envs"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
  )
  shipped_tasks = {
    field.strip()
    for line in result.stdout.splitlines()
    if len(fields := line.split("|")) >= 3
    if (field := fields[2]).strip().startswith("Unitree-")
  }

  assert shipped_tasks
  assert all(task.startswith("Unitree-G1") for task in shipped_tasks)
