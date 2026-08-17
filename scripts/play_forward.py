"""Play a trained policy with a FIXED forward velocity command.

Unlike scripts/play.py (which uses random heading + speed sampling, so the robot
wanders and rarely walks off a stair edge), this script overrides the twist
command to a constant forward velocity so you can clearly see the robot climb
and descend stairs.

Usage:
  python -m scripts.play_forward Unitree-G1-H20-BalanceCurriculum

Append CLI options only when intentionally overriding the release defaults.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ForwardPlayConfig:
  checkpoint_file: str = "artifacts/g1-commanded-locomotion-v1/model_37300.pt"
  num_envs: int = 1
  device: str | None = None
  vx: float = 0.5
  """Constant forward (x) velocity command, m/s. Use negative to walk backward."""
  vy: float = 0.0
  wz: float = 0.0
  """Constant yaw-rate command, rad/s (0 = walk straight)."""


def run(task_id: str, cfg: ForwardPlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  # The stairs config tunes nconmax/njmax low for memory efficiency at 2048
  # training envs. With only a few play envs on complex stair terrain the
  # per-env contact peak is higher, so raise the caps to avoid overflow.
  env_cfg.sim.nconmax = 512
  if env_cfg.sim.njmax is not None:
    env_cfg.sim.njmax = 2000

  # Force a constant forward command: no heading control, no standing envs,
  # zero-width ranges so every resample yields the same velocity.
  tw = env_cfg.commands["twist"]
  assert isinstance(tw, UniformVelocityCommandCfg)
  tw.heading_command = False
  tw.ranges.heading = None  # required when heading_command is False
  tw.rel_standing_envs = 0.0
  tw.ranges.lin_vel_x = (cfg.vx, cfg.vx)
  tw.ranges.lin_vel_y = (cfg.vy, cfg.vy)
  tw.ranges.ang_vel_z = (cfg.wz, cfg.wz)

  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  print(f"[INFO] Loading checkpoint: {resume_path}")
  print(f"[INFO] Fixed command  vx={cfg.vx}  vy={cfg.vy}  wz={cfg.wz}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  from mjlab.viewer import NativeMujocoViewer

  NativeMujocoViewer(env, policy).run()


def main():
  import mjlab.tasks

  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    ForwardPlayConfig,
    args=remaining,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run(chosen_task, args)


if __name__ == "__main__":
  main()
