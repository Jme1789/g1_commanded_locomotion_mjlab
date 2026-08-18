# Configuration Guide

The validated release task is `Unitree-G1-H20-BalanceCurriculum`. The
repository follows the MJLab convention of separating task registration,
environment configuration, RL configuration, reusable MDP terms, and runtime
deployment configuration.

## Where to make a change

| Goal | File | Entry point |
| --- | --- | --- |
| Register or select a task | `src/tasks/velocity/config/g1/__init__.py` | `register_mjlab_task(...)` |
| Change the final terrain, commands, curriculum, or reward wiring | `src/tasks/velocity/config/g1/env_cfgs.py` | `unitree_g1_h20_balance_curriculum_env_cfg()` |
| Change PPO, actor/critic, learning rate, iterations, or run name | `src/tasks/velocity/config/g1/rl_cfg.py` | `unitree_g1_h20_balance_curriculum_ppo_runner_cfg()` |
| Change shared velocity-task observations or rewards | `src/tasks/velocity/velocity_env_cfg.py` | `make_velocity_env_cfg()` |
| Change swing-height command behavior | `src/tasks/velocity/mdp/swing_height_command.py` | `DiscreteSwingHeightCommandCfg` |
| Change step-length command behavior | `src/tasks/velocity/mdp/step_length_command.py` | `DiscreteStepLengthCommandCfg` |
| Change knee-lift reward math | `src/tasks/velocity/mdp/knee_lift_reward.py` | `CommandedKneeLift` |
| Change knee-forward reward math | `src/tasks/velocity/mdp/knee_forward_reward.py` | `CommandedKneeForward` |
| Change deterministic Play defaults | `scripts/play_forward.py` | `PlayForwardConfig` |
| Change simulator or local gamepad settings | `simulate/config.yaml` and `simulate/config/gamepads/` | YAML configuration |
| Change G1 FSM or policy deployment settings | `deploy/robots/g1/config/config.yaml` | `FSM` and policy sections |
| Change exported policy observation/action contract | `deploy/robots/g1/config/policy/velocity/v1/params/deploy.yaml` | observation and action terms |

## Final task composition

The final environment is intentionally layered so each training stage changes
one behavior:

```text
three-height + knee lift + step length
  -> fixed high lift
  -> longer forward step
  -> standing replay
  -> symmetric yaw replay
  -> H20 balance curriculum
```

The final function sets 2,048 environments and the stair/rough-terrain
curriculum. Its matching RL function sets 5,001 iterations, TensorBoard
logging, and the `h20-balance-curriculum-r0` run name.

Inspect the effective configuration without starting training:

```bash
python scripts/train.py \
  Unitree-G1-H20-BalanceCurriculum \
  --print-effective-config
```

Command-line overrides take precedence over registered task defaults. Changes
to observation dimensions, action dimensions, term order, or network topology
also require retraining, ONNX export, and matching deployment configuration.
