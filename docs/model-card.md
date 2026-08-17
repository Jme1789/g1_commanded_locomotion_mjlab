# Model card: G1 commanded locomotion v1

## Summary

The selected model is a PPO actor-critic checkpoint for the Unitree G1 29-DoF
robot. It accepts velocity, swing-height, and step-length commands and produces
29 joint-position targets. The release keeps the training checkpoint and the
matching deployment ONNX model.

| Item | Value |
| --- | --- |
| task | `Unitree-G1-H20-BalanceCurriculum` |
| algorithm | PPO through RSL-RL/MJLab |
| actor observation | 100 |
| critic observation | 302 |
| action | 29 |
| selected iteration | 37300 |
| source segment | 32300 through 37300 |
| training environments | 2048 |
| seed | 42 |

## Intended use

The model is intended for:

- G1 commanded-locomotion research in MuJoCo-Warp;
- deterministic visual comparison of lift and step-length command levels;
- sim-to-sim validation through the bundled G1 controller;
- carefully supervised G1 experiments after dependency, network, and safety
  checks.

It is not intended as a certified safety controller, a general obstacle
traversal model, or a drop-in policy for a different robot or observation
layout.

## Inputs and outputs

The actor input contains deployable proprioception, gait phase, previous action,
a 3-D twist command, one swing-height scalar, and one step-length scalar. The
critic additionally receives privileged training state. Exact term ordering and
normalization are preserved in:

- `artifacts/g1-commanded-locomotion-v1/env.yaml`;
- `deploy/robots/g1/config/policy/velocity/v1/params/deploy.yaml`.

The output is a 29-element normalized joint-position action transformed by the
deployment action scale and offset.

## Training evidence

The retained run segment contains 5,001 logged iterations. Tail-100 summaries
from that segment include:

| Metric | Tail-100 mean |
| --- | ---: |
| mean episode reward | 12.1843 |
| mean episode length | 974.765 |
| terrain curriculum level | 1.7868 |
| high swing peak | 0.1522 m |
| high knee-lift peak | 0.0551 m |
| long knee-forward peak | 0.1478 m |
| long landing reach | 0.1116 m |

The compact machine-readable record is
`artifacts/g1-commanded-locomotion-v1/training_summary.json`. The raw
TensorBoard event and intermediate checkpoints are intentionally excluded.

Low-command swing and knee metrics in the retained resumed segment contain no
finite samples. This is reported rather than silently imputed. Medium/high lift
and long-step trends remain available, but these statistics do not by
themselves prove reliable stair traversal.

## Selected artifacts

| Artifact | Purpose |
| --- | --- |
| `model_37300.pt` | training resume, audit, and Play |
| `agent.yaml` | final PPO configuration snapshot |
| `env.yaml` | final environment/observation/reward snapshot |
| `training_summary.json` | compact scalar history summary |
| velocity `policy.onnx` | matching deployment actor |
| `MANIFEST.sha256` | integrity verification |

Run `sha256sum --check artifacts/g1-commanded-locomotion-v1/MANIFEST.sha256`
from the repository root before deployment.

## Limitations

- The model is trained in simulation and remains subject to sim-to-real gap.
- Command separation is visible but not perfectly linear across all gaits.
- The selected policy does not guarantee traversal of arbitrary stair height,
  edge geometry, friction, payload, or disturbances.
- A confirmed fall enters a separate damping FSM; recovery behavior is not part
  of this velocity policy.
- The optional GetUp policy is not redistributed in this release.
- Real-robot operation requires a compatible 29-DoF G1, matching gains/order,
  correct DDS interface, and supervised safety setup.

## Reproducibility

Use the exact task and configuration snapshots above. Hardware, NVIDIA driver,
CUDA, Warp/MuJoCo-Warp version, and GPU memory can change throughput or numerical
details. The checkpoint and ONNX hashes are the release identity; do not assume
that a file with the same name has the same weights.
