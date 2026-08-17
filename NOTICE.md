# Notices and attribution

## Upstream project

This repository is derived from
[Unitree Robotics' unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab).
Upstream source and this project's source modifications are distributed under
the Apache License 2.0 in `LICENCE`, except where a bundled component states a
different license.

The project also builds on MJLab, MuJoCo/MuJoCo-Warp, RSL-RL, PyTorch, Warp,
Unitree SDK2, CycloneDDS, FastAPI, Uvicorn, and other dependencies installed by
the user. Their names and licenses remain the property of their respective
authors.

## Bundled third-party components

- ONNX Runtime redistributable files are stored under
  `deploy/thirdparty/onnxruntime-*` with Microsoft's license and notices.
- cnpy source is stored under `deploy/thirdparty/cnpy` with its own license.
- MuJoCo headers, libraries, and simulator support files are stored under
  `simulate/mujoco` and remain subject to the MuJoCo license shipped there.
- Unitree G1 model assets and example G1 mimic data originate from the upstream
  Unitree repository and remain subject to the upstream terms.

Do not remove bundled license or notice files when redistributing binaries.

## Trained artifacts

`artifacts/g1-commanded-locomotion-v1/model_37300.pt` and the matching velocity
ONNX were produced for this project from the included G1 training configuration.
They are published for research and reproducibility under the repository
license, subject to the licenses of upstream software and robot assets used to
produce them.

## GetUp reference

The Fallen damping FSM and GetUp integration code are included, but the
externally sourced GetUp ONNX weight is not. The implementation was evaluated
against the public
[ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) reference. At the
time this release was prepared, that repository did not expose an explicit
model redistribution license. Users must independently obtain a compatible
model and verify their right to use it.
