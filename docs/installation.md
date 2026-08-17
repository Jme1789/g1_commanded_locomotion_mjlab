# Installation

## Supported release paths

| Path | Host |
| --- | --- |
| training and Play | Ubuntu, Python 3.11, NVIDIA GPU recommended |
| gamepad console | Linux with `/dev/input/js*` |
| MuJoCo sim-to-sim | x86_64 Linux |
| real controller | aarch64 Unitree G1 |

The Python package can be inspected on CPU, but MuJoCo-Warp training and Play
are expected to use a compatible NVIDIA driver/CUDA stack.

## Python environment

Using a dedicated environment is recommended:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

The package pins the core MJLab, MuJoCo-Warp, FastAPI, Uvicorn, HTTPX, and
PyYAML dependencies in `setup.py`. Install the correct CUDA-enabled PyTorch
build for the host before training if it is not already supplied by the
environment.

Verify task discovery:

~~~bash
python -m scripts.list_envs
~~~

Every project-defined `Unitree-*` entry should be G1. The installed MJLab dependency may also list its built-in `Mjlab-*` examples, including other robot names.

## Native dependencies

A typical Ubuntu host needs:

~~~bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  libboost-all-dev libeigen3-dev libyaml-cpp-dev \
  libspdlog-dev libfmt-dev libglfw3-dev zlib1g-dev
~~~

Unitree SDK2 and CycloneDDS must be installed separately. The simulator CMake
configuration expects Unitree's package configuration under
`/opt/unitree_robotics/lib/cmake`. The G1 controller also expects CycloneDDS
headers/libraries in the system paths shown in
`deploy/robots/g1/CMakeLists.txt`.

## Build the native targets

~~~bash
cmake -S simulate -B simulate/build
cmake --build simulate/build --target unitree_mujoco jstest -j2

cmake -S deploy/robots/g1 -B deploy/robots/g1/build
cmake --build deploy/robots/g1/build --target g1_ctrl -j2
~~~

The repository carries platform-specific ONNX Runtime files for x86_64 and
aarch64. CMake selects the directory from `CMAKE_SYSTEM_PROCESSOR`.

## Verify the release artifact

~~~bash
sha256sum --check artifacts/g1-commanded-locomotion-v1/MANIFEST.sha256
~~~

This verifies the selected PT checkpoint, its configuration snapshots, compact
training evidence, and matching velocity ONNX policy.

## Local web tools

Training console:

~~~bash
python -m scripts.tuning_console
~~~

Gamepad mapping console:

~~~bash
python -m scripts.gamepad_calibrator
~~~

Both bind to the local loopback interface by default. Do not expose them to an
untrusted network.

## Common setup failures

- **CUDA out of memory during graph creation:** reduce
  `--env.scene.num-envs`; 4096 environments can exceed an 11 GiB GPU.
- **No gamepad candidate:** check `/dev/input/js0`, permissions, and controller
  mode, then regenerate `active.yaml`.
- **DDS loopback multicast warning:** disabling multicast on `lo` is expected
  for local sim-to-sim as long as both processes connect.
- **ONNX Runtime not found:** inspect the selected architecture and
  `LD_LIBRARY_PATH`.
- **Clock skew on G1:** synchronize clocks before rebuilding.
