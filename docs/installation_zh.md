# 安装说明

## 支持的运行路径

| 功能 | 运行环境 |
| --- | --- |
| 训练与 Play | Ubuntu、Python 3.11，建议 NVIDIA GPU |
| 手柄映射台 | 能提供 `/dev/input/js*` 的 Linux |
| MuJoCo Sim2Sim | x86_64 Linux |
| 实机控制器 | aarch64 Unitree G1 |

Python 代码可以在 CPU 环境检查，但 MuJoCo-Warp 训练和 Play 需要兼容的
NVIDIA 驱动、CUDA 与 PyTorch。

## Python 环境

建议使用独立环境：

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

`setup.py` 固定了 MJLab、MuJoCo-Warp、FastAPI、Uvicorn、HTTPX 和 PyYAML
等核心依赖。若环境未预装 PyTorch，请先按本机 CUDA 版本安装正确的 GPU 版本。

检查任务注册：

~~~bash
python -m scripts.list_envs
~~~

本项目自行注册的 `Unitree-*` 条目应全部是 G1；已安装的 MJLab 依赖仍可能同时列出自带的 `Mjlab-*` 示例，其中可能包含其他机器人名称。

## 本地 C++ 依赖

Ubuntu 常用依赖：

~~~bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  libboost-all-dev libeigen3-dev libyaml-cpp-dev \
  libspdlog-dev libfmt-dev libglfw3-dev zlib1g-dev
~~~

Unitree SDK2 与 CycloneDDS 需要另行安装。仿真 CMake 默认从
`/opt/unitree_robotics/lib/cmake` 查找 Unitree SDK；控制器还会按
`deploy/robots/g1/CMakeLists.txt` 中的系统路径查找 CycloneDDS。

## 构建仿真和控制器

~~~bash
cmake -S simulate -B simulate/build
cmake --build simulate/build --target unitree_mujoco jstest -j2

cmake -S deploy/robots/g1 -B deploy/robots/g1/build
cmake --build deploy/robots/g1/build --target g1_ctrl -j2
~~~

仓库保留了 x86_64 与 aarch64 对应的 ONNX Runtime 文件，CMake 根据
`CMAKE_SYSTEM_PROCESSOR` 自动选择。

## 校验最终模型

~~~bash
sha256sum --check artifacts/g1-commanded-locomotion-v1/MANIFEST.sha256
~~~

该清单覆盖最终 PT、训练配置快照、精简训练指标、趋势图和对应 velocity ONNX。

## 本地网页工具

训练调参台：

~~~bash
python -m scripts.tuning_console
~~~

手柄映射台：

~~~bash
python -m scripts.gamepad_calibrator
~~~

两者默认只监听本地回环地址，不应暴露到不可信网络。

## 常见问题

- **创建仿真图时显存不足：** 降低 `--env.scene.num-envs`；11 GiB 显存
  通常无法承受 4096 个当前地形环境。
- **找不到手柄：** 检查 `/dev/input/js0`、input 组权限和手柄模式，再重新
  生成 `active.yaml`。
- **loopback 不支持 multicast：** 本地 Sim2Sim 中通常是正常提示，只要两个
  进程能完成 DDS 连接。
- **找不到 ONNX Runtime：** 检查机器架构和 `LD_LIBRARY_PATH`。
- **G1 上出现 clock skew：** 先同步电脑和机器人时间，再重新构建。
