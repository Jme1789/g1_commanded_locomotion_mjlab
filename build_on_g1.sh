#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
controller_dir="${root_dir}/deploy/robots/g1"
build_dir="${controller_dir}/build"
onnx_lib_dir="${root_dir}/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "错误：此部署包必须在 aarch64 G1 上构建。" >&2
  exit 1
fi

for command_name in cmake c++ sha256sum ldd; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "错误：缺少构建命令 ${command_name}。" >&2
    exit 1
  fi
done

cd "${root_dir}"
sha256sum --check artifacts/g1-commanded-locomotion-v1/MANIFEST.sha256

cmake \
  -S "${controller_dir}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target g1_ctrl -j"$(nproc)"

controller="${build_dir}/g1_ctrl"
if [[ ! -x "${controller}" ]]; then
  echo "错误：构建结束后没有生成 g1_ctrl。" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${onnx_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if ldd "${controller}" | grep -q "not found"; then
  ldd "${controller}" >&2
  echo "错误：g1_ctrl 仍有未满足的动态库依赖。" >&2
  exit 1
fi

echo "G1 ARM64 控制器构建完成：${controller}"
