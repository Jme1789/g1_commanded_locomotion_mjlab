#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
controller="${root_dir}/deploy/robots/g1/build/g1_ctrl"
onnx_lib_dir="${root_dir}/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib"
network_interface="${1:-}"
gamepad_device="${2:-/dev/input/js0}"

if [[ -z "${network_interface}" ]]; then
  echo "用法：$0 <G1网络接口> [手柄设备，默认/dev/input/js0]" >&2
  echo "示例：$0 eth0 /dev/input/js0" >&2
  exit 2
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "错误：只能在 aarch64 G1 上启动此控制器。" >&2
  exit 1
fi

if [[ ! -x "${controller}" ]]; then
  echo "错误：尚未构建控制器，请先运行 ./build_on_g1.sh。" >&2
  exit 1
fi

if [[ ! -d "/sys/class/net/${network_interface}" ]]; then
  echo "错误：网络接口 ${network_interface} 不存在。" >&2
  exit 1
fi

if [[ ! -e "${gamepad_device}" ]]; then
  echo "错误：未找到第三方手柄设备 ${gamepad_device}。" >&2
  exit 1
fi

if [[ ! -r "${gamepad_device}" ]]; then
  echo "错误：当前用户无权读取 ${gamepad_device}；请加入 input 组或使用具备权限的账户。" >&2
  exit 1
fi

for required_file in \
  "${root_dir}/deploy/robots/g1/config/config.yaml" \
  "${root_dir}/deploy/robots/g1/config/policy/velocity/v1/exported/policy.onnx" \
  "${root_dir}/deploy/robots/g1/config/policy/mimic/dance1_subject2/exported/policy.onnx"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "错误：缺少运行文件 ${required_file}。" >&2
    exit 1
  fi
done

getup_policy="${root_dir}/deploy/robots/g1/config/policy/getup/amp_reference/exported/policy.onnx"
if [[ ! -f "${getup_policy}" ]]; then
  echo "警告：GetUp policy 未安装；Fallen 阻尼保护可用，但长按 A 起身会失败并保持 Fallen。" >&2
fi

export LD_LIBRARY_PATH="${onnx_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${controller}" \
  --network="${network_interface}" \
  --custom-joystick="${gamepad_device}" \
  --log
