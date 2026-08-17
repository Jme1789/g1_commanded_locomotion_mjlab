#pragma once

#include <filesystem>
#include <vector>

#include "gamepad/gamepad_profile.h"

namespace gamepad {

struct DiscoveryRoots {
  std::filesystem::path dev_input;
  std::filesystem::path sys_class_input;
  std::filesystem::path by_id;

  static DiscoveryRoots System();
};

std::vector<DeviceDescriptor> EnumerateLinuxJoysticks(
    const DiscoveryRoots& roots = DiscoveryRoots::System());
DeviceDescriptor SelectUniqueDevice(
    const DeviceIdentity& identity,
    const std::vector<DeviceDescriptor>& descriptors);

}  // namespace gamepad
