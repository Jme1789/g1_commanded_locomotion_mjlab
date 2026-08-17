#pragma once

#include <filesystem>
#include <memory>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

#include "gamepad/device_discovery.h"
#include "gamepad/gamepad_profile.h"
#include "gamepad/logical_mapper.h"
#include "joystick/joystick.h"

namespace gamepad {

struct EventBatch {
  RawSnapshot snapshot;
  bool connected{true};
};

class JoystickEventSource {
 public:
  virtual ~JoystickEventSource() = default;
  virtual EventBatch Drain() = 0;
};

class LinuxJoystickEventSource final : public JoystickEventSource {
 public:
  explicit LinuxJoystickEventSource(const DeviceDescriptor& descriptor);
  LinuxJoystickEventSource(const DeviceDescriptor& descriptor,
                           std::unique_ptr<Joystick> joystick);
  EventBatch Drain() override;

 private:
  DeviceDescriptor descriptor_;
  std::unique_ptr<Joystick> joystick_;
  RawSnapshot snapshot_;
};

class ConfiguredJoystick final
    : public unitree::common::UnitreeJoystick {
 public:
  static std::shared_ptr<ConfiguredJoystick> Create(
      const std::filesystem::path& active_path);
  ConfiguredJoystick(
      GamepadProfile profile,
      std::unique_ptr<JoystickEventSource> source);
  void update() override;

 private:
  void Apply(const LogicalSnapshot& logical);

  GamepadProfile profile_;
  std::unique_ptr<JoystickEventSource> source_;
  float lt_threshold_{0.5F};
  float rt_threshold_{0.5F};
  bool disconnected_{false};
};

}  // namespace gamepad
