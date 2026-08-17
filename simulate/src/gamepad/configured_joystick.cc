#include "gamepad/configured_joystick.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

namespace gamepad {
namespace {

std::string ReadError(const DeviceDescriptor& descriptor, int error) {
  return descriptor.path.string() + ": joystick read failed: " +
         std::strerror(error);
}

float TriggerThreshold(const TriggerBinding& binding) {
  if (const auto* axis = std::get_if<AxisTriggerBinding>(&binding)) {
    return axis->threshold;
  }
  if (const auto* button = std::get_if<ButtonTriggerBinding>(&binding)) {
    return button->threshold;
  }
  return 0.5F;
}

class ForceImmediateAxisUpdates {
 public:
  explicit ForceImmediateAxisUpdates(
      std::array<unitree::common::Axis*, 6> axes)
      : axes_(axes) {
    for (std::size_t index = 0; index < axes_.size(); ++index) {
      smoothing_[index] = axes_[index]->smooth;
      axes_[index]->smooth = 1.0F;
    }
  }

  ~ForceImmediateAxisUpdates() {
    for (std::size_t index = 0; index < axes_.size(); ++index) {
      axes_[index]->smooth = smoothing_[index];
    }
  }

  ForceImmediateAxisUpdates(const ForceImmediateAxisUpdates&) = delete;
  ForceImmediateAxisUpdates& operator=(
      const ForceImmediateAxisUpdates&) = delete;

 private:
  std::array<unitree::common::Axis*, 6> axes_;
  std::array<float, 6> smoothing_{};
};

}  // namespace

LinuxJoystickEventSource::LinuxJoystickEventSource(
    const DeviceDescriptor& descriptor)
    : LinuxJoystickEventSource(
          descriptor, std::make_unique<Joystick>(descriptor.path.string())) {}

LinuxJoystickEventSource::LinuxJoystickEventSource(
    const DeviceDescriptor& descriptor, std::unique_ptr<Joystick> joystick)
    : descriptor_(descriptor),
      joystick_(std::move(joystick)),
      snapshot_{std::vector<std::int16_t>(descriptor.axis_count, 0),
                std::vector<std::uint8_t>(descriptor.button_count, 0)} {
  if (!joystick_ || !joystick_->isFound()) {
    const int error = joystick_ ? joystick_->lastError() : ENODEV;
    throw GamepadConfigError(ReadError(descriptor_, error));
  }
}

EventBatch LinuxJoystickEventSource::Drain() {
  while (true) {
    JoystickEvent event{};
    switch (joystick_->readEvent(&event)) {
      case Joystick::ReadStatus::Event:
        if (event.isButton()) {
          if (event.number >= snapshot_.buttons.size()) {
            throw GamepadConfigError(
                "event.button: index " + std::to_string(event.number) +
                " exceeds detected button count " +
                std::to_string(snapshot_.buttons.size()));
          }
          snapshot_.buttons[event.number] =
              event.value == 0 ? std::uint8_t{0} : std::uint8_t{1};
        } else if (event.isAxis()) {
          if (event.number >= snapshot_.axes.size()) {
            throw GamepadConfigError(
                "event.axis: index " + std::to_string(event.number) +
                " exceeds detected axis count " +
                std::to_string(snapshot_.axes.size()));
          }
          snapshot_.axes[event.number] = event.value;
        }
        break;
      case Joystick::ReadStatus::WouldBlock:
        return EventBatch{snapshot_, true};
      case Joystick::ReadStatus::Disconnected:
        return EventBatch{snapshot_, false};
      case Joystick::ReadStatus::Error:
        throw GamepadConfigError(
            ReadError(descriptor_, joystick_->lastError()));
    }
  }
}

std::shared_ptr<ConfiguredJoystick> ConfiguredJoystick::Create(
    const std::filesystem::path& active_path) {
  ActiveSelection selection;
  try {
    selection = LoadActiveSelection(active_path);
  } catch (const GamepadConfigError& error) {
    throw GamepadConfigError(
        "active " + active_path.string() + ": " + error.what());
  }

  const auto profile_path =
      (active_path.parent_path() / selection.profile).lexically_normal();
  GamepadProfile profile = [&profile_path]() {
    try {
      return LoadGamepadProfile(profile_path);
    } catch (const GamepadConfigError& error) {
      throw GamepadConfigError(
          "profile " + profile_path.string() + ": " + error.what());
    }
  }();

  const auto descriptors = EnumerateLinuxJoysticks();
  const auto descriptor = SelectUniqueDevice(selection.device, descriptors);
  try {
    ValidateProfile(profile, descriptor);
  } catch (const GamepadConfigError& error) {
    throw GamepadConfigError(
        "profile " + profile_path.string() + ": " + error.what());
  }
  return std::make_shared<ConfiguredJoystick>(
      std::move(profile),
      std::make_unique<LinuxJoystickEventSource>(descriptor));
}

ConfiguredJoystick::ConfiguredJoystick(
    GamepadProfile profile, std::unique_ptr<JoystickEventSource> source)
    : profile_(std::move(profile)),
      source_(std::move(source)),
      lt_threshold_(TriggerThreshold(profile_.triggers.at("lt"))),
      rt_threshold_(TriggerThreshold(profile_.triggers.at("rt"))) {
  if (!source_) {
    throw GamepadConfigError("event_source: must not be null");
  }
}

void ConfiguredJoystick::Apply(const LogicalSnapshot& logical) {
  back(logical.back);
  start(logical.start);
  LS(logical.left_stick);
  RS(logical.right_stick);
  LB(logical.lb);
  RB(logical.rb);
  A(logical.a);
  B(logical.b);
  X(logical.x);
  Y(logical.y);
  up(logical.up);
  down(logical.down);
  left(logical.left);
  right(logical.right);
  F1(false);
  F2(false);
  lx(logical.lx);
  ly(logical.ly);
  rx(logical.rx);
  ry(logical.ry);
  LT(logical.lt > lt_threshold_);
  RT(logical.rt > rt_threshold_);
}

void ConfiguredJoystick::update() {
  if (disconnected_) {
    return;
  }
  const EventBatch batch = source_->Drain();
  if (!batch.connected) {
    {
      ForceImmediateAxisUpdates immediate(
          {&lx, &ly, &rx, &ry, &LT, &RT});
      Apply(LogicalSnapshot{});
    }
    disconnected_ = true;
    std::cerr << "Gamepad disconnected; input forced to neutral." << std::endl;
    return;
  }
  Apply(MapLogicalSnapshot(profile_, batch.snapshot));
}

}  // namespace gamepad
