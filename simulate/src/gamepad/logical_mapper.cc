#include "gamepad/logical_mapper.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <variant>

namespace gamepad {
namespace {

std::int16_t AxisAt(const RawSnapshot& raw, std::size_t index,
                    const std::string& path) {
  if (index >= raw.axes.size()) {
    throw GamepadConfigError(path + ": axis index exceeds raw snapshot");
  }
  return raw.axes[index];
}

bool ButtonAt(const RawSnapshot& raw, std::size_t index,
              const std::string& path) {
  if (index >= raw.buttons.size()) {
    throw GamepadConfigError(path + ": button index exceeds raw snapshot");
  }
  return raw.buttons[index] != 0U;
}

float NormalizeSignedAxis(std::int16_t value) {
  const float denominator = value < 0 ? 32768.0F : 32767.0F;
  return std::clamp(static_cast<float>(value) / denominator, -1.0F,
                    1.0F);
}

float NormalizeStick(std::int16_t value, const StickBinding& binding) {
  const float denominator =
      value < binding.center
          ? static_cast<float>(binding.center - binding.minimum)
          : static_cast<float>(binding.maximum - binding.center);
  float scaled = denominator <= 0.0F
                     ? 0.0F
                     : static_cast<float>(value - binding.center) / denominator;
  scaled = std::clamp(scaled, -1.0F, 1.0F);
  if (binding.invert) {
    scaled = -scaled;
  }
  return std::fabs(scaled) < binding.deadzone ? 0.0F : scaled;
}

float NormalizeTrigger(std::int16_t value,
                       const AxisTriggerBinding& binding) {
  const float denominator =
      static_cast<float>(binding.pressed - binding.released);
  const float scaled = denominator == 0.0F
                           ? 0.0F
                           : static_cast<float>(value - binding.released) /
                                 denominator;
  return std::clamp(scaled, 0.0F, 1.0F);
}

float MapStick(const GamepadProfile& profile, const RawSnapshot& raw,
               const std::string& name) {
  const auto& binding = profile.sticks.at(name);
  const auto* stick = std::get_if<StickBinding>(&binding);
  if (stick == nullptr) {
    return 0.0F;
  }
  return NormalizeStick(AxisAt(raw, stick->axis, "sticks." + name + ".axis"),
                        *stick);
}

float MapTrigger(const GamepadProfile& profile, const RawSnapshot& raw,
                 const std::string& name) {
  const auto& binding = profile.triggers.at(name);
  if (const auto* axis = std::get_if<AxisTriggerBinding>(&binding)) {
    return NormalizeTrigger(
        AxisAt(raw, axis->index, "triggers." + name + ".index"), *axis);
  }
  if (const auto* button = std::get_if<ButtonTriggerBinding>(&binding)) {
    return ButtonAt(raw, button->index, "triggers." + name + ".index") ? 1.0F
                                                                        : 0.0F;
  }
  return 0.0F;
}

bool MapButton(const GamepadProfile& profile, const RawSnapshot& raw,
               const std::string& name) {
  const auto& binding = profile.buttons.at(name);
  const auto* button = std::get_if<ButtonBinding>(&binding);
  return button != nullptr &&
         ButtonAt(raw, button->index, "buttons." + name + ".index");
}

bool MapDpad(const GamepadProfile& profile, const RawSnapshot& raw,
             const std::string& name) {
  const auto& binding = profile.dpad.at(name);
  if (const auto* button = std::get_if<ButtonBinding>(&binding)) {
    return ButtonAt(raw, button->index, "dpad." + name + ".index");
  }
  if (const auto* axis = std::get_if<AxisDpadBinding>(&binding)) {
    const auto value = NormalizeSignedAxis(
        AxisAt(raw, axis->index, "dpad." + name + ".index"));
    return axis->direction == DpadDirection::kNegative
               ? value <= -axis->threshold
               : value >= axis->threshold;
  }
  return false;
}

}  // namespace

LogicalSnapshot MapLogicalSnapshot(const GamepadProfile& profile,
                                   const RawSnapshot& raw) {
  LogicalSnapshot logical;
  logical.lx = MapStick(profile, raw, "left_x");
  logical.ly = MapStick(profile, raw, "left_y");
  logical.rx = MapStick(profile, raw, "right_x");
  logical.ry = MapStick(profile, raw, "right_y");
  logical.lt = MapTrigger(profile, raw, "lt");
  logical.rt = MapTrigger(profile, raw, "rt");
  logical.up = MapDpad(profile, raw, "up");
  logical.down = MapDpad(profile, raw, "down");
  logical.left = MapDpad(profile, raw, "left");
  logical.right = MapDpad(profile, raw, "right");
  logical.a = MapButton(profile, raw, "a");
  logical.b = MapButton(profile, raw, "b");
  logical.x = MapButton(profile, raw, "x");
  logical.y = MapButton(profile, raw, "y");
  logical.lb = MapButton(profile, raw, "lb");
  logical.rb = MapButton(profile, raw, "rb");
  logical.start = MapButton(profile, raw, "start");
  logical.back = MapButton(profile, raw, "back");
  logical.left_stick = MapButton(profile, raw, "left_stick");
  logical.right_stick = MapButton(profile, raw, "right_stick");
  return logical;
}

}  // namespace gamepad
