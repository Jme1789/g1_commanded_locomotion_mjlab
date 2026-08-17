#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace gamepad {

class GamepadConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct DeviceIdentity {
  std::string vendor_id;
  std::string product_id;
  std::string name;
  std::optional<std::string> serial;
};

struct DeviceDescriptor {
  std::filesystem::path path;
  DeviceIdentity identity;
  std::size_t axis_count;
  std::size_t button_count;
  std::optional<std::filesystem::path> by_id_path;
};

struct UnsupportedBinding {};

struct CorrelatedButton {
  std::size_t index;
  int observed_within_ms;
};

struct StickBinding {
  std::size_t axis;
  std::int16_t center;
  std::int16_t minimum;
  std::int16_t maximum;
  bool invert;
  float deadzone;
};

struct AxisTriggerBinding {
  std::size_t index;
  std::int16_t released;
  std::int16_t pressed;
  float threshold;
  std::optional<CorrelatedButton> correlated_button;
};

struct ButtonTriggerBinding {
  std::size_t index;
  float threshold;
};

using TriggerBinding =
    std::variant<AxisTriggerBinding, ButtonTriggerBinding, UnsupportedBinding>;

struct ButtonBinding {
  std::size_t index;
};

enum class DpadDirection {
  kNegative,
  kPositive,
};

struct AxisDpadBinding {
  std::size_t index;
  DpadDirection direction;
  float threshold;
};

using DpadBinding =
    std::variant<AxisDpadBinding, ButtonBinding, UnsupportedBinding>;

struct GamepadProfile {
  int schema_version;
  DeviceIdentity device;
  std::map<std::string, std::variant<StickBinding, UnsupportedBinding>> sticks;
  std::map<std::string, TriggerBinding> triggers;
  std::map<std::string, std::variant<ButtonBinding, UnsupportedBinding>> buttons;
  std::map<std::string, DpadBinding> dpad;
};

struct ActiveSelection {
  int schema_version;
  std::filesystem::path profile;
  DeviceIdentity device;
};

ActiveSelection LoadActiveSelection(const std::filesystem::path& path);
GamepadProfile LoadGamepadProfile(const std::filesystem::path& path);
void ValidateProfile(const GamepadProfile& profile,
                     const DeviceDescriptor& descriptor);

}  // namespace gamepad
