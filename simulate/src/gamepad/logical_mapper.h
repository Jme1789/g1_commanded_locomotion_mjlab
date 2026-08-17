#pragma once

#include <cstdint>
#include <vector>

#include "gamepad/gamepad_profile.h"

namespace gamepad {

struct RawSnapshot {
  std::vector<std::int16_t> axes;
  std::vector<std::uint8_t> buttons;
};

struct LogicalSnapshot {
  float lx{0.0F}, ly{0.0F}, rx{0.0F}, ry{0.0F};
  float lt{0.0F}, rt{0.0F};
  bool up{false}, down{false}, left{false}, right{false};
  bool a{false}, b{false}, x{false}, y{false};
  bool lb{false}, rb{false}, start{false}, back{false};
  bool left_stick{false}, right_stick{false};
};

LogicalSnapshot MapLogicalSnapshot(const GamepadProfile& profile,
                                   const RawSnapshot& raw);

}  // namespace gamepad
