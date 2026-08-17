#pragma once

#include "isaaclab/devices/gamepad/single_step_velocity_controller.h"

#include <cmath>
#include <stdexcept>

namespace isaaclab::external_control {

enum class StepLengthLevel { kShort, kMedium, kLong };

[[nodiscard]] inline const char* to_string(StepLengthLevel level) noexcept {
  switch (level) {
    case StepLengthLevel::kShort:
      return "short";
    case StepLengthLevel::kMedium:
      return "medium";
    case StepLengthLevel::kLong:
      return "long";
  }
  return "unknown";
}

struct StepLengthSelectorConfig {
  float short_enter{-0.6F};
  float short_exit{-0.4F};
  float long_exit{0.4F};
  float long_enter{0.6F};
  float short_m{0.2F};
  float medium_m{0.3F};
  float long_m{0.4F};
};

class StepLengthSelector {
 public:
  explicit StepLengthSelector(StepLengthSelectorConfig config)
      : config_(config) {
    validate(config_);
  }

  void reset() noexcept { level_ = StepLengthLevel::kMedium; }

  void update(float logical_right_x) noexcept {
    if (!std::isfinite(logical_right_x)) {
      reset();
      return;
    }
    if (level_ == StepLengthLevel::kMedium) {
      if (logical_right_x >= config_.long_enter) {
        level_ = StepLengthLevel::kLong;
      } else if (logical_right_x <= config_.short_enter) {
        level_ = StepLengthLevel::kShort;
      }
    } else if (level_ == StepLengthLevel::kLong) {
      if (logical_right_x <= config_.long_exit) {
        level_ = StepLengthLevel::kMedium;
      }
    } else if (logical_right_x >= config_.short_exit) {
      level_ = StepLengthLevel::kMedium;
    }
  }

  [[nodiscard]] StepLengthLevel level() const noexcept { return level_; }

  [[nodiscard]] float value_m() const noexcept {
    if (level_ == StepLengthLevel::kShort) {
      return config_.short_m;
    }
    if (level_ == StepLengthLevel::kLong) {
      return config_.long_m;
    }
    return config_.medium_m;
  }

 private:
  static void validate(const StepLengthSelectorConfig& config) {
    const float values[] = {
        config.short_enter, config.short_exit, config.long_exit,
        config.long_enter, config.short_m,     config.medium_m,
        config.long_m,
    };
    for (const float value : values) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("invalid step-length selector configuration");
      }
    }
    if (!(config.short_enter < config.short_exit &&
          config.short_exit < config.long_exit &&
          config.long_exit < config.long_enter &&
          0.0F < config.short_m && config.short_m < config.medium_m &&
          config.medium_m < config.long_m)) {
      throw std::invalid_argument("invalid step-length selector configuration");
    }
  }

  StepLengthSelectorConfig config_;
  StepLengthLevel level_{StepLengthLevel::kMedium};
};

[[nodiscard]] inline VelocityCommand with_shoulder_yaw(
    VelocityCommand command, bool lb, bool rb, bool lt, float speed,
    float boost_multiplier, bool yaw_enabled = true) {
  if (!std::isfinite(speed) || speed < 0.0F) {
    throw std::invalid_argument("shoulder yaw speed must be finite and non-negative");
  }
  if (!std::isfinite(boost_multiplier) || boost_multiplier < 1.0F) {
    throw std::invalid_argument(
        "yaw boost multiplier must be finite and at least one");
  }
  const float effective_speed = lt ? speed * boost_multiplier : speed;
  if (!std::isfinite(effective_speed)) {
    throw std::invalid_argument("boosted shoulder yaw speed must be finite");
  }
  command[2] = !yaw_enabled || lb == rb
                   ? 0.0F
                   : (lb ? effective_speed : -effective_speed);
  return command;
}

}  // namespace isaaclab::external_control
