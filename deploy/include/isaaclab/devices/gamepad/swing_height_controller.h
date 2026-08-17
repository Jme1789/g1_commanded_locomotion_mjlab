#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace isaaclab::external_control {

enum class StepHeightLevel { kLow, kMedium, kHigh };

enum class GaitHalf { kFirst, kSecond };

enum class SwingLeg { kLeft, kRight };

struct StepHeightSelectorConfig {
  float low_enter{-0.60F};
  float low_exit{-0.40F};
  float high_exit{0.40F};
  float high_enter{0.60F};
  float low_m{0.05F};
  float medium_m{0.10F};
  float high_m{0.15F};
};

class StepHeightSelector {
 public:
  explicit StepHeightSelector(StepHeightSelectorConfig config)
      : config_(config) {
    validate_config(config_);
  }

  void reset() noexcept { level_ = StepHeightLevel::kMedium; }

  void update(float logical_right_y) noexcept {
    if (!std::isfinite(logical_right_y)) {
      reset();
      return;
    }

    if (level_ == StepHeightLevel::kMedium) {
      if (logical_right_y >= config_.high_enter) {
        level_ = StepHeightLevel::kHigh;
      } else if (logical_right_y <= config_.low_enter) {
        level_ = StepHeightLevel::kLow;
      }
    } else if (level_ == StepHeightLevel::kHigh) {
      if (logical_right_y <= config_.high_exit) {
        level_ = StepHeightLevel::kMedium;
      }
    } else if (logical_right_y >= config_.low_exit) {
      level_ = StepHeightLevel::kMedium;
    }
  }

  [[nodiscard]] StepHeightLevel level() const noexcept { return level_; }

  [[nodiscard]] float value_m() const noexcept {
    if (level_ == StepHeightLevel::kLow) {
      return config_.low_m;
    }
    if (level_ == StepHeightLevel::kHigh) {
      return config_.high_m;
    }
    return config_.medium_m;
  }

 private:
  static void validate_config(const StepHeightSelectorConfig& config) {
    if (!std::isfinite(config.low_enter) ||
        !std::isfinite(config.low_exit) ||
        !std::isfinite(config.high_exit) ||
        !std::isfinite(config.high_enter) ||
        !std::isfinite(config.low_m) ||
        !std::isfinite(config.medium_m) ||
        !std::isfinite(config.high_m) ||
        !(config.low_enter < config.low_exit &&
          config.low_exit < config.high_exit &&
          config.high_exit < config.high_enter) ||
        !(0.0F < config.low_m && config.low_m < config.medium_m &&
          config.medium_m < config.high_m)) {
      throw std::invalid_argument("invalid step height selector configuration");
    }
  }

  StepHeightSelectorConfig config_;
  StepHeightLevel level_{StepHeightLevel::kMedium};
};

struct JointTriplet {
  std::size_t hip_pitch;
  std::size_t knee;
  std::size_t ankle_pitch;
};

struct JointLimit {
  float minimum;
  float maximum;
};

struct SwingHeightConfig {
  bool enabled{false};
  std::size_t action_dim{29};
  SwingLeg first_half_swing_leg{SwingLeg::kLeft};
  JointTriplet left{0, 3, 4};
  JointTriplet right{6, 9, 10};
  std::array<float, 3> low{0.025F, -0.05F, 0.025F};
  std::array<float, 3> medium{0.0F, 0.0F, 0.0F};
  std::array<float, 3> high{-0.08F, 0.16F, -0.08F};
  std::array<float, 3> max_abs_offset{0.12F, 0.24F, 0.12F};
  std::array<float, 3> max_delta_per_tick{0.02F, 0.04F, 0.02F};
  std::array<JointLimit, 3> joint_limits{{
      {-2.5307F, 2.8798F},
      {-0.087267F, 2.8798F},
      {-0.87267F, 0.5236F},
  }};
  float step_dt{0.02F};
  float gait_period{0.6F};
  float active_phase_advance{0.5F};
};

struct SwingHeightInput {
  bool active{false};
  bool forward_step{false};
  StepHeightLevel level{StepHeightLevel::kMedium};
  GaitHalf half{GaitHalf::kFirst};
  float progress{0.0F};
};

class SwingHeightController {
 public:
  explicit SwingHeightController(SwingHeightConfig config) : config_(config) {
    validate_config(config_);
  }

  [[nodiscard]] std::vector<float> apply(
      const std::vector<float>& policy_targets,
      const SwingHeightInput& input) const {
    if (policy_targets.size() != config_.action_dim) {
      throw std::invalid_argument("unexpected policy target dimension");
    }
    if (!config_.enabled || !input.active || !input.forward_step ||
        input.level == StepHeightLevel::kMedium ||
        !valid_level(input.level) || !valid_half(input.half) ||
        !std::isfinite(input.progress) || input.progress <= 0.0F ||
        input.progress >= 1.0F) {
      return policy_targets;
    }

    const auto& profile =
        input.level == StepHeightLevel::kLow ? config_.low : config_.high;
    const SwingLeg leg = swing_leg_for(input.half);
    const JointTriplet indices =
        leg == SwingLeg::kLeft ? config_.left : config_.right;
    const std::array<std::size_t, 3> target_indices{
        indices.hip_pitch,
        indices.knee,
        indices.ankle_pitch,
    };
    const float wave = std::sin(kPi * input.progress);
    const float envelope = wave * wave;
    auto result = policy_targets;
    for (std::size_t index = 0; index < target_indices.size(); ++index) {
      const std::size_t target_index = target_indices[index];
      result[target_index] = std::clamp(
          policy_targets[target_index] + envelope * profile[index],
          config_.joint_limits[index].minimum,
          config_.joint_limits[index].maximum);
    }
    return result;
  }

  [[nodiscard]] SwingLeg swing_leg_for(GaitHalf half) const noexcept {
    if (half == GaitHalf::kFirst) {
      return config_.first_half_swing_leg;
    }
    return config_.first_half_swing_leg == SwingLeg::kLeft
               ? SwingLeg::kRight
               : SwingLeg::kLeft;
  }

 private:
  static constexpr float kPi = 3.14159265358979323846F;

  static bool valid_level(StepHeightLevel level) noexcept {
    return level == StepHeightLevel::kLow ||
           level == StepHeightLevel::kMedium ||
           level == StepHeightLevel::kHigh;
  }

  static bool valid_half(GaitHalf half) noexcept {
    return half == GaitHalf::kFirst || half == GaitHalf::kSecond;
  }

  static bool valid_leg(SwingLeg leg) noexcept {
    return leg == SwingLeg::kLeft || leg == SwingLeg::kRight;
  }

  static bool finite_array(const std::array<float, 3>& values) noexcept {
    return std::all_of(values.begin(), values.end(), [](float value) {
      return std::isfinite(value);
    });
  }

  static void validate_config(const SwingHeightConfig& config) {
    if (config.action_dim == 0 || !valid_leg(config.first_half_swing_leg) ||
        !finite_array(config.low) || !finite_array(config.medium) ||
        !finite_array(config.high) ||
        !finite_array(config.max_abs_offset) ||
        !finite_array(config.max_delta_per_tick) ||
        !std::isfinite(config.step_dt) ||
        !std::isfinite(config.gait_period) ||
        !std::isfinite(config.active_phase_advance) ||
        config.step_dt <= 0.0F || config.gait_period <= 0.0F ||
        config.active_phase_advance <= 0.0F ||
        config.active_phase_advance > 1.0F) {
      throw std::invalid_argument("invalid swing height configuration");
    }

    const std::array<std::size_t, 6> indices{
        config.left.hip_pitch,   config.left.knee,
        config.left.ankle_pitch, config.right.hip_pitch,
        config.right.knee,       config.right.ankle_pitch,
    };
    for (std::size_t outer = 0; outer < indices.size(); ++outer) {
      if (indices[outer] >= config.action_dim) {
        throw std::invalid_argument("swing height joint index is out of range");
      }
      for (std::size_t inner = outer + 1; inner < indices.size(); ++inner) {
        if (indices[outer] == indices[inner]) {
          throw std::invalid_argument("swing height joint indices must be unique");
        }
      }
    }

    const float progress_per_tick =
        config.step_dt /
        (config.gait_period * config.active_phase_advance);
    if (!std::isfinite(progress_per_tick) || progress_per_tick > 1.0F) {
      throw std::invalid_argument("invalid swing height timing");
    }
    const float worst_envelope_step = kPi * progress_per_tick;
    for (std::size_t index = 0; index < 3; ++index) {
      const JointLimit limit = config.joint_limits[index];
      if (!std::isfinite(limit.minimum) || !std::isfinite(limit.maximum) ||
          !(limit.minimum < limit.maximum) ||
          config.medium[index] != 0.0F ||
          config.max_abs_offset[index] <= 0.0F ||
          config.max_delta_per_tick[index] <= 0.0F) {
        throw std::invalid_argument("invalid swing height joint bounds");
      }
      for (const auto* profile : {&config.low, &config.high}) {
        if (std::fabs((*profile)[index]) > config.max_abs_offset[index] ||
            std::fabs((*profile)[index]) * worst_envelope_step >
                config.max_delta_per_tick[index]) {
          throw std::invalid_argument("swing height profile exceeds bounds");
        }
      }
    }
  }

  SwingHeightConfig config_;
};

[[nodiscard]] inline const char* to_string(StepHeightLevel level) noexcept {
  switch (level) {
    case StepHeightLevel::kLow:
      return "low";
    case StepHeightLevel::kMedium:
      return "medium";
    case StepHeightLevel::kHigh:
      return "high";
  }
  return "unknown";
}

[[nodiscard]] inline const char* to_string(GaitHalf half) noexcept {
  switch (half) {
    case GaitHalf::kFirst:
      return "first";
    case GaitHalf::kSecond:
      return "second";
  }
  return "unknown";
}

[[nodiscard]] inline const char* to_string(SwingLeg leg) noexcept {
  switch (leg) {
    case SwingLeg::kLeft:
      return "left";
    case SwingLeg::kRight:
      return "right";
  }
  return "unknown";
}

}  // namespace isaaclab::external_control
