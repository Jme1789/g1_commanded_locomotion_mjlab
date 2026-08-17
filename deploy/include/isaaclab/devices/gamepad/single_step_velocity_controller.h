#pragma once

#include "isaaclab/devices/gamepad/swing_height_controller.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace isaaclab::external_control {

using VelocityCommand = std::array<float, 3>;

enum class DpadDirection {
  kNone,
  kUp,
  kDown,
  kLeft,
  kRight,
};

struct DpadLevels {
  bool up{false};
  bool down{false};
  bool left{false};
  bool right{false};
};

enum class SingleStepState {
  kIdle,
  kArmed,
  kActive,
  kWaitRelease,
  kAborted,
};

enum class SingleStepAbortReason {
  kNone,
  kArmedTimeout,
  kActiveTimeout,
  kInvalidInput,
  kExternalStop,
};

struct SingleStepConfig {
  VelocityCommand up_command{0.5F, 0.0F, 0.0F};
  VelocityCommand down_command{-0.5F, 0.0F, 0.0F};
  VelocityCommand left_command{0.0F, 0.5F, 0.0F};
  VelocityCommand right_command{0.0F, -0.5F, 0.0F};
  float phase_advance{0.5F};
  float armed_timeout_s{0.7F};
  float active_timeout_s{0.7F};
};

struct SingleStepInput {
  DpadLevels dpad{};
  float gait_phase{0.0F};
  float dt{0.02F};
  VelocityCommand passthrough_command{0.0F, 0.0F, 0.0F};
  StepHeightLevel requested_height{StepHeightLevel::kMedium};
};

class SingleStepVelocityController {
 public:
  explicit SingleStepVelocityController(SingleStepConfig config)
      : config_(config) {
    validate_config(config_);
  }

  void reset(DpadLevels dpad, float gait_phase) {
    timer_s_ = 0.0F;
    phase_progress_ = 0.0F;
    active_half_ = GaitHalf::kFirst;
    previous_phase_ = normalize_phase(gait_phase);
    previous_dpad_ = dpad;
    latched_direction_ = DpadDirection::kNone;
    latched_height_ = StepHeightLevel::kMedium;
    latched_command_ = {0.0F, 0.0F, 0.0F};
    abort_reason_ = SingleStepAbortReason::kNone;
    command_ = {0.0F, 0.0F, 0.0F};
    state_ = any_pressed(dpad) ? SingleStepState::kWaitRelease
                               : SingleStepState::kIdle;
  }

  void update(const SingleStepInput& input) {
    if (!std::isfinite(input.dt) || input.dt <= 0.0F ||
        !std::isfinite(input.gait_phase) ||
        !valid_height_level(input.requested_height)) {
      abort(SingleStepAbortReason::kInvalidInput);
      return;
    }

    const float current_phase = normalize_phase(input.gait_phase);
    const bool crossed_boundary =
        crossed_half_cycle_boundary(previous_phase_, current_phase);

    if (state_ == SingleStepState::kIdle) {
      const auto direction = unique_direction(input.dpad);
      if (direction != DpadDirection::kNone &&
          !is_pressed(previous_dpad_, direction)) {
        state_ = SingleStepState::kArmed;
        timer_s_ = 0.0F;
        phase_progress_ = 0.0F;
        latched_direction_ = direction;
        latched_height_ = direction == DpadDirection::kUp
                              ? input.requested_height
                              : StepHeightLevel::kMedium;
        latched_command_ = command_for(direction);
        command_ = {0.0F, 0.0F, 0.0F};
      } else {
        command_ = input.passthrough_command;
      }
    } else if (state_ == SingleStepState::kArmed) {
      if (crossed_boundary) {
        state_ = SingleStepState::kActive;
        phase_progress_ = 0.0F;
        active_half_ = current_phase < 0.5F ? GaitHalf::kFirst
                                            : GaitHalf::kSecond;
        timer_s_ = 0.0F;
        command_ = latched_command_;
      } else if ((timer_s_ += input.dt) >= config_.armed_timeout_s) {
        abort(SingleStepAbortReason::kArmedTimeout);
      } else {
        command_ = {0.0F, 0.0F, 0.0F};
      }
    } else if (state_ == SingleStepState::kActive) {
      phase_progress_ += wrapped_phase_delta(previous_phase_, current_phase);
      if (phase_progress_ >= config_.phase_advance) {
        state_ = SingleStepState::kWaitRelease;
        timer_s_ = 0.0F;
        command_ = {0.0F, 0.0F, 0.0F};
      } else if ((timer_s_ += input.dt) >= config_.active_timeout_s) {
        abort(SingleStepAbortReason::kActiveTimeout);
      } else {
        command_ = latched_command_;
      }
    } else if (state_ == SingleStepState::kWaitRelease) {
      if (!any_pressed(input.dpad)) {
        state_ = SingleStepState::kIdle;
        latched_direction_ = DpadDirection::kNone;
        latched_height_ = StepHeightLevel::kMedium;
        latched_command_ = {0.0F, 0.0F, 0.0F};
        command_ = input.passthrough_command;
      } else {
        command_ = {0.0F, 0.0F, 0.0F};
      }
    } else if (!any_pressed(input.dpad)) {
      state_ = SingleStepState::kIdle;
      abort_reason_ = SingleStepAbortReason::kNone;
      latched_direction_ = DpadDirection::kNone;
      latched_height_ = StepHeightLevel::kMedium;
      latched_command_ = {0.0F, 0.0F, 0.0F};
      command_ = input.passthrough_command;
    } else {
      command_ = {0.0F, 0.0F, 0.0F};
    }

    previous_phase_ = current_phase;
    previous_dpad_ = input.dpad;
  }

  void abort(
      SingleStepAbortReason reason = SingleStepAbortReason::kExternalStop) noexcept {
    state_ = SingleStepState::kAborted;
    abort_reason_ = reason;
    timer_s_ = 0.0F;
    phase_progress_ = 0.0F;
    active_half_ = GaitHalf::kFirst;
    latched_direction_ = DpadDirection::kNone;
    latched_height_ = StepHeightLevel::kMedium;
    latched_command_ = {0.0F, 0.0F, 0.0F};
    command_ = {0.0F, 0.0F, 0.0F};
  }

  [[nodiscard]] const VelocityCommand& command() const noexcept {
    return command_;
  }
  [[nodiscard]] SingleStepState state() const noexcept { return state_; }
  [[nodiscard]] SingleStepAbortReason abort_reason() const noexcept {
    return abort_reason_;
  }
  [[nodiscard]] DpadDirection latched_direction() const noexcept {
    return latched_direction_;
  }
  [[nodiscard]] StepHeightLevel latched_height() const noexcept {
    return latched_height_;
  }
  [[nodiscard]] GaitHalf active_half() const noexcept { return active_half_; }
  [[nodiscard]] float active_progress() const noexcept {
    if (state_ != SingleStepState::kActive) {
      return 0.0F;
    }
    return std::clamp(phase_progress_ / config_.phase_advance, 0.0F, 1.0F);
  }

 private:
  static bool any_pressed(const DpadLevels& levels) noexcept {
    return levels.up || levels.down || levels.left || levels.right;
  }

  static bool valid_height_level(StepHeightLevel level) noexcept {
    return level == StepHeightLevel::kLow ||
           level == StepHeightLevel::kMedium ||
           level == StepHeightLevel::kHigh;
  }

  static DpadDirection unique_direction(const DpadLevels& levels) noexcept {
    const int pressed_count =
        static_cast<int>(levels.up) + static_cast<int>(levels.down) +
        static_cast<int>(levels.left) + static_cast<int>(levels.right);
    if (pressed_count != 1) {
      return DpadDirection::kNone;
    }
    if (levels.up) {
      return DpadDirection::kUp;
    }
    if (levels.down) {
      return DpadDirection::kDown;
    }
    if (levels.left) {
      return DpadDirection::kLeft;
    }
    return DpadDirection::kRight;
  }

  static bool is_pressed(const DpadLevels& levels,
                         DpadDirection direction) noexcept {
    switch (direction) {
      case DpadDirection::kUp:
        return levels.up;
      case DpadDirection::kDown:
        return levels.down;
      case DpadDirection::kLeft:
        return levels.left;
      case DpadDirection::kRight:
        return levels.right;
      case DpadDirection::kNone:
        return false;
    }
    return false;
  }

  VelocityCommand command_for(DpadDirection direction) const {
    switch (direction) {
      case DpadDirection::kUp:
        return config_.up_command;
      case DpadDirection::kDown:
        return config_.down_command;
      case DpadDirection::kLeft:
        return config_.left_command;
      case DpadDirection::kRight:
        return config_.right_command;
      case DpadDirection::kNone:
        return {0.0F, 0.0F, 0.0F};
    }
    return {0.0F, 0.0F, 0.0F};
  }

  static float normalize_phase(float phase) {
    const float normalized = std::fmod(phase, 1.0F);
    return normalized < 0.0F ? normalized + 1.0F : normalized;
  }

  static float wrapped_phase_delta(float previous, float current) {
    return current >= previous ? current - previous
                               : current + 1.0F - previous;
  }

  static bool crossed_half_cycle_boundary(float previous, float current) {
    const int previous_bucket = static_cast<int>(previous * 2.0F);
    const int current_bucket = static_cast<int>(current * 2.0F);
    return previous_bucket != current_bucket;
  }

  static bool command_is_finite(const VelocityCommand& command) {
    for (const float value : command) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
    return true;
  }

  static void validate_config(const SingleStepConfig& config) {
    if (!command_is_finite(config.up_command) ||
        !command_is_finite(config.down_command) ||
        !command_is_finite(config.left_command) ||
        !command_is_finite(config.right_command) ||
        !std::isfinite(config.phase_advance) ||
        !std::isfinite(config.armed_timeout_s) ||
        !std::isfinite(config.active_timeout_s) ||
        config.phase_advance <= 0.0F || config.phase_advance > 1.0F ||
        config.armed_timeout_s <= 0.0F || config.active_timeout_s <= 0.0F) {
      throw std::invalid_argument(
          "invalid single-step controller configuration");
    }
  }

  SingleStepConfig config_;
  VelocityCommand command_{0.0F, 0.0F, 0.0F};
  VelocityCommand latched_command_{0.0F, 0.0F, 0.0F};
  SingleStepState state_{SingleStepState::kIdle};
  SingleStepAbortReason abort_reason_{SingleStepAbortReason::kNone};
  DpadDirection latched_direction_{DpadDirection::kNone};
  StepHeightLevel latched_height_{StepHeightLevel::kMedium};
  GaitHalf active_half_{GaitHalf::kFirst};
  float timer_s_{0.0F};
  float phase_progress_{0.0F};
  float previous_phase_{0.0F};
  DpadLevels previous_dpad_{};
};

[[nodiscard]] inline const char* to_string(DpadDirection direction) noexcept {
  switch (direction) {
    case DpadDirection::kNone:
      return "none";
    case DpadDirection::kUp:
      return "up";
    case DpadDirection::kDown:
      return "down";
    case DpadDirection::kLeft:
      return "left";
    case DpadDirection::kRight:
      return "right";
  }
  return "unknown";
}

[[nodiscard]] inline const char* to_string(SingleStepState state) noexcept {
  switch (state) {
    case SingleStepState::kIdle:
      return "idle";
    case SingleStepState::kArmed:
      return "armed";
    case SingleStepState::kActive:
      return "active";
    case SingleStepState::kWaitRelease:
      return "wait_release";
    case SingleStepState::kAborted:
      return "aborted";
  }
  return "unknown";
}

[[nodiscard]] inline const char* to_string(
    SingleStepAbortReason reason) noexcept {
  switch (reason) {
    case SingleStepAbortReason::kNone:
      return "none";
    case SingleStepAbortReason::kArmedTimeout:
      return "armed_timeout";
    case SingleStepAbortReason::kActiveTimeout:
      return "active_timeout";
    case SingleStepAbortReason::kInvalidInput:
      return "invalid_input";
    case SingleStepAbortReason::kExternalStop:
      return "external_stop";
  }
  return "unknown";
}

}  // namespace isaaclab::external_control
