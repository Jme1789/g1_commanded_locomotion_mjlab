#include "isaaclab/devices/gamepad/single_step_velocity_controller.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

using isaaclab::external_control::DpadDirection;
using isaaclab::external_control::DpadLevels;
using isaaclab::external_control::GaitHalf;
using isaaclab::external_control::SingleStepAbortReason;
using isaaclab::external_control::SingleStepConfig;
using isaaclab::external_control::SingleStepInput;
using isaaclab::external_control::SingleStepState;
using isaaclab::external_control::SingleStepVelocityController;
using isaaclab::external_control::StepHeightLevel;
using isaaclab::external_control::VelocityCommand;

void expect_true(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void expect_state(const SingleStepVelocityController& controller,
                  SingleStepState expected) {
  expect_true(controller.state() == expected, "unexpected controller state");
}

void expect_abort_reason(const SingleStepVelocityController& controller,
                         SingleStepAbortReason expected) {
  expect_true(controller.abort_reason() == expected, "unexpected abort reason");
}

void expect_command(const SingleStepVelocityController& controller,
                    const VelocityCommand& expected) {
  const auto& actual = controller.command();
  for (std::size_t index = 0; index < actual.size(); ++index) {
    expect_true(std::fabs(actual[index] - expected[index]) <= 1e-6F,
                "unexpected controller command");
  }
}

DpadLevels levels_for(DpadDirection direction) {
  switch (direction) {
    case DpadDirection::kUp:
      return {true, false, false, false};
    case DpadDirection::kDown:
      return {false, true, false, false};
    case DpadDirection::kLeft:
      return {false, false, true, false};
    case DpadDirection::kRight:
      return {false, false, false, true};
    case DpadDirection::kNone:
      return {};
  }
  return {};
}

SingleStepInput frame(
    DpadLevels dpad, float phase, float dt = 0.02F,
    VelocityCommand passthrough = {0.3F, 0.0F, 0.0F},
    StepHeightLevel height = StepHeightLevel::kMedium) {
  return {dpad, phase, dt, passthrough, height};
}

void test_idle_passes_through_without_mutating_reads() {
  SingleStepVelocityController controller({});
  controller.reset({}, 0.2F);
  controller.update(frame({}, 0.22F, 0.02F, {0.25F, -0.1F, 0.3F}));

  expect_state(controller, SingleStepState::kIdle);
  expect_command(controller, {0.25F, -0.1F, 0.3F});
  const auto state_before_reads = controller.state();
  const auto command_1 = controller.command();
  const auto command_2 = controller.command();
  expect_true(command_1 == command_2, "command reads must be immutable");
  expect_true(controller.state() == state_before_reads,
              "command reads must not advance state");
}

void test_reset_while_any_direction_is_held_requires_full_release() {
  const DpadDirection directions[] = {
      DpadDirection::kUp,
      DpadDirection::kDown,
      DpadDirection::kLeft,
      DpadDirection::kRight,
  };
  for (const auto direction : directions) {
    SingleStepVelocityController controller({});
    const auto held = levels_for(direction);
    controller.reset(held, 0.2F);
    expect_state(controller, SingleStepState::kWaitRelease);
    expect_command(controller, {0.0F, 0.0F, 0.0F});

    controller.update(frame(held, 0.25F, 0.02F, {0.4F, 0.0F, 0.0F}));
    expect_state(controller, SingleStepState::kWaitRelease);
    controller.update(frame({}, 0.28F, 0.02F, {0.4F, 0.0F, 0.0F}));
    expect_state(controller, SingleStepState::kIdle);
    expect_command(controller, {0.4F, 0.0F, 0.0F});
  }
}

void test_press_arms_then_waits_for_next_boundary() {
  SingleStepVelocityController controller({});
  controller.reset({}, 0.40F);

  controller.update(frame(levels_for(DpadDirection::kUp), 0.48F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kArmed);
  expect_command(controller, {0.0F, 0.0F, 0.0F});

  controller.update(frame(levels_for(DpadDirection::kUp), 0.49F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kArmed);

  controller.update(frame(levels_for(DpadDirection::kUp), 0.51F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kActive);
  expect_command(controller, {0.5F, 0.0F, 0.0F});
}

void test_armed_activates_at_a_wrapped_half_cycle_boundary() {
  SingleStepVelocityController controller({});
  controller.reset({}, 0.90F);

  controller.update(frame(levels_for(DpadDirection::kUp), 0.99F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kArmed);

  controller.update(frame(levels_for(DpadDirection::kUp), 0.01F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kActive);
  expect_command(controller, {0.5F, 0.0F, 0.0F});
}

void test_press_on_boundary_waits_for_following_boundary() {
  SingleStepVelocityController controller({});
  controller.reset({}, 0.40F);

  controller.update(frame(levels_for(DpadDirection::kUp), 0.51F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kArmed);
  expect_command(controller, {0.0F, 0.0F, 0.0F});

  controller.update(frame(levels_for(DpadDirection::kUp), 0.99F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kArmed);

  controller.update(frame(levels_for(DpadDirection::kUp), 0.01F,
                          0.02F, {0.2F, 0.0F, 0.0F}));
  expect_state(controller, SingleStepState::kActive);
  expect_command(controller, {0.5F, 0.0F, 0.0F});
}

void test_active_step_runs_to_completion_after_direction_is_released() {
  SingleStepVelocityController controller({});
  controller.reset({}, 0.40F);
  controller.update(frame(levels_for(DpadDirection::kUp), 0.48F));
  controller.update(frame(levels_for(DpadDirection::kUp), 0.51F));
  expect_state(controller, SingleStepState::kActive);

  controller.update(frame({}, 0.75F));
  expect_command(controller, {0.5F, 0.0F, 0.0F});
  const auto command_1 = controller.command();
  const auto command_2 = controller.command();
  expect_true(command_1 == command_2,
              "command reads must not add phase progress");

  controller.update(frame({}, 0.99F));
  expect_command(controller, {0.5F, 0.0F, 0.0F});

  controller.update(frame({}, 0.02F));
  expect_state(controller, SingleStepState::kWaitRelease);
  expect_command(controller, {0.0F, 0.0F, 0.0F});

  controller.update(frame({}, 0.04F));
  expect_state(controller, SingleStepState::kIdle);
  expect_command(controller, {0.3F, 0.0F, 0.0F});
}

void test_held_direction_cannot_retrigger_after_active_completion() {
  SingleStepVelocityController controller({});
  const auto held = levels_for(DpadDirection::kDown);
  controller.reset({}, 0.40F);
  controller.update(frame(held, 0.48F));
  controller.update(frame(held, 0.51F));
  expect_state(controller, SingleStepState::kActive);

  controller.update(frame(held, 0.01F));
  expect_state(controller, SingleStepState::kWaitRelease);
  expect_command(controller, {0.0F, 0.0F, 0.0F});

  controller.update(frame(held, 0.10F));
  expect_state(controller, SingleStepState::kWaitRelease);
  expect_command(controller, {0.0F, 0.0F, 0.0F});
}

void test_all_dpad_directions_latch_the_expected_command() {
  struct Case {
    DpadDirection direction;
    VelocityCommand expected;
  };
  const Case cases[] = {
      {DpadDirection::kUp, {0.5F, 0.0F, 0.0F}},
      {DpadDirection::kDown, {-0.5F, 0.0F, 0.0F}},
      {DpadDirection::kLeft, {0.0F, 0.5F, 0.0F}},
      {DpadDirection::kRight, {0.0F, -0.5F, 0.0F}},
  };

  for (const auto& item : cases) {
    SingleStepVelocityController controller({});
    controller.reset({}, 0.40F);
    controller.update(frame(levels_for(item.direction), 0.48F));
    expect_state(controller, SingleStepState::kArmed);
    expect_true(controller.latched_direction() == item.direction,
                "unexpected latched direction");

    controller.update(frame(levels_for(item.direction), 0.51F));
    expect_state(controller, SingleStepState::kActive);
    expect_command(controller, item.expected);

    controller.update(frame({}, 0.75F));
    expect_state(controller, SingleStepState::kActive);
    expect_command(controller, item.expected);
  }
}

void test_conflict_clearing_while_one_direction_is_held_does_not_arm() {
  SingleStepVelocityController controller({});
  controller.reset({}, 0.40F);
  controller.update(frame({true, false, false, true}, 0.42F));
  expect_state(controller, SingleStepState::kIdle);
  expect_command(controller, {0.3F, 0.0F, 0.0F});

  controller.update(frame({true, false, false, false}, 0.44F));
  expect_state(controller, SingleStepState::kIdle);
  controller.update(frame({}, 0.46F));
  controller.update(frame({true, false, false, false}, 0.48F));
  expect_state(controller, SingleStepState::kArmed);
}

void test_active_direction_is_latched_and_waits_for_all_dpad_release() {
  SingleStepVelocityController controller({});
  controller.reset({}, 0.40F);
  controller.update(frame({false, false, true, false}, 0.48F));
  controller.update(frame({false, false, true, false}, 0.51F));
  expect_command(controller, {0.0F, 0.5F, 0.0F});

  controller.update(frame({false, false, false, true}, 0.75F));
  expect_command(controller, {0.0F, 0.5F, 0.0F});
  controller.update(frame({false, false, false, true}, 0.01F));
  expect_state(controller, SingleStepState::kWaitRelease);
  expect_command(controller, {0.0F, 0.0F, 0.0F});

  controller.update(frame({false, false, false, true}, 0.03F));
  expect_state(controller, SingleStepState::kWaitRelease);
  controller.update(frame({}, 0.05F));
  expect_state(controller, SingleStepState::kIdle);
}

void test_up_latches_height_until_the_step_returns_idle() {
  SingleStepVelocityController controller({});
  const auto up = levels_for(DpadDirection::kUp);
  controller.reset({}, 0.40F);
  controller.update(frame(up, 0.48F, 0.02F, {0.0F, 0.0F, 0.0F},
                          StepHeightLevel::kHigh));
  expect_true(controller.latched_height() == StepHeightLevel::kHigh,
              "Up must latch High");

  controller.update(frame(up, 0.51F, 0.02F, {0.0F, 0.0F, 0.0F},
                          StepHeightLevel::kLow));
  expect_state(controller, SingleStepState::kActive);
  expect_true(controller.latched_height() == StepHeightLevel::kHigh,
              "active stick changes must not rewrite height");

  controller.update(frame({}, 0.01F, 0.02F, {0.0F, 0.0F, 0.0F},
                          StepHeightLevel::kLow));
  expect_state(controller, SingleStepState::kWaitRelease);
  expect_true(controller.latched_height() == StepHeightLevel::kHigh,
              "height must remain latched through completion");
  controller.update(frame({}, 0.03F));
  expect_state(controller, SingleStepState::kIdle);
  expect_true(controller.latched_height() == StepHeightLevel::kMedium,
              "returning idle must clear latched height");
}

void test_nonforward_steps_force_medium_height() {
  const DpadDirection directions[] = {
      DpadDirection::kDown,
      DpadDirection::kLeft,
      DpadDirection::kRight,
  };
  for (const auto direction : directions) {
    SingleStepVelocityController controller({});
    controller.reset({}, 0.40F);
    controller.update(frame(levels_for(direction), 0.48F, 0.02F,
                            {0.0F, 0.0F, 0.0F}, StepHeightLevel::kHigh));
    expect_state(controller, SingleStepState::kArmed);
    expect_true(controller.latched_height() == StepHeightLevel::kMedium,
                "nonforward steps must not latch height correction");
  }
}

void test_active_half_and_progress_follow_the_latched_phase_window() {
  const auto up = levels_for(DpadDirection::kUp);
  SingleStepVelocityController second_half_controller({});
  second_half_controller.reset({}, 0.40F);
  expect_true(second_half_controller.active_progress() == 0.0F,
              "idle progress must be zero");
  second_half_controller.update(frame(up, 0.48F));
  expect_true(second_half_controller.active_progress() == 0.0F,
              "armed progress must be zero");
  second_half_controller.update(frame(up, 0.51F));
  expect_state(second_half_controller, SingleStepState::kActive);
  expect_true(second_half_controller.active_half() == GaitHalf::kSecond,
              "0.5 boundary must start the second half");
  expect_true(second_half_controller.active_progress() == 0.0F,
              "new active interval must start at zero progress");
  second_half_controller.update(frame(up, 0.76F));
  expect_true(
      std::fabs(second_half_controller.active_progress() - 0.5F) <= 1e-6F,
      "quarter phase advance must be half of the active interval");

  SingleStepVelocityController first_half_controller({});
  first_half_controller.reset({}, 0.90F);
  first_half_controller.update(frame(up, 0.99F));
  first_half_controller.update(frame(up, 0.01F));
  expect_state(first_half_controller, SingleStepState::kActive);
  expect_true(first_half_controller.active_half() == GaitHalf::kFirst,
              "wrapped boundary must start the first half");
}

void test_abort_clears_latched_height() {
  SingleStepVelocityController controller({});
  const auto up = levels_for(DpadDirection::kUp);
  controller.reset({}, 0.40F);
  controller.update(frame(up, 0.48F, 0.02F, {0.0F, 0.0F, 0.0F},
                          StepHeightLevel::kLow));
  expect_true(controller.latched_height() == StepHeightLevel::kLow,
              "precondition: Low must be latched");
  controller.abort(SingleStepAbortReason::kExternalStop);
  expect_true(controller.latched_height() == StepHeightLevel::kMedium,
              "abort must clear latched height");
  expect_true(controller.active_progress() == 0.0F,
              "abort must clear active progress");
}

void test_armed_timeout_and_same_tick_boundary_precedence() {
  SingleStepVelocityController controller({});
  const auto up = levels_for(DpadDirection::kUp);
  controller.reset({}, 0.2F);
  controller.update(frame(up, 0.21F));
  controller.update(frame(up, 0.22F, 0.35F));
  expect_state(controller, SingleStepState::kArmed);
  controller.update(frame(up, 0.23F, 0.35F));
  expect_state(controller, SingleStepState::kAborted);
  expect_abort_reason(controller, SingleStepAbortReason::kArmedTimeout);
  expect_command(controller, {0.0F, 0.0F, 0.0F});

  SingleStepConfig config;
  config.armed_timeout_s = 0.02F;
  SingleStepVelocityController boundary_controller(config);
  boundary_controller.reset({}, 0.4F);
  boundary_controller.update(frame(up, 0.45F));
  boundary_controller.update(frame(up, 0.51F));
  expect_state(boundary_controller, SingleStepState::kActive);
  expect_abort_reason(boundary_controller, SingleStepAbortReason::kNone);
}

void test_active_timeout_and_same_tick_completion_precedence() {
  SingleStepVelocityController controller({});
  const auto up = levels_for(DpadDirection::kUp);
  controller.reset({}, 0.4F);
  controller.update(frame(up, 0.48F));
  controller.update(frame(up, 0.51F));
  controller.update(frame(up, 0.60F, 0.35F));
  expect_state(controller, SingleStepState::kActive);
  controller.update(frame(up, 0.70F, 0.35F));
  expect_state(controller, SingleStepState::kAborted);
  expect_abort_reason(controller, SingleStepAbortReason::kActiveTimeout);
  expect_command(controller, {0.0F, 0.0F, 0.0F});

  SingleStepConfig config;
  config.active_timeout_s = 0.02F;
  SingleStepVelocityController completion_controller(config);
  completion_controller.reset({}, 0.4F);
  completion_controller.update(frame(up, 0.48F));
  completion_controller.update(frame(up, 0.51F));
  completion_controller.update(frame(up, 0.01F));
  expect_state(completion_controller, SingleStepState::kWaitRelease);
  expect_abort_reason(completion_controller, SingleStepAbortReason::kNone);
}

void test_invalid_inputs_abort_immediately() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const SingleStepInput invalid_inputs[] = {
      frame({}, 0.2F, 0.0F),
      frame({}, 0.2F, -0.01F),
      frame({}, 0.2F, infinity),
      frame({}, nan, 0.02F),
      frame({}, infinity, 0.02F),
      frame({}, 0.2F, 0.02F, {0.0F, 0.0F, 0.0F},
            static_cast<StepHeightLevel>(99)),
  };
  for (const auto& input : invalid_inputs) {
    SingleStepVelocityController controller({});
    controller.reset({}, 0.2F);
    controller.update(input);
    expect_state(controller, SingleStepState::kAborted);
    expect_abort_reason(controller, SingleStepAbortReason::kInvalidInput);
    expect_command(controller, {0.0F, 0.0F, 0.0F});
  }
}

void test_external_stop_and_aborted_release_safety() {
  const auto up = levels_for(DpadDirection::kUp);
  SingleStepVelocityController idle_controller({});
  idle_controller.reset({}, 0.2F);
  idle_controller.abort(SingleStepAbortReason::kExternalStop);

  SingleStepVelocityController armed_controller({});
  armed_controller.reset({}, 0.2F);
  armed_controller.update(frame(up, 0.25F));
  armed_controller.abort(SingleStepAbortReason::kExternalStop);

  SingleStepVelocityController active_controller({});
  active_controller.reset({}, 0.4F);
  active_controller.update(frame(up, 0.48F));
  active_controller.update(frame(up, 0.51F));
  active_controller.abort(SingleStepAbortReason::kExternalStop);

  for (const auto* controller :
       {&idle_controller, &armed_controller, &active_controller}) {
    expect_state(*controller, SingleStepState::kAborted);
    expect_abort_reason(*controller, SingleStepAbortReason::kExternalStop);
    expect_command(*controller, {0.0F, 0.0F, 0.0F});
  }

  armed_controller.update(frame(up, 0.30F));
  expect_state(armed_controller, SingleStepState::kAborted);
  expect_command(armed_controller, {0.0F, 0.0F, 0.0F});
  armed_controller.update(frame({}, 0.35F));
  expect_state(armed_controller, SingleStepState::kIdle);
  armed_controller.update(frame(up, 0.40F));
  expect_state(armed_controller, SingleStepState::kArmed);
}

void expect_invalid_config(SingleStepConfig config) {
  bool threw = false;
  try {
    SingleStepVelocityController controller(config);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect_true(threw, "invalid constructor configuration must throw");
}

void test_invalid_constructor_configuration_throws() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  SingleStepConfig config;
  config.up_command[0] = infinity;
  expect_invalid_config(config);
  config = {};
  config.down_command[0] = nan;
  expect_invalid_config(config);
  config = {};
  config.left_command[1] = infinity;
  expect_invalid_config(config);
  config = {};
  config.right_command[1] = nan;
  expect_invalid_config(config);
  config = {};
  config.phase_advance = 0.0F;
  expect_invalid_config(config);
  config = {};
  config.phase_advance = 1.01F;
  expect_invalid_config(config);
  config = {};
  config.armed_timeout_s = 0.0F;
  expect_invalid_config(config);
  config = {};
  config.active_timeout_s = nan;
  expect_invalid_config(config);
}

}  // namespace

int main() {
  test_idle_passes_through_without_mutating_reads();
  test_reset_while_any_direction_is_held_requires_full_release();
  test_press_arms_then_waits_for_next_boundary();
  test_armed_activates_at_a_wrapped_half_cycle_boundary();
  test_press_on_boundary_waits_for_following_boundary();
  test_active_step_runs_to_completion_after_direction_is_released();
  test_held_direction_cannot_retrigger_after_active_completion();
  test_all_dpad_directions_latch_the_expected_command();
  test_conflict_clearing_while_one_direction_is_held_does_not_arm();
  test_active_direction_is_latched_and_waits_for_all_dpad_release();
  test_up_latches_height_until_the_step_returns_idle();
  test_nonforward_steps_force_medium_height();
  test_active_half_and_progress_follow_the_latched_phase_window();
  test_abort_clears_latched_height();
  test_armed_timeout_and_same_tick_boundary_precedence();
  test_active_timeout_and_same_tick_completion_precedence();
  test_invalid_inputs_abort_immediately();
  test_external_stop_and_aborted_release_safety();
  test_invalid_constructor_configuration_throws();
  std::cout << "all controller tests passed\n";
  return 0;
}
