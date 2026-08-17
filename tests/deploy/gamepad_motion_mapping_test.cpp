#include "isaaclab/devices/gamepad/gamepad_motion_mapping.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

using isaaclab::external_control::StepLengthLevel;
using isaaclab::external_control::StepLengthSelector;
using isaaclab::external_control::StepLengthSelectorConfig;
using isaaclab::external_control::to_string;
using isaaclab::external_control::with_shoulder_yaw;

void expect_true(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void expect_near(float actual, float expected, const char* message) {
  expect_true(std::fabs(actual - expected) <= 1e-6F, message);
}

void test_right_x_selects_step_length_with_hysteresis() {
  StepLengthSelector selector({-0.60F, -0.40F, 0.40F, 0.60F,
                               0.20F, 0.30F, 0.40F});
  expect_true(selector.level() == StepLengthLevel::kMedium, "start medium");
  expect_near(selector.value_m(), 0.30F, "medium value");
  expect_true(std::string(to_string(selector.level())) == "medium",
              "medium level is observable");

  selector.update(0.60F);
  expect_true(selector.level() == StepLengthLevel::kLong, "enter long");
  expect_true(std::string(to_string(selector.level())) == "long",
              "long level is observable");
  selector.update(0.50F);
  expect_true(selector.level() == StepLengthLevel::kLong, "hold long");
  selector.update(0.40F);
  expect_true(selector.level() == StepLengthLevel::kMedium, "exit long");

  selector.update(-0.60F);
  expect_true(selector.level() == StepLengthLevel::kShort, "enter short");
  expect_true(std::string(to_string(selector.level())) == "short",
              "short level is observable");
  selector.update(-0.50F);
  expect_true(selector.level() == StepLengthLevel::kShort, "hold short");
  selector.update(-0.40F);
  expect_true(selector.level() == StepLengthLevel::kMedium, "exit short");

  selector.update(0.80F);
  selector.update(std::numeric_limits<float>::quiet_NaN());
  expect_true(selector.level() == StepLengthLevel::kMedium,
              "non-finite input resets medium");
}

void test_shoulder_buttons_replace_only_yaw_and_lt_boosts_speed() {
  const std::array<float, 3> step{0.5F, -0.25F, 0.9F};

  const auto left =
      with_shoulder_yaw(step, true, false, false, 1.0F, 1.5F);
  expect_near(left[0], 0.5F, "LB preserves vx");
  expect_near(left[1], -0.25F, "LB preserves vy");
  expect_near(left[2], 1.0F, "LB turns left at base speed");
  expect_near(
      with_shoulder_yaw(step, true, false, true, 1.0F, 1.5F)[2],
      1.5F, "LT boosts LB by the configured multiplier");
  expect_near(
      with_shoulder_yaw(step, true, false, true, 1.0F, 1.5F)[2],
      1.5F, "held LT plus LB remains boosted on repeated frames");
  expect_near(
      with_shoulder_yaw(step, true, false, false, 1.0F, 1.5F)[2],
      1.0F, "releasing LT restores base yaw immediately");

  expect_near(
      with_shoulder_yaw(step, false, true, false, 1.0F, 1.5F)[2],
      -1.0F, "RB turns right at base speed");
  expect_near(
      with_shoulder_yaw(step, false, true, true, 1.0F, 1.5F)[2],
      -1.5F, "LT boosts RB by the configured multiplier");
  expect_near(
      with_shoulder_yaw(step, false, false, true, 1.0F, 1.5F)[2],
      0.0F, "LT alone does not create yaw");
  expect_near(
      with_shoulder_yaw(step, true, true, true, 1.0F, 1.5F)[2],
      0.0F, "LB plus RB remains conflict-safe while boosted");

  const auto stopped = with_shoulder_yaw(
      {0.0F, 0.0F, 0.0F}, true, false, true, 1.0F, 1.5F, false);
  expect_near(stopped[2], 0.0F,
              "an aborted command remains zero while LT and LB are held");

  bool rejected_small_multiplier = false;
  try {
    (void)with_shoulder_yaw(step, true, false, true, 1.0F, 0.9F);
  } catch (const std::invalid_argument&) {
    rejected_small_multiplier = true;
  }
  expect_true(rejected_small_multiplier,
              "boost multiplier below one must be rejected");

  bool rejected_non_finite_multiplier = false;
  try {
    (void)with_shoulder_yaw(
        step, true, false, true, 1.0F,
        std::numeric_limits<float>::quiet_NaN());
  } catch (const std::invalid_argument&) {
    rejected_non_finite_multiplier = true;
  }
  expect_true(rejected_non_finite_multiplier,
              "non-finite boost multiplier must be rejected");
}

}  // namespace

int main() {
  test_right_x_selects_step_length_with_hysteresis();
  test_shoulder_buttons_replace_only_yaw_and_lt_boosts_speed();
  std::cout << "all gamepad motion mapping tests passed\n";
  return 0;
}
