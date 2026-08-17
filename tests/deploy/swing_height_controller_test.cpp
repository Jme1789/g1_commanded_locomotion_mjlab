#include "isaaclab/devices/gamepad/swing_height_controller.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using isaaclab::external_control::GaitHalf;
using isaaclab::external_control::StepHeightLevel;
using isaaclab::external_control::StepHeightSelector;
using isaaclab::external_control::StepHeightSelectorConfig;
using isaaclab::external_control::SwingHeightConfig;
using isaaclab::external_control::SwingHeightController;
using isaaclab::external_control::SwingHeightInput;
using isaaclab::external_control::SwingLeg;

void expect_true(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void expect_near(float actual, float expected, const char* message) {
  expect_true(std::fabs(actual - expected) <= 1e-6F, message);
}

void expect_equal(const std::vector<float>& actual,
                  const std::vector<float>& expected,
                  const char* message) {
  expect_true(actual.size() == expected.size(), message);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::fabs(actual[index] - expected[index]) > 1e-6F) {
      std::cerr << message << " at index " << index << '\n';
      std::exit(1);
    }
  }
}

SwingHeightConfig enabled_config() {
  SwingHeightConfig config;
  config.enabled = true;
  return config;
}

void expect_invalid_selector(StepHeightSelectorConfig config) {
  bool threw = false;
  try {
    StepHeightSelector selector(config);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect_true(threw, "invalid selector configuration must throw");
}

void expect_invalid_overlay(SwingHeightConfig config) {
  bool threw = false;
  try {
    SwingHeightController controller(config);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect_true(threw, "invalid overlay configuration must throw");
}

void test_selector_uses_hysteresis_and_resets_invalid_input() {
  StepHeightSelector selector({-0.60F, -0.40F, 0.40F, 0.60F,
                               0.05F, 0.10F, 0.15F});
  expect_true(selector.level() == StepHeightLevel::kMedium, "reset medium");
  expect_near(selector.value_m(), 0.10F, "reset medium value");

  selector.update(0.60F);
  expect_true(selector.level() == StepHeightLevel::kHigh, "enter high");
  expect_near(selector.value_m(), 0.15F, "high value");
  selector.update(0.50F);
  expect_true(selector.level() == StepHeightLevel::kHigh, "hold high");
  selector.update(0.40F);
  expect_true(selector.level() == StepHeightLevel::kMedium, "exit high");

  selector.update(-0.60F);
  expect_true(selector.level() == StepHeightLevel::kLow, "enter low");
  expect_near(selector.value_m(), 0.05F, "low value");
  selector.update(-0.50F);
  expect_true(selector.level() == StepHeightLevel::kLow, "hold low");
  selector.update(-0.40F);
  expect_true(selector.level() == StepHeightLevel::kMedium, "exit low");

  selector.update(0.80F);
  selector.reset();
  expect_true(selector.level() == StepHeightLevel::kMedium,
              "explicit reset must select medium");
  selector.update(std::numeric_limits<float>::quiet_NaN());
  expect_true(selector.level() == StepHeightLevel::kMedium,
              "non-finite input must select medium");
  expect_near(selector.value_m(), 0.10F,
              "non-finite input must restore medium value");
}

void test_inactive_medium_and_boundary_inputs_are_exact_noops() {
  const std::vector<float> baseline(29, 0.1F);
  const SwingHeightController controller(enabled_config());
  const SwingHeightInput inputs[] = {
      {false, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 0.5F},
      {true, false, StepHeightLevel::kHigh, GaitHalf::kFirst, 0.5F},
      {true, true, StepHeightLevel::kMedium, GaitHalf::kFirst, 0.5F},
      {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 0.0F},
      {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 1.0F},
      {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst,
       std::numeric_limits<float>::quiet_NaN()},
      {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, -0.1F},
      {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 1.1F},
  };
  for (const auto& input : inputs) {
    expect_equal(controller.apply(baseline, input), baseline,
                 "inactive or invalid input must be a no-op");
  }

  SwingHeightConfig disabled = enabled_config();
  disabled.enabled = false;
  const SwingHeightController disabled_controller(disabled);
  expect_equal(
      disabled_controller.apply(
          baseline,
          {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 0.5F}),
      baseline, "disabled controller must be an exact no-op");
}

void test_profiles_modify_only_the_selected_swing_leg() {
  const std::vector<float> baseline(29, 0.1F);
  const SwingHeightController controller(enabled_config());

  auto expected_high = baseline;
  expected_high[0] = 0.02F;
  expected_high[3] = 0.26F;
  expected_high[4] = 0.02F;
  expect_equal(
      controller.apply(
          baseline,
          {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 0.5F}),
      expected_high, "High must modify only the first-half left leg");

  auto expected_low = baseline;
  expected_low[6] = 0.125F;
  expected_low[9] = 0.05F;
  expected_low[10] = 0.125F;
  expect_equal(
      controller.apply(
          baseline,
          {true, true, StepHeightLevel::kLow, GaitHalf::kSecond, 0.5F}),
      expected_low, "Low must modify only the second-half right leg");

  SwingHeightConfig reversed = enabled_config();
  reversed.first_half_swing_leg = SwingLeg::kRight;
  const SwingHeightController reversed_controller(reversed);
  auto expected_reversed = baseline;
  expected_reversed[6] = 0.02F;
  expected_reversed[9] = 0.26F;
  expected_reversed[10] = 0.02F;
  expect_equal(
      reversed_controller.apply(
          baseline,
          {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 0.5F}),
      expected_reversed, "configured half-to-leg mapping must be honored");
}

void test_modified_targets_are_clamped_to_physical_limits() {
  std::vector<float> baseline(29, 0.1F);
  baseline[3] = 2.85F;
  const SwingHeightController controller(enabled_config());
  const auto actual = controller.apply(
      baseline,
      {true, true, StepHeightLevel::kHigh, GaitHalf::kFirst, 0.5F});
  expect_near(actual[3], 2.8798F, "left knee must clamp at its upper limit");
}

void test_invalid_configuration_fails_before_runtime() {
  expect_invalid_selector(
      {-0.40F, -0.60F, 0.40F, 0.60F, 0.05F, 0.10F, 0.15F});
  expect_invalid_selector(
      {-0.60F, -0.40F, 0.70F, 0.60F, 0.05F, 0.10F, 0.15F});
  expect_invalid_selector(
      {-0.60F, -0.40F, 0.40F, 0.60F, 0.10F, 0.10F, 0.15F});
  expect_invalid_selector(
      {-0.60F, -0.40F, 0.40F, 0.60F, 0.05F, 0.10F,
       std::numeric_limits<float>::infinity()});

  SwingHeightConfig config = enabled_config();
  config.medium[0] = 0.001F;
  expect_invalid_overlay(config);

  config = enabled_config();
  config.left.knee = config.left.hip_pitch;
  expect_invalid_overlay(config);

  config = enabled_config();
  config.right.ankle_pitch = 29;
  expect_invalid_overlay(config);

  config = enabled_config();
  config.high[0] = std::numeric_limits<float>::quiet_NaN();
  expect_invalid_overlay(config);

  config = enabled_config();
  config.high[1] = 0.25F;
  expect_invalid_overlay(config);

  config = enabled_config();
  config.joint_limits[1] = {1.0F, 0.0F};
  expect_invalid_overlay(config);

  config = enabled_config();
  config.high[1] = 0.20F;
  expect_invalid_overlay(config);

  config = enabled_config();
  config.action_dim = 0;
  expect_invalid_overlay(config);
}

}  // namespace

int main() {
  test_selector_uses_hysteresis_and_resets_invalid_input();
  test_inactive_medium_and_boundary_inputs_are_exact_noops();
  test_profiles_modify_only_the_selected_swing_leg();
  test_modified_targets_are_clamped_to_physical_limits();
  test_invalid_configuration_fails_before_runtime();
  std::cout << "all swing height tests passed\n";
  return 0;
}
