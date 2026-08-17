// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include "isaaclab/utils/utils.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include "isaaclab/devices/gamepad/single_step_velocity_controller.h"
#include "isaaclab/devices/gamepad/gamepad_motion_mapping.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
        configure_external_velocity_control();
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        reset_external_velocity_control();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        update_external_velocity_command();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);
    }

    using VelocityCommand = external_control::VelocityCommand;

    [[nodiscard]] VelocityCommand joystick_velocity_command() const
    {
        const auto ranges = cfg["commands"]["base_velocity"]["ranges"];
        auto* joystick = robot->data.joystick;
        if (joystick == nullptr) {
            return {0.0F, 0.0F, 0.0F};
        }
        VelocityCommand command{
            std::clamp(joystick->ly(), ranges["lin_vel_x"][0].as<float>(),
                       ranges["lin_vel_x"][1].as<float>()),
            std::clamp(-joystick->lx(), ranges["lin_vel_y"][0].as<float>(),
                       ranges["lin_vel_y"][1].as<float>()),
            std::clamp(-joystick->rx(), ranges["ang_vel_z"][0].as<float>(),
                       ranges["ang_vel_z"][1].as<float>()),
        };
        return apply_shoulder_yaw(command);
    }

    [[nodiscard]] VelocityCommand velocity_command() const
    {
        if (external_velocity_command_snapshot_.has_value()) {
            return *external_velocity_command_snapshot_;
        }
        return joystick_velocity_command();
    }

    [[nodiscard]] float swing_height_command_m() const noexcept
    {
        return step_height_selector_.has_value()
                   ? step_height_selector_->value_m()
                   : 0.10F;
    }

    [[nodiscard]] float step_length_command_m() const noexcept
    {
        return step_length_selector_.has_value()
                   ? step_length_selector_->value_m()
                   : 0.30F;
    }

    [[nodiscard]] std::vector<float> apply_external_joint_position_command(
        const std::vector<float>& policy_targets) const
    {
        if (!single_step_velocity_controller_.has_value() ||
            !swing_height_controller_.has_value()) {
            return policy_targets;
        }
        const external_control::SwingHeightInput input{
            single_step_velocity_controller_->state() ==
                external_control::SingleStepState::kActive,
            single_step_velocity_controller_->latched_direction() ==
                external_control::DpadDirection::kUp,
            single_step_velocity_controller_->latched_height(),
            single_step_velocity_controller_->active_half(),
            single_step_velocity_controller_->active_progress(),
        };
        return swing_height_controller_->apply(policy_targets, input);
    }

    void stop_external_velocity_command() noexcept
    {
        if (single_step_velocity_controller_.has_value()) {
            const auto previous_state = single_step_velocity_controller_->state();
            single_step_velocity_controller_->abort(
                external_control::SingleStepAbortReason::kExternalStop);
            external_velocity_command_snapshot_ = VelocityCommand{0.0F, 0.0F, 0.0F};
            log_external_velocity_transition(previous_state);
        }
    }

private:
    [[nodiscard]] external_control::DpadLevels dpad_levels() const noexcept
    {
        auto* joystick = robot->data.joystick;
        if (joystick == nullptr) {
            return {};
        }
        return {
            static_cast<bool>(joystick->up()),
            static_cast<bool>(joystick->down()),
            static_cast<bool>(joystick->left()),
            static_cast<bool>(joystick->right()),
        };
    }

    [[nodiscard]] float logical_right_y() const noexcept
    {
        auto* joystick = robot->data.joystick;
        return joystick == nullptr ? 0.0F : joystick->ry();
    }

    [[nodiscard]] float logical_right_x() const noexcept
    {
        auto* joystick = robot->data.joystick;
        return joystick == nullptr ? 0.0F : joystick->rx();
    }

    [[nodiscard]] VelocityCommand apply_shoulder_yaw(
        VelocityCommand command) const
    {
        if (!motion_mapping_enabled_) {
            return command;
        }
        auto* joystick = robot->data.joystick;
        const bool yaw_enabled =
            !single_step_velocity_controller_.has_value() ||
            single_step_velocity_controller_->state() !=
                external_control::SingleStepState::kAborted;
        return external_control::with_shoulder_yaw(
            command,
            joystick != nullptr && static_cast<bool>(joystick->LB()),
            joystick != nullptr && static_cast<bool>(joystick->RB()),
            joystick != nullptr && joystick->LT.pressed,
            shoulder_yaw_speed_, yaw_boost_multiplier_, yaw_enabled);
    }

    void configure_external_velocity_control()
    {
        const auto external_control_cfg = cfg["external_control"];
        const auto single_step = external_control_cfg["single_step"];
        const auto motion_mapping = external_control_cfg["motion_mapping"];
        const auto swing_height = external_control_cfg["swing_height"];
        const bool single_step_enabled =
            single_step && single_step["enabled"].as<bool>();
        const bool swing_height_enabled =
            swing_height && swing_height["enabled"].as<bool>();
        motion_mapping_enabled_ =
            motion_mapping && motion_mapping["enabled"].as<bool>();
        if (motion_mapping_enabled_) {
            const auto thresholds = motion_mapping["right_x_thresholds"];
            const auto lengths = motion_mapping["step_length_m"];
            try {
                step_length_selector_.emplace(
                    external_control::StepLengthSelectorConfig{
                        thresholds["short_enter"].as<float>(),
                        thresholds["short_exit"].as<float>(),
                        thresholds["long_exit"].as<float>(),
                        thresholds["long_enter"].as<float>(),
                        lengths["short"].as<float>(),
                        lengths["medium"].as<float>(),
                        lengths["long"].as<float>(),
                    });
                shoulder_yaw_speed_ =
                    motion_mapping["shoulder_yaw_speed"].as<float>();
                yaw_boost_multiplier_ =
                    motion_mapping["yaw_boost_multiplier"].as<float>();
                const auto yaw_range =
                    cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"];
                const float boosted_yaw_speed =
                    shoulder_yaw_speed_ * yaw_boost_multiplier_;
                if (!std::isfinite(shoulder_yaw_speed_) ||
                    shoulder_yaw_speed_ < 0.0F ||
                    !std::isfinite(yaw_boost_multiplier_) ||
                    yaw_boost_multiplier_ < 1.0F ||
                    !std::isfinite(boosted_yaw_speed) ||
                    boosted_yaw_speed > yaw_range[1].as<float>() ||
                    -boosted_yaw_speed < yaw_range[0].as<float>()) {
                    throw std::runtime_error(
                        "shoulder yaw base or boosted speed is invalid");
                }
            } catch (const std::exception& exception) {
                throw std::runtime_error(
                    std::string("invalid external_control.motion_mapping configuration: ") +
                    exception.what());
            }
        }
        if (swing_height_enabled && !single_step_enabled) {
            throw std::runtime_error(
                "external_control.swing_height requires single_step");
        }
        if (!single_step_enabled) {
            return;
        }
        if (!cfg["observations"]["gait_phase"]) {
            throw std::runtime_error("external_control.single_step requires gait_phase observation");
        }
        try {
            const auto parse_command = [&](const char* direction) {
                const auto values = single_step["direction_commands"][direction]
                                        .as<std::vector<float>>();
                if (values.size() != 3) {
                    throw std::runtime_error(
                        std::string(
                            "external_control.single_step.direction_commands.") +
                        direction + " must contain exactly 3 values");
                }
                return VelocityCommand{values[0], values[1], values[2]};
            };
            external_control::SingleStepConfig config{
                parse_command("up"),
                parse_command("down"),
                parse_command("left"),
                parse_command("right"),
                single_step["phase_advance"].as<float>(),
                single_step["armed_timeout_s"].as<float>(),
                single_step["active_timeout_s"].as<float>(),
            };
            const auto ranges = cfg["commands"]["base_velocity"]["ranges"];
            const char* range_names[] = {
                "lin_vel_x",
                "lin_vel_y",
                "ang_vel_z",
            };
            const auto validate_command =
                [&](const char* direction, const VelocityCommand& command) {
                    for (std::size_t index = 0; index < command.size(); ++index) {
                        const auto range = ranges[range_names[index]];
                        if (!std::isfinite(command[index]) ||
                            command[index] < range[0].as<float>() ||
                            command[index] > range[1].as<float>()) {
                            throw std::runtime_error(
                                std::string(
                                    "external_control.single_step.direction_commands.") +
                                direction + "[" + std::to_string(index) +
                                "] is outside " + range_names[index] + " range");
                        }
                    }
                };
            validate_command("up", config.up_command);
            validate_command("down", config.down_command);
            validate_command("left", config.left_command);
            validate_command("right", config.right_command);
            single_step_velocity_controller_.emplace(config);
            external_velocity_command_snapshot_ = VelocityCommand{0.0F, 0.0F, 0.0F};
        } catch (const std::exception& exception) {
            throw std::runtime_error(std::string("invalid external_control.single_step configuration: ") + exception.what());
        }

        if (!swing_height_enabled) {
            return;
        }
        try {
            const auto thresholds = swing_height["right_y_thresholds"];
            external_control::StepHeightSelectorConfig selector_config{
                thresholds["low_enter"].as<float>(),
                thresholds["low_exit"].as<float>(),
                thresholds["high_exit"].as<float>(),
                thresholds["high_enter"].as<float>(),
            };
            const auto height_commands = swing_height["height_command_m"];
            if (height_commands) {
                selector_config.low_m =
                    height_commands["low"].as<float>();
                selector_config.medium_m =
                    height_commands["medium"].as<float>();
                selector_config.high_m =
                    height_commands["high"].as<float>();
            }
            step_height_selector_.emplace(selector_config);

            const bool joint_overlay_enabled =
                swing_height["joint_overlay_enabled"].as<bool>(true);
            if (!joint_overlay_enabled) {
                return;
            }

            const auto parse_float_triplet = [](
                const YAML::Node& node, const char* field) {
                const auto values = node.as<std::vector<float>>();
                if (values.size() != 3) {
                    throw std::runtime_error(
                        std::string(field) + " must contain exactly 3 values");
                }
                return std::array<float, 3>{values[0], values[1], values[2]};
            };
            const auto parse_joint_triplet = [](
                const YAML::Node& node, const char* field) {
                const auto values = node.as<std::vector<std::size_t>>();
                if (values.size() != 3) {
                    throw std::runtime_error(
                        std::string(field) + " must contain exactly 3 values");
                }
                return external_control::JointTriplet{
                    values[0], values[1], values[2]};
            };
            const auto parse_joint_limit = [](
                const YAML::Node& node, const char* field) {
                const auto values = node.as<std::vector<float>>();
                if (values.size() != 2) {
                    throw std::runtime_error(
                        std::string(field) + " must contain exactly 2 values");
                }
                return external_control::JointLimit{values[0], values[1]};
            };

            const auto first_half_name =
                swing_height["first_half_swing_leg"].as<std::string>();
            external_control::SwingLeg first_half_swing_leg;
            if (first_half_name == "left") {
                first_half_swing_leg = external_control::SwingLeg::kLeft;
            } else if (first_half_name == "right") {
                first_half_swing_leg = external_control::SwingLeg::kRight;
            } else {
                throw std::runtime_error(
                    "first_half_swing_leg must be left or right");
            }

            external_control::SwingHeightConfig config;
            config.enabled = true;
            config.action_dim =
                static_cast<std::size_t>(action_manager->total_action_dim());
            config.first_half_swing_leg = first_half_swing_leg;
            config.left = parse_joint_triplet(
                swing_height["joint_indices"]["left"], "joint_indices.left");
            config.right = parse_joint_triplet(
                swing_height["joint_indices"]["right"], "joint_indices.right");
            config.low = parse_float_triplet(
                swing_height["profiles"]["low"], "profiles.low");
            config.medium = parse_float_triplet(
                swing_height["profiles"]["medium"], "profiles.medium");
            config.high = parse_float_triplet(
                swing_height["profiles"]["high"], "profiles.high");
            config.max_abs_offset = parse_float_triplet(
                swing_height["max_abs_offset"], "max_abs_offset");
            config.max_delta_per_tick = parse_float_triplet(
                swing_height["max_delta_per_tick"], "max_delta_per_tick");
            config.joint_limits = std::array<external_control::JointLimit, 3>{
                parse_joint_limit(swing_height["joint_limits"]["hip_pitch"],
                                  "joint_limits.hip_pitch"),
                parse_joint_limit(swing_height["joint_limits"]["knee"],
                                  "joint_limits.knee"),
                parse_joint_limit(swing_height["joint_limits"]["ankle_pitch"],
                                  "joint_limits.ankle_pitch"),
            };
            config.step_dt = step_dt;
            config.gait_period = cfg["observations"]["gait_phase"]
                                     ["params"]["period"].as<float>();
            config.active_phase_advance =
                single_step["phase_advance"].as<float>();

            swing_height_controller_.emplace(config);
        } catch (const std::exception& exception) {
            throw std::runtime_error(
                std::string(
                    "invalid external_control.swing_height configuration: ") +
                exception.what());
        }
    }
    void reset_external_velocity_control()
    {
        if (step_length_selector_.has_value()) {
            step_length_selector_->reset();
        }
        if (step_height_selector_.has_value()) {
            step_height_selector_->reset();
        }
        if (!single_step_velocity_controller_.has_value()) {
            return;
        }
        single_step_velocity_controller_->reset(dpad_levels(), global_phase);
        external_velocity_command_snapshot_ = single_step_velocity_controller_->command();
    }

    void update_external_velocity_command()
    {
        if (!single_step_velocity_controller_.has_value()) {
            return;
        }
        if (step_height_selector_.has_value()) {
            step_height_selector_->update(logical_right_y());
        }
        if (step_length_selector_.has_value()) {
            const auto previous_level = step_length_selector_->level();
            step_length_selector_->update(logical_right_x());
            if (step_length_selector_->level() != previous_level) {
                try {
                    std::cout << "[step_length] level="
                              << external_control::to_string(
                                     step_length_selector_->level())
                              << " value_m=" << step_length_selector_->value_m()
                              << std::endl;
                } catch (...) {
                }
            }
        }
        external_control::SingleStepInput input;
        input.dpad = dpad_levels();
        input.gait_phase = global_phase;
        input.dt = step_dt;
        input.passthrough_command = joystick_velocity_command();
        input.requested_height = step_height_selector_.has_value()
                                     ? step_height_selector_->level()
                                     : external_control::StepHeightLevel::kMedium;
        const auto previous_state = single_step_velocity_controller_->state();
        single_step_velocity_controller_->update(input);
        external_velocity_command_snapshot_ =
            apply_shoulder_yaw(single_step_velocity_controller_->command());
        log_external_velocity_transition(previous_state);
    }

    void log_external_velocity_transition(external_control::SingleStepState previous_state) noexcept
    {
        const auto current_state = single_step_velocity_controller_->state();
        if (current_state == previous_state) {
            return;
        }
        try {
            std::cout << "[single_step] " << external_control::to_string(previous_state)
                      << " -> " << external_control::to_string(current_state);
            if (current_state == external_control::SingleStepState::kArmed) {
                std::cout << " (direction="
                          << external_control::to_string(
                                 single_step_velocity_controller_->latched_direction())
                          << ", height="
                          << external_control::to_string(
                                 single_step_velocity_controller_->latched_height())
                          << ")";
            }
            if (current_state == external_control::SingleStepState::kActive &&
                swing_height_controller_.has_value()) {
                std::cout << " (swing_leg="
                          << external_control::to_string(
                                 swing_height_controller_->swing_leg_for(
                                     single_step_velocity_controller_->active_half()))
                          << ")";
            }
            if (current_state == external_control::SingleStepState::kAborted) {
                std::cout << " (" << external_control::to_string(single_step_velocity_controller_->abort_reason()) << ")";
            }
            std::cout << std::endl;
        } catch (...) {
            // Logging must not change the noexcept stop path.
        }
    }

public:
    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
private:
    std::optional<external_control::SingleStepVelocityController> single_step_velocity_controller_;
    std::optional<VelocityCommand> external_velocity_command_snapshot_;
    std::optional<external_control::StepHeightSelector> step_height_selector_;
    std::optional<external_control::SwingHeightController> swing_height_controller_;
    std::optional<external_control::StepLengthSelector> step_length_selector_;
    bool motion_mapping_enabled_{false};
    float shoulder_yaw_speed_{0.0F};
    float yaw_boost_multiplier_{1.0F};
};

};
