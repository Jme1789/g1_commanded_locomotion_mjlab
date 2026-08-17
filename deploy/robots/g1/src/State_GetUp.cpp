#include "State_GetUp.h"

#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "unitree_articulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t kG1ActionCount = 29;

getup_control::GetUpRecoveryConfig parse_recovery_config(
    const YAML::Node& state_config)
{
    const auto recovery = state_config["recovery"];
    return {
        recovery["upright_tilt_max_rad"].as<float>(),
        recovery["angular_speed_max_rad_s"].as<float>(),
        recovery["joint_speed_max_rad_s"].as<float>(),
        recovery["stable_duration_s"].as<double>(),
        recovery["timeout_s"].as<double>(),
        recovery["max_consecutive_inference_failures"].as<int>(),
        recovery["max_sample_gap_s"].as<double>(),
    };
}

const char* failure_reason_name(
    getup_control::GetUpFailureReason reason) noexcept
{
    switch (reason) {
    case getup_control::GetUpFailureReason::kNone:
        return "none";
    case getup_control::GetUpFailureReason::kTimeout:
        return "timeout";
    case getup_control::GetUpFailureReason::kInference:
        return "inference";
    case getup_control::GetUpFailureReason::kInvalidInput:
        return "invalid_input";
    }
    return "unknown";
}

bool all_finite(const std::vector<float>& values) noexcept
{
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

}  // namespace

State_GetUp::State_GetUp(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string),
      policy_dir_(param::parser_policy_dir(
          param::config["FSM"][state_string]["policy_dir"].as<std::string>())),
      recovery_monitor_(parse_recovery_config(
          param::config["FSM"][state_string]))
{
    const auto state_config = param::config["FSM"][state_string];
    const auto success_state =
        state_config["success_state"].as<std::string>();
    const auto failure_state =
        state_config["failure_state"].as<std::string>();
    success_state_id_ = FSMStringMap.right.at(success_state);
    failure_state_id_ = FSMStringMap.right.at(failure_state);
    registered_checks.emplace_back(
        [this] {
            return outcome_.load(std::memory_order_relaxed) ==
                   getup_control::GetUpOutcome::kSucceeded;
        },
        success_state_id_);
    registered_checks.emplace_back(
        [this] {
            return outcome_.load(std::memory_order_relaxed) ==
                   getup_control::GetUpOutcome::kFailed;
        },
        failure_state_id_);
}

State_GetUp::~State_GetUp()
{
    stop_policy_thread();
}

void State_GetUp::initialize_policy()
{
    if (env_) {
        return;
    }

    auto candidate = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir_ / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate));
    candidate->alg = std::make_unique<isaaclab::OrtRunner>(
        (policy_dir_ / "exported" / "policy.onnx").string());

    if (candidate->action_manager->total_action_dim() !=
            static_cast<int>(kG1ActionCount) ||
        candidate->alg->get_action().size() != kG1ActionCount ||
        candidate->robot->data.joint_ids_map.size() != kG1ActionCount ||
        candidate->robot->data.joint_stiffness.size() != kG1ActionCount ||
        candidate->robot->data.joint_damping.size() != kG1ActionCount) {
        throw std::runtime_error(
            "GetUp policy must use exactly 29 G1 joint actions");
    }
    env_ = std::move(candidate);
}

void State_GetUp::enter()
{
    stop_policy_thread();
    outcome_.store(
        getup_control::GetUpOutcome::kRunning, std::memory_order_relaxed);
    failure_reason_.store(
        getup_control::GetUpFailureReason::kNone,
        std::memory_order_relaxed);
    recovery_monitor_.reset();

    try {
        initialize_policy();
        env_->robot->update();
        if (env_->robot->data.joint_pos.size() !=
            static_cast<Eigen::Index>(kG1ActionCount)) {
            throw std::runtime_error(
                "GetUp robot state must contain 29 joint positions");
        }

        const auto motor_count = lowcmd->msg_.motor_cmd().size();
        if (!all_finite(env_->robot->data.joint_stiffness) ||
            !all_finite(env_->robot->data.joint_damping)) {
            throw std::runtime_error(
                "GetUp gains must contain only finite values");
        }
        std::vector<float> initial_targets(kG1ActionCount);
        for (std::size_t index = 0; index < kG1ActionCount; ++index) {
            const float mapped_index =
                env_->robot->data.joint_ids_map[index];
            const float joint_position =
                env_->robot->data.joint_pos[index];
            if (!std::isfinite(mapped_index) ||
                mapped_index < 0.0F ||
                std::floor(mapped_index) != mapped_index ||
                mapped_index >= static_cast<float>(motor_count) ||
                !std::isfinite(joint_position)) {
                throw std::runtime_error(
                    "GetUp joint state or motor map is invalid");
            }
            initial_targets[index] = joint_position;
        }
        for (std::size_t index = 0; index < kG1ActionCount; ++index) {
            const auto motor_index = static_cast<std::size_t>(
                env_->robot->data.joint_ids_map[index]);
            auto& motor = lowcmd->msg_.motor_cmd()[motor_index];
            motor.kp() = env_->robot->data.joint_stiffness[index];
            motor.kd() = env_->robot->data.joint_damping[index];
            motor.q() = initial_targets[index];
            motor.dq() = 0.0F;
            motor.tau() = 0.0F;
        }
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            latest_targets_ = std::move(initial_targets);
        }
    } catch (const std::exception& exception) {
        spdlog::error("GetUp entry failed: {}", exception.what());
        set_terminal_outcome(
            getup_control::GetUpOutcome::kFailed,
            getup_control::GetUpFailureReason::kInvalidInput);
        return;
    }

    policy_thread_running_.store(true, std::memory_order_relaxed);
    policy_thread_ = std::thread(&State_GetUp::policy_loop, this);
    spdlog::info("GetUp: recovery policy started with zero velocity command");
}

void State_GetUp::run()
{
    if (!env_) {
        return;
    }
    std::vector<float> targets;
    {
        std::lock_guard<std::mutex> lock(target_mutex_);
        targets = latest_targets_;
    }
    if (targets.size() != kG1ActionCount || !all_finite(targets)) {
        set_terminal_outcome(
            getup_control::GetUpOutcome::kFailed,
            getup_control::GetUpFailureReason::kInvalidInput);
        return;
    }
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const auto motor_index = static_cast<std::size_t>(
            env_->robot->data.joint_ids_map[index]);
        lowcmd->msg_.motor_cmd()[motor_index].q() = targets[index];
    }
}

void State_GetUp::exit()
{
    stop_policy_thread();
    const auto reason = failure_reason_.load(std::memory_order_relaxed);
    spdlog::info(
        "GetUp: recovery state exited (reason={})",
        failure_reason_name(reason));
}

void State_GetUp::policy_loop() noexcept
{
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(env_->step_dt));

    try {
        env_->reset();
    } catch (const std::exception& exception) {
        spdlog::error("GetUp reset failed: {}", exception.what());
        set_terminal_outcome(
            getup_control::GetUpOutcome::kFailed,
            getup_control::GetUpFailureReason::kInference);
        return;
    }

    auto last_sample_time = clock::now();
    auto next_deadline = last_sample_time;
    while (policy_thread_running_.load(std::memory_order_relaxed)) {
        next_deadline += period;
        bool inference_healthy = false;
        try {
            env_->step();
            const auto raw_actions = env_->alg->get_action();
            const auto processed_targets =
                env_->action_manager->processed_actions();
            if (raw_actions.size() != kG1ActionCount ||
                processed_targets.size() != kG1ActionCount ||
                !all_finite(raw_actions) ||
                !all_finite(processed_targets)) {
                throw std::runtime_error(
                    "GetUp inference returned invalid joint targets");
            }
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                latest_targets_ = processed_targets;
            }
            inference_healthy = true;
        } catch (const std::exception& exception) {
            spdlog::warn("GetUp inference failed: {}", exception.what());
        } catch (...) {
            spdlog::warn("GetUp inference failed with an unknown error");
        }

        const auto sample_time = clock::now();
        const double measured_dt = std::chrono::duration<double>(
            sample_time - last_sample_time).count();
        last_sample_time = sample_time;
        const auto result = recovery_monitor_.update(
            recovery_sample(inference_healthy), measured_dt);
        if (result == getup_control::GetUpOutcome::kSucceeded) {
            spdlog::info("GetUp: upright stability confirmed");
            set_terminal_outcome(
                result, getup_control::GetUpFailureReason::kNone);
            break;
        }
        if (result == getup_control::GetUpOutcome::kFailed) {
            const auto reason = recovery_monitor_.failure_reason();
            spdlog::error(
                "GetUp: recovery aborted ({})",
                failure_reason_name(reason));
            set_terminal_outcome(result, reason);
            break;
        }
        const auto now = clock::now();
        if (now < next_deadline) {
            std::this_thread::sleep_until(next_deadline);
        } else {
            next_deadline = now;
        }
    }
}

void State_GetUp::stop_policy_thread() noexcept
{
    policy_thread_running_.store(false, std::memory_order_relaxed);
    if (policy_thread_.joinable()) {
        policy_thread_.join();
    }
}

getup_control::GetUpRecoverySample State_GetUp::recovery_sample(
    bool inference_healthy) const noexcept
{
    const auto& data = env_->robot->data;
    const float torso_tilt = getup_control::torso_tilt_from_quaternion(
        data.root_quat_w.w(), data.root_quat_w.x(),
        data.root_quat_w.y(), data.root_quat_w.z());
    const float roll_pitch_speed = std::hypot(
        data.root_ang_vel_b[0], data.root_ang_vel_b[1]);
    float max_joint_speed = 0.0F;
    for (Eigen::Index index = 0; index < data.joint_vel.size(); ++index) {
        if (!std::isfinite(data.joint_vel[index])) {
            max_joint_speed =
                std::numeric_limits<float>::quiet_NaN();
            break;
        }
        max_joint_speed = std::max(
            max_joint_speed, std::abs(data.joint_vel[index]));
    }
    return {
        torso_tilt,
        roll_pitch_speed,
        max_joint_speed,
        inference_healthy,
    };
}

void State_GetUp::set_terminal_outcome(
    getup_control::GetUpOutcome outcome,
    getup_control::GetUpFailureReason reason) noexcept
{
    failure_reason_.store(reason, std::memory_order_relaxed);
    outcome_.store(outcome, std::memory_order_relaxed);
    policy_thread_running_.store(false, std::memory_order_relaxed);
}
