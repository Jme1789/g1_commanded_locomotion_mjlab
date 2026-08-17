#pragma once

#include "FSM/FSMState.h"
#include "FSM/getup_control.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace isaaclab {
class ManagerBasedRLEnv;
}

class State_GetUp : public FSMState
{
public:
    State_GetUp(int state_mode, std::string state_string);
    ~State_GetUp();

    void enter() override;
    void run() override;
    void exit() override;

private:
    void initialize_policy();
    void policy_loop() noexcept;
    void stop_policy_thread() noexcept;
    getup_control::GetUpRecoverySample recovery_sample(
        bool inference_healthy) const noexcept;
    void set_terminal_outcome(
        getup_control::GetUpOutcome outcome,
        getup_control::GetUpFailureReason reason) noexcept;

    std::filesystem::path policy_dir_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;
    getup_control::GetUpRecoveryMonitor recovery_monitor_;
    int success_state_id_{0};
    int failure_state_id_{0};

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};
    std::atomic<getup_control::GetUpOutcome> outcome_{
        getup_control::GetUpOutcome::kRunning};
    std::atomic<getup_control::GetUpFailureReason> failure_reason_{
        getup_control::GetUpFailureReason::kNone};

    std::mutex target_mutex_;
    std::vector<float> latest_targets_;
};

REGISTER_FSM(State_GetUp)
