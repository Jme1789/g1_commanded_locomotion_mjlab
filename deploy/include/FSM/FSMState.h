#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "FSM/getup_control.h"
#include "common/CustomJoystick.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                if (target_fsm == "GetUp") {
                    registered_checks.emplace_back(
                        std::make_pair(
                            [func]()->bool {
                                const bool requested =
                                    func(FSMState::lowstate->joystick);
                                return getup_request_gate().update(
                                    requested, getup_torso_tilt_rad(),
                                    monotonic_seconds());
                            },
                            fsm_id
                        )
                    );
                } else {
                    registered_checks.emplace_back(
                        std::make_pair(
                            [func]()->bool{ return func(FSMState::lowstate->joystick); },
                            fsm_id
                        )
                    );
                }
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if (custom_joystick) {
            custom_joystick->poll();
            custom_joystick->apply_to(lowstate->joystick);
        }
        if(keyboard) keyboard->update();
        auto& getup_gate = getup_request_gate_storage();
        if (getup_gate && !lowstate->joystick.A.pressed) {
            getup_gate->update(false, 0.0F, monotonic_seconds());
        }
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
    static std::unique_ptr<CustomJoystick> custom_joystick;

protected:
    static double monotonic_seconds() noexcept
    {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(
            clock::now().time_since_epoch()).count();
    }

    static float getup_torso_tilt_rad() noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            const auto& quaternion = lowstate->msg_.imu_state().quaternion();
            if (quaternion.size() < 4) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return getup_control::torso_tilt_from_quaternion(
                quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
        } catch (...) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    static void require_getup_request_release()
    {
        getup_request_gate().require_release();
    }

private:
    static getup_control::GetUpRequestGate& getup_request_gate()
    {
        auto& gate = getup_request_gate_storage();
        if (!gate) {
            const auto trigger = param::config["FSM"]["GetUp"]["trigger"];
            gate = std::make_unique<getup_control::GetUpRequestGate>(
                getup_control::GetUpRequestConfig{
                    trigger["hold_s"].as<double>(),
                    trigger["fallen_tilt_min_rad"].as<float>(),
                    trigger["max_update_gap_s"].as<double>(),
                });
        }
        return *gate;
    }

    static std::unique_ptr<getup_control::GetUpRequestGate>&
    getup_request_gate_storage() noexcept
    {
        static std::unique_ptr<getup_control::GetUpRequestGate> gate;
        return gate;
    }
};
