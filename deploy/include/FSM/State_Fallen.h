// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "State_Passive.h"

class State_Fallen : public State_Passive
{
public:
    State_Fallen(int state, std::string state_string = "Fallen")
        : State_Passive(state, std::move(state_string))
    {
    }

    void enter() override
    {
        State_Passive::enter();
        require_getup_request_release();
        spdlog::warn(
            "Fallen: damping protection active; release A before GetUp");
    }
};

REGISTER_FSM(State_Fallen)
