#include "FSM/getup_control.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool hold_until(
    getup_control::GetUpRequestGate& gate,
    double start_s,
    double end_s,
    bool requested,
    float tilt_rad)
{
    bool fired = false;
    for (double now = start_s; now <= end_s + 1.0e-9; now += 0.05) {
        fired = gate.update(requested, tilt_rad, now) || fired;
    }
    return fired;
}

void test_request_counts_only_fallen_hold_time()
{
    getup_control::GetUpRequestGate gate({1.0, 1.0F, 0.10});

    require(!hold_until(gate, 0.0, 1.5, true, 0.2F),
            "upright A hold must not trigger GetUp");
    require(!hold_until(gate, 1.55, 2.50, true, 1.2F),
            "fallen hold shorter than one second must not trigger");
    require(gate.update(true, 1.2F, 2.55),
            "one continuous fallen second must trigger GetUp");
}

void test_release_and_sample_gap_cancel_pending_hold()
{
    getup_control::GetUpRequestGate gate({1.0, 1.0F, 0.10});

    require(!hold_until(gate, 0.0, 0.60, true, 1.3F),
            "partial hold must remain pending");
    require(!gate.update(false, 1.3F, 0.65), "release cannot trigger");
    require(!hold_until(gate, 0.70, 1.60, true, 1.3F),
            "release must reset accumulated hold time");
    require(gate.update(true, 1.3F, 1.70),
            "new continuous hold must trigger after reset");

    gate.update(false, 1.3F, 1.75);
    require(!hold_until(gate, 2.0, 2.60, true, 1.3F),
            "a long sampling gap must start a new hold window");
    require(!gate.update(true, 1.3F, 3.0),
            "another long gap must not complete an old hold");
}

void test_request_is_one_shot_until_physical_release()
{
    getup_control::GetUpRequestGate gate({1.0, 1.0F, 0.10});

    require(hold_until(gate, 0.0, 1.05, true, 1.2F),
            "first hold must trigger");
    require(!hold_until(gate, 1.10, 3.0, true, 1.2F),
            "held A must not auto-retry after a consumed request");
    require(!gate.update(false, 1.2F, 3.05), "release only rearms");
    require(hold_until(gate, 3.10, 4.15, true, 1.2F),
            "a fresh press after release must trigger again");
}

void test_fallen_entry_requires_release_before_getup_hold()
{
    getup_control::GetUpRequestGate gate({1.0, 1.0F, 0.10});

    gate.require_release();
    require(!hold_until(gate, 0.0, 2.0, true, 1.2F),
            "A held across Fallen entry must not trigger GetUp");
    require(!gate.update(false, 1.2F, 2.05),
            "physical release only rearms GetUp");
    require(!hold_until(gate, 2.10, 3.05, true, 1.2F),
            "a fresh hold shorter than one second must not trigger");
    require(gate.update(true, 1.2F, 3.10),
            "a fresh one-second hold after release must trigger GetUp");
}

void test_fallen_detector_requires_continuous_tilt()
{
    getup_control::FallenDetector detector({1.0F, 0.20, 0.10});

    require(!detector.update(1.2F, 0.00),
            "first fallen sample only starts confirmation");
    require(!detector.update(1.2F, 0.05),
            "short tilt must remain unconfirmed");
    require(!detector.update(0.8F, 0.10),
            "upright sample must cancel pending fall");
    require(!detector.update(1.2F, 0.15),
            "new tilt starts a new confirmation window");
    require(!detector.update(1.2F, 0.25),
            "less than 0.2 continuous seconds must remain unconfirmed");
    require(detector.update(1.2F, 0.35),
            "0.2 continuous tilted seconds must confirm Fallen");
}

void test_fallen_detector_resets_on_invalid_or_late_samples()
{
    getup_control::FallenDetector detector({1.0F, 0.20, 0.10});

    require(!detector.update(1.2F, 0.00),
            "first fallen sample starts confirmation");
    require(!detector.update(1.2F, 0.10),
            "partial confirmation must remain pending");
    require(!detector.update(1.2F, 0.25),
            "a sample gap over the limit must reset confirmation");
    require(!detector.update(1.2F, 0.35),
            "confirmation must rebuild after a gap");
    require(detector.update(1.2F, 0.45),
            "a complete fresh window after a gap must confirm");

    detector.reset();
    require(!detector.update(1.2F, 1.00),
            "reset detector starts a new window");
    require(!detector.update(
                std::numeric_limits<float>::quiet_NaN(), 1.10),
            "non-finite tilt must fail closed and reset");
    require(!detector.update(1.2F, 1.15),
            "valid input after invalid data starts over");
    require(!detector.update(1.2F, 1.25),
            "invalid data must not preserve old elapsed time");
    require(detector.update(1.2F, 1.35),
            "a complete fresh window after invalid data must confirm");
}

void test_invalid_fallen_detector_config_is_rejected()
{
    bool rejected = false;
    try {
        getup_control::FallenDetector invalid(
            {1.0F, 0.20, 0.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected,
            "non-positive detector sample gap must be rejected");
}

void test_invalid_request_samples_fail_closed()
{
    getup_control::GetUpRequestGate gate({1.0, 1.0F, 0.10});
    require(!hold_until(gate, 0.0, 0.60, true, 1.2F),
            "partial hold must remain pending");
    require(!gate.update(true, std::numeric_limits<float>::quiet_NaN(), 0.65),
            "non-finite tilt must fail closed");
    require(!hold_until(gate, 0.70, 1.60, true, 1.2F),
            "invalid input must reset pending hold time");
}

void test_quaternion_tilt_matches_upright_side_and_inverted_poses()
{
    constexpr float half_sqrt_two = 0.70710678118F;
    const float upright = getup_control::torso_tilt_from_quaternion(
        1.0F, 0.0F, 0.0F, 0.0F);
    const float side = getup_control::torso_tilt_from_quaternion(
        half_sqrt_two, half_sqrt_two, 0.0F, 0.0F);
    const float inverted = getup_control::torso_tilt_from_quaternion(
        0.0F, 1.0F, 0.0F, 0.0F);
    const float yaw_only = getup_control::torso_tilt_from_quaternion(
        half_sqrt_two, 0.0F, 0.0F, half_sqrt_two);

    require(std::abs(upright) < 1.0e-6F,
            "upright quaternion must have zero torso tilt");
    require(std::abs(side - 1.57079632679F) < 1.0e-5F,
            "side fall must have pi/2 torso tilt");
    require(std::abs(inverted - 3.14159265359F) < 1.0e-5F,
            "inverted quaternion must have pi torso tilt");
    require(std::abs(yaw_only) < 1.0e-5F,
            "yaw rotation must not count as torso tilt");
    require(std::isnan(getup_control::torso_tilt_from_quaternion(
                0.0F, 0.0F, 0.0F, 0.0F)),
            "invalid quaternion must fail closed as non-finite tilt");
}

getup_control::GetUpRecoverySample stable_sample()
{
    return {0.20F, 0.20F, 0.40F, true};
}

void test_recovery_requires_continuous_stability()
{
    getup_control::GetUpRecoveryMonitor monitor(
        {0.35F, 0.50F, 1.0F, 1.0, 8.0, 3});

    for (int index = 0; index < 40; ++index) {
        require(monitor.update(stable_sample(), 0.02) ==
                    getup_control::GetUpOutcome::kRunning,
                "less than one stable second must remain running");
    }
    auto unstable = stable_sample();
    unstable.max_joint_speed_rad_s = 1.2F;
    require(monitor.update(unstable, 0.02) ==
                getup_control::GetUpOutcome::kRunning,
            "one unstable sample must reset stability");
    for (int index = 0; index < 49; ++index) {
        require(monitor.update(stable_sample(), 0.02) ==
                    getup_control::GetUpOutcome::kRunning,
                "reset stability window must be rebuilt continuously");
    }
    require(monitor.update(stable_sample(), 0.02) ==
                getup_control::GetUpOutcome::kSucceeded,
            "one continuous stable second must complete recovery");
}

void test_recovery_timeout_and_inference_failures_abort()
{
    getup_control::GetUpRecoveryMonitor timeout_monitor(
        {0.35F, 0.50F, 1.0F, 1.0, 0.10, 3});
    getup_control::GetUpRecoverySample fallen{1.2F, 0.1F, 0.1F, true};
    for (int index = 0; index < 4; ++index) {
        require(timeout_monitor.update(fallen, 0.02) ==
                    getup_control::GetUpOutcome::kRunning,
                "recovery must run before timeout");
    }
    require(timeout_monitor.update(fallen, 0.02) ==
                getup_control::GetUpOutcome::kFailed,
            "recovery timeout must fail");
    require(timeout_monitor.failure_reason() ==
                getup_control::GetUpFailureReason::kTimeout,
            "timeout failure reason must be retained");

    getup_control::GetUpRecoveryMonitor inference_monitor(
        {0.35F, 0.50F, 1.0F, 1.0, 8.0, 3});
    auto failed_inference = stable_sample();
    failed_inference.inference_healthy = false;
    require(inference_monitor.update(failed_inference, 0.02) ==
                getup_control::GetUpOutcome::kRunning,
            "one inference failure must preserve the last valid target");
    require(inference_monitor.update(failed_inference, 0.02) ==
                getup_control::GetUpOutcome::kRunning,
            "two inference failures must remain below the limit");
    require(inference_monitor.update(failed_inference, 0.02) ==
                getup_control::GetUpOutcome::kFailed,
            "three consecutive inference failures must abort");
    require(inference_monitor.failure_reason() ==
                getup_control::GetUpFailureReason::kInference,
            "inference failure reason must be retained");
}

void test_invalid_recovery_sample_aborts()
{
    getup_control::GetUpRecoveryMonitor monitor(
        {0.35F, 0.50F, 1.0F, 1.0, 8.0, 3});
    auto invalid = stable_sample();
    invalid.torso_tilt_rad = std::numeric_limits<float>::infinity();
    require(monitor.update(invalid, 0.02) ==
                getup_control::GetUpOutcome::kFailed,
            "non-finite recovery input must abort");
    require(monitor.failure_reason() ==
                getup_control::GetUpFailureReason::kInvalidInput,
            "invalid input failure reason must be retained");
}

void test_long_sample_gap_resets_stability_but_counts_wall_time()
{
    getup_control::GetUpRecoveryMonitor monitor(
        {0.35F, 0.50F, 1.0F, 1.0, 8.0, 3, 0.10});
    require(monitor.update(stable_sample(), 0.60) ==
                getup_control::GetUpOutcome::kRunning,
            "a scheduler gap must not count as continuous stability");
    for (int index = 0; index < 49; ++index) {
        require(monitor.update(stable_sample(), 0.02) ==
                    getup_control::GetUpOutcome::kRunning,
                "stability must be rebuilt after a scheduler gap");
    }
    require(monitor.update(stable_sample(), 0.02) ==
                getup_control::GetUpOutcome::kSucceeded,
            "one real continuous second after the gap must succeed");

    getup_control::GetUpRecoveryMonitor timeout_monitor(
        {0.35F, 0.50F, 1.0F, 1.0, 0.50, 3, 0.10});
    require(timeout_monitor.update(stable_sample(), 0.60) ==
                getup_control::GetUpOutcome::kFailed,
            "timeout must use measured wall-clock delta");
}

}  // namespace

int main()
{
    try {
        test_request_counts_only_fallen_hold_time();
        test_release_and_sample_gap_cancel_pending_hold();
        test_request_is_one_shot_until_physical_release();
        test_fallen_entry_requires_release_before_getup_hold();
        test_fallen_detector_requires_continuous_tilt();
        test_fallen_detector_resets_on_invalid_or_late_samples();
        test_invalid_fallen_detector_config_is_rejected();
        test_invalid_request_samples_fail_closed();
        test_quaternion_tilt_matches_upright_side_and_inverted_poses();
        test_recovery_requires_continuous_stability();
        test_recovery_timeout_and_inference_failures_abort();
        test_invalid_recovery_sample_aborts();
        test_long_sample_gap_resets_stability_but_counts_wall_time();
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "all GetUp control tests passed\n";
    return EXIT_SUCCESS;
}
