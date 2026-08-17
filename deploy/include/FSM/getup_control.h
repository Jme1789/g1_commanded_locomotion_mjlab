#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace getup_control {

inline float torso_tilt_from_quaternion(
    float w, float x, float y, float z) noexcept
{
    const double norm_squared =
        static_cast<double>(w) * w + static_cast<double>(x) * x +
        static_cast<double>(y) * y + static_cast<double>(z) * z;
    if (!std::isfinite(norm_squared) ||
        norm_squared <= std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const double normalized_x = static_cast<double>(x) / std::sqrt(norm_squared);
    const double normalized_y = static_cast<double>(y) / std::sqrt(norm_squared);
    const double upright_cosine = std::clamp(
        1.0 - 2.0 * (normalized_x * normalized_x +
                     normalized_y * normalized_y),
        -1.0, 1.0);
    return static_cast<float>(std::acos(upright_cosine));
}

struct GetUpRequestConfig
{
    double hold_seconds{1.0};
    float fallen_tilt_min_rad{1.0F};
    double max_update_gap_seconds{0.10};
};

class GetUpRequestGate
{
public:
    explicit GetUpRequestGate(GetUpRequestConfig config)
        : config_(config)
    {
        if (!std::isfinite(config_.hold_seconds) ||
            config_.hold_seconds <= 0.0 ||
            !std::isfinite(config_.fallen_tilt_min_rad) ||
            config_.fallen_tilt_min_rad < 0.0F ||
            !std::isfinite(config_.max_update_gap_seconds) ||
            config_.max_update_gap_seconds <= 0.0) {
            throw std::invalid_argument("invalid GetUp request configuration");
        }
    }

    void require_release() noexcept
    {
        armed_ = false;
        reset_pending();
        has_last_update_ = false;
    }

    bool update(bool requested, float torso_tilt_rad, double now_seconds) noexcept
    {
        if (!std::isfinite(torso_tilt_rad) || !std::isfinite(now_seconds)) {
            reset_pending();
            has_last_update_ = false;
            return false;
        }

        if (!requested) {
            armed_ = true;
            reset_pending();
            last_update_seconds_ = now_seconds;
            has_last_update_ = true;
            return false;
        }

        if (!armed_) {
            last_update_seconds_ = now_seconds;
            has_last_update_ = true;
            return false;
        }

        const bool fallen = torso_tilt_rad >= config_.fallen_tilt_min_rad;
        if (!fallen) {
            reset_pending();
            last_update_seconds_ = now_seconds;
            has_last_update_ = true;
            return false;
        }

        if (!has_last_update_ || !was_qualifying_) {
            held_seconds_ = 0.0;
            was_qualifying_ = true;
            last_update_seconds_ = now_seconds;
            has_last_update_ = true;
            return false;
        }

        const double delta_seconds = now_seconds - last_update_seconds_;
        last_update_seconds_ = now_seconds;
        if (delta_seconds < 0.0 ||
            delta_seconds > config_.max_update_gap_seconds + 1.0e-9) {
            held_seconds_ = 0.0;
            return false;
        }

        held_seconds_ += delta_seconds;
        if (held_seconds_ + 1.0e-9 < config_.hold_seconds) {
            return false;
        }

        armed_ = false;
        reset_pending();
        return true;
    }

private:
    void reset_pending() noexcept
    {
        held_seconds_ = 0.0;
        was_qualifying_ = false;
    }

    GetUpRequestConfig config_;
    bool armed_{true};
    bool was_qualifying_{false};
    bool has_last_update_{false};
    double held_seconds_{0.0};
    double last_update_seconds_{0.0};
};

struct FallenDetectorConfig
{
    float fallen_tilt_min_rad{1.0F};
    double confirm_duration_seconds{0.20};
    double max_update_gap_seconds{0.10};
};

class FallenDetector
{
public:
    explicit FallenDetector(FallenDetectorConfig config)
        : config_(config)
    {
        if (!std::isfinite(config_.fallen_tilt_min_rad) ||
            config_.fallen_tilt_min_rad < 0.0F ||
            !std::isfinite(config_.confirm_duration_seconds) ||
            config_.confirm_duration_seconds <= 0.0 ||
            !std::isfinite(config_.max_update_gap_seconds) ||
            config_.max_update_gap_seconds <= 0.0) {
            throw std::invalid_argument(
                "invalid Fallen detector configuration");
        }
    }

    void reset() noexcept
    {
        qualifying_ = false;
        has_last_update_ = false;
        confirmed_seconds_ = 0.0;
        last_update_seconds_ = 0.0;
    }

    bool update(float torso_tilt_rad, double now_seconds) noexcept
    {
        if (!std::isfinite(torso_tilt_rad) ||
            !std::isfinite(now_seconds)) {
            reset();
            return false;
        }

        const bool fallen =
            torso_tilt_rad >= config_.fallen_tilt_min_rad;
        if (!has_last_update_) {
            qualifying_ = fallen;
            has_last_update_ = true;
            last_update_seconds_ = now_seconds;
            confirmed_seconds_ = 0.0;
            return false;
        }

        const double delta_seconds = now_seconds - last_update_seconds_;
        last_update_seconds_ = now_seconds;
        if (delta_seconds < 0.0 ||
            delta_seconds > config_.max_update_gap_seconds + 1.0e-9) {
            qualifying_ = fallen;
            confirmed_seconds_ = 0.0;
            return false;
        }

        if (!fallen) {
            qualifying_ = false;
            confirmed_seconds_ = 0.0;
            return false;
        }
        if (!qualifying_) {
            qualifying_ = true;
            confirmed_seconds_ = 0.0;
            return false;
        }

        confirmed_seconds_ += delta_seconds;
        return confirmed_seconds_ + 1.0e-9 >=
               config_.confirm_duration_seconds;
    }

private:
    FallenDetectorConfig config_;
    bool qualifying_{false};
    bool has_last_update_{false};
    double confirmed_seconds_{0.0};
    double last_update_seconds_{0.0};
};

enum class GetUpOutcome
{
    kRunning,
    kSucceeded,
    kFailed,
};

enum class GetUpFailureReason
{
    kNone,
    kTimeout,
    kInference,
    kInvalidInput,
};

struct GetUpRecoveryConfig
{
    float upright_tilt_max_rad{0.35F};
    float angular_speed_max_rad_s{0.50F};
    float joint_speed_max_rad_s{1.0F};
    double stable_duration_seconds{1.0};
    double timeout_seconds{8.0};
    int max_consecutive_inference_failures{3};
    double max_sample_gap_seconds{0.10};
};

struct GetUpRecoverySample
{
    float torso_tilt_rad{0.0F};
    float roll_pitch_angular_speed_rad_s{0.0F};
    float max_joint_speed_rad_s{0.0F};
    bool inference_healthy{true};
};

class GetUpRecoveryMonitor
{
public:
    explicit GetUpRecoveryMonitor(GetUpRecoveryConfig config)
        : config_(config)
    {
        if (!std::isfinite(config_.upright_tilt_max_rad) ||
            config_.upright_tilt_max_rad < 0.0F ||
            !std::isfinite(config_.angular_speed_max_rad_s) ||
            config_.angular_speed_max_rad_s < 0.0F ||
            !std::isfinite(config_.joint_speed_max_rad_s) ||
            config_.joint_speed_max_rad_s < 0.0F ||
            !std::isfinite(config_.stable_duration_seconds) ||
            config_.stable_duration_seconds <= 0.0 ||
            !std::isfinite(config_.timeout_seconds) ||
            config_.timeout_seconds <= 0.0 ||
            config_.max_consecutive_inference_failures <= 0 ||
            !std::isfinite(config_.max_sample_gap_seconds) ||
            config_.max_sample_gap_seconds <= 0.0) {
            throw std::invalid_argument("invalid GetUp recovery configuration");
        }
    }

    void reset() noexcept
    {
        outcome_ = GetUpOutcome::kRunning;
        failure_reason_ = GetUpFailureReason::kNone;
        elapsed_seconds_ = 0.0;
        stable_seconds_ = 0.0;
        consecutive_inference_failures_ = 0;
    }

    GetUpOutcome update(const GetUpRecoverySample& sample, double dt_seconds) noexcept
    {
        if (outcome_ != GetUpOutcome::kRunning) {
            return outcome_;
        }
        if (!valid_sample(sample, dt_seconds)) {
            return fail(GetUpFailureReason::kInvalidInput);
        }

        elapsed_seconds_ += dt_seconds;
        const bool sample_gap =
            dt_seconds > config_.max_sample_gap_seconds + 1.0e-9;
        if (!sample.inference_healthy) {
            stable_seconds_ = 0.0;
            ++consecutive_inference_failures_;
            if (consecutive_inference_failures_ >=
                config_.max_consecutive_inference_failures) {
                return fail(GetUpFailureReason::kInference);
            }
        } else {
            consecutive_inference_failures_ = 0;
            const bool stable =
                sample.torso_tilt_rad <= config_.upright_tilt_max_rad &&
                sample.roll_pitch_angular_speed_rad_s <=
                    config_.angular_speed_max_rad_s &&
                sample.max_joint_speed_rad_s <= config_.joint_speed_max_rad_s;
            stable_seconds_ =
                stable && !sample_gap ? stable_seconds_ + dt_seconds : 0.0;
            if (stable_seconds_ + 1.0e-9 >=
                config_.stable_duration_seconds) {
                outcome_ = GetUpOutcome::kSucceeded;
                return outcome_;
            }
        }

        if (elapsed_seconds_ + 1.0e-9 >= config_.timeout_seconds) {
            return fail(GetUpFailureReason::kTimeout);
        }
        return outcome_;
    }

    [[nodiscard]] GetUpFailureReason failure_reason() const noexcept
    {
        return failure_reason_;
    }

private:
    static bool valid_sample(
        const GetUpRecoverySample& sample,
        double dt_seconds) noexcept
    {
        return std::isfinite(dt_seconds) && dt_seconds > 0.0 &&
               std::isfinite(sample.torso_tilt_rad) &&
               std::isfinite(sample.roll_pitch_angular_speed_rad_s) &&
               std::isfinite(sample.max_joint_speed_rad_s) &&
               sample.torso_tilt_rad >= 0.0F &&
               sample.roll_pitch_angular_speed_rad_s >= 0.0F &&
               sample.max_joint_speed_rad_s >= 0.0F;
    }

    GetUpOutcome fail(GetUpFailureReason reason) noexcept
    {
        outcome_ = GetUpOutcome::kFailed;
        failure_reason_ = reason;
        return outcome_;
    }

    GetUpRecoveryConfig config_;
    GetUpOutcome outcome_{GetUpOutcome::kRunning};
    GetUpFailureReason failure_reason_{GetUpFailureReason::kNone};
    double elapsed_seconds_{0.0};
    double stable_seconds_{0.0};
    int consecutive_inference_failures_{0};
};

}  // namespace getup_control
