#pragma once

#include <errno.h>
#include <fcntl.h>
#include <linux/joystick.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

class CustomJoystick {
 public:
  explicit CustomJoystick(const std::string& device = "/dev/input/js0")
      : fd_(::open(device.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC)),
        owns_fd_(true) {
    if (fd_ < 0) {
      throw std::runtime_error(
          "CustomJoystick: failed to open '" + device + "': " +
          std::strerror(errno));
    }
    std::cout << "CustomJoystick: reading from '" << device << "'\n";
  }

  CustomJoystick(int fd, bool owns_fd) : fd_(fd), owns_fd_(owns_fd) {
    if (fd_ < 0) {
      throw std::invalid_argument("CustomJoystick: fd must be non-negative");
    }
    const int flags = ::fcntl(fd_, F_GETFL, 0);
    if (flags < 0 || ::fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
      const int error = errno;
      if (owns_fd_) {
        ::close(fd_);
      }
      fd_ = -1;
      throw std::runtime_error(
          "CustomJoystick: failed to configure nonblocking input: " +
          std::string(std::strerror(error)));
    }
  }

  CustomJoystick(const CustomJoystick&) = delete;
  CustomJoystick& operator=(const CustomJoystick&) = delete;

  ~CustomJoystick() {
    if (owns_fd_ && fd_ >= 0) {
      ::close(fd_);
    }
  }

  [[nodiscard]] bool is_connected() const noexcept {
    return connected_ && fd_ >= 0;
  }

  void poll() noexcept {
    if (!is_connected()) {
      return;
    }

    while (true) {
      js_event event{};
      const ssize_t count = ::read(fd_, &event, sizeof(event));
      if (count == static_cast<ssize_t>(sizeof(event))) {
        handle_event(event);
        continue;
      }
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return;
      }
      disconnect();
      return;
    }
  }

  void apply_to(unitree::common::UnitreeJoystick& joystick) const {
    const bool immediate = !is_connected();
    std::array<unitree::common::Axis*, 6> target_axes{
        &joystick.lx, &joystick.ly, &joystick.rx,
        &joystick.ry, &joystick.LT, &joystick.RT};
    std::array<float, 6> previous_smoothing{};
    if (immediate) {
      for (std::size_t index = 0; index < target_axes.size(); ++index) {
        previous_smoothing[index] = target_axes[index]->smooth;
        target_axes[index]->smooth = 1.0F;
      }
    }

    joystick.lx(axes_[0]);
    joystick.ly(-axes_[1]);
    joystick.rx(axes_[2]);
    joystick.ry(-axes_[3]);
    joystick.LT(triggers_[0]);
    joystick.RT(triggers_[1]);

    joystick.A(buttons_[0]);
    joystick.B(buttons_[1]);
    joystick.X(buttons_[3]);
    joystick.Y(buttons_[4]);
    joystick.LB(buttons_[6]);
    joystick.RB(buttons_[7]);
    joystick.LS(buttons_[8]);
    joystick.RS(buttons_[9]);
    joystick.back(buttons_[10]);
    joystick.start(buttons_[11]);

    joystick.left(dpad_x_ < 0);
    joystick.right(dpad_x_ > 0);
    joystick.up(dpad_y_ < 0);
    joystick.down(dpad_y_ > 0);

    if (immediate) {
      for (std::size_t index = 0; index < target_axes.size(); ++index) {
        target_axes[index]->smooth = previous_smoothing[index];
      }
    }
  }

 private:
  static float normalize_axis(std::int16_t value) noexcept {
    const float normalized =
        value < 0 ? static_cast<float>(value) / 32768.0F
                  : static_cast<float>(value) / 32767.0F;
    return std::fabs(normalized) < 0.05F
               ? 0.0F
               : std::clamp(normalized, -1.0F, 1.0F);
  }

  static float normalize_trigger(std::int16_t value) noexcept {
    return std::clamp(
        (static_cast<float>(value) + 32767.0F) / 65534.0F, 0.0F, 1.0F);
  }

  void handle_event(const js_event& event) noexcept {
    const std::uint8_t type = event.type & ~JS_EVENT_INIT;
    if (type == JS_EVENT_BUTTON && event.number < buttons_.size()) {
      buttons_[event.number] = event.value != 0;
      return;
    }
    if (type != JS_EVENT_AXIS) {
      return;
    }

    if (event.number < 4) {
      axes_[event.number] = normalize_axis(event.value);
    } else if (event.number == 4) {
      triggers_[1] = normalize_trigger(event.value);
    } else if (event.number == 5) {
      triggers_[0] = normalize_trigger(event.value);
    } else if (event.number == 6) {
      dpad_x_ = event.value > 16384 ? 1 : event.value < -16384 ? -1 : 0;
    } else if (event.number == 7) {
      dpad_y_ = event.value > 16384 ? 1 : event.value < -16384 ? -1 : 0;
    }
  }

  void disconnect() noexcept {
    if (!connected_) {
      return;
    }
    connected_ = false;
    axes_.fill(0.0F);
    triggers_.fill(0.0F);
    buttons_.fill(false);
    dpad_x_ = 0;
    dpad_y_ = 0;
    std::cerr << "CustomJoystick: disconnected; input forced to neutral. "
                 "Restart g1_ctrl after reconnecting the gamepad.\n";
  }

  int fd_{-1};
  bool owns_fd_{false};
  bool connected_{true};
  std::array<float, 4> axes_{};
  std::array<float, 2> triggers_{};
  std::array<bool, 16> buttons_{};
  int dpad_x_{0};
  int dpad_y_{0};
};
