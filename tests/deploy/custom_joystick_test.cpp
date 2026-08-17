#include <fcntl.h>
#include <linux/joystick.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "common/CustomJoystick.h"
#include "param.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

void expect_near(float actual, float expected, const std::string& message) {
  if (std::fabs(actual - expected) > 1.0e-4F) {
    std::cerr << "FAIL: " << message << " (actual=" << actual
              << ", expected=" << expected << ")\n";
    std::exit(1);
  }
}

void write_event(int fd, std::uint8_t type, std::uint8_t number,
                 std::int16_t value) {
  js_event event{};
  event.type = type;
  event.number = number;
  event.value = value;
  const auto written = ::write(fd, &event, sizeof(event));
  expect_true(written == static_cast<ssize_t>(sizeof(event)),
              "synthetic joystick event must be written completely");
}

void make_axes_immediate(unitree::common::UnitreeJoystick& target) {
  target.lx.smooth = 1.0F;
  target.ly.smooth = 1.0F;
  target.rx.smooth = 1.0F;
  target.ry.smooth = 1.0F;
  target.LT.smooth = 1.0F;
  target.RT.smooth = 1.0F;
}

void test_verified_beitong_mapping_and_edges() {
  int pipe_fds[2]{};
  expect_true(::pipe2(pipe_fds, O_NONBLOCK) == 0,
              "nonblocking test pipe must open");

  CustomJoystick source(pipe_fds[0], true);
  unitree::common::UnitreeJoystick target;
  make_axes_immediate(target);

  write_event(pipe_fds[1], JS_EVENT_AXIS, 0, 16384);
  write_event(pipe_fds[1], JS_EVENT_AXIS, 1, -32767);
  write_event(pipe_fds[1], JS_EVENT_AXIS, 2, 24575);
  write_event(pipe_fds[1], JS_EVENT_AXIS, 3, -16384);
  write_event(pipe_fds[1], JS_EVENT_AXIS, 4, 32767);
  write_event(pipe_fds[1], JS_EVENT_AXIS, 5, 32767);
  write_event(pipe_fds[1], JS_EVENT_AXIS, 7, -32767);
  write_event(pipe_fds[1], JS_EVENT_BUTTON, 0, 1);
  write_event(pipe_fds[1], JS_EVENT_BUTTON, 6, 1);
  write_event(pipe_fds[1], JS_EVENT_BUTTON, 7, 1);
  write_event(pipe_fds[1], JS_EVENT_BUTTON, 11, 1);

  source.poll();
  source.apply_to(target);

  expect_near(target.lx(), 16384.0F / 32767.0F, "axis 0 maps to left X");
  expect_near(target.ly(), 1.0F, "axis 1 maps to inverted left Y");
  expect_near(target.rx(), 24575.0F / 32767.0F, "axis 2 maps to right X");
  expect_near(target.ry(), 16384.0F / 32768.0F,
              "axis 3 maps to inverted right Y");
  expect_true(target.RT.pressed, "axis 4 maps to RT");
  expect_true(target.LT.pressed, "axis 5 maps to LT");
  expect_true(target.up.pressed, "negative axis 7 maps to D-pad Up");
  expect_true(target.A.pressed, "button 0 maps to A");
  expect_true(target.LB.pressed, "button 6 maps to LB");
  expect_true(target.RB.pressed, "button 7 maps to RB");
  expect_true(target.start.on_pressed, "button 11 creates one Start edge");

  source.apply_to(target);
  expect_true(target.start.pressed, "held Start remains level-pressed");
  expect_true(!target.start.on_pressed,
              "repeated frames do not create duplicate Start edges");

  write_event(pipe_fds[1], JS_EVENT_BUTTON, 11, 0);
  source.poll();
  source.apply_to(target);
  expect_true(target.start.on_released,
              "releasing Start creates one release edge");

  ::close(pipe_fds[1]);
}

void test_disconnect_forces_every_output_neutral() {
  int pipe_fds[2]{};
  expect_true(::pipe2(pipe_fds, O_NONBLOCK) == 0,
              "nonblocking test pipe must open");

  CustomJoystick source(pipe_fds[0], true);
  unitree::common::UnitreeJoystick target;
  make_axes_immediate(target);

  write_event(pipe_fds[1], JS_EVENT_AXIS, 0, 32767);
  write_event(pipe_fds[1], JS_EVENT_AXIS, 5, 32767);
  write_event(pipe_fds[1], JS_EVENT_BUTTON, 0, 1);
  source.poll();
  source.apply_to(target);
  expect_true(target.A.pressed && target.LT.pressed && target.lx() > 0.9F,
              "fixture must begin non-neutral");

  ::close(pipe_fds[1]);
  source.poll();
  source.apply_to(target);

  expect_true(!source.is_connected(), "EOF marks the controller disconnected");
  expect_true(!target.A.pressed && !target.LT.pressed,
              "disconnect releases held controls");
  expect_near(target.lx(), 0.0F, "disconnect zeros stick axes immediately");
}

void test_missing_device_fails_closed() {
  bool rejected = false;
  try {
    CustomJoystick source("/definitely/missing/js-device");
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  expect_true(rejected, "an unavailable requested device must reject startup");
}

void test_custom_joystick_cli_selects_requested_device() {
  char executable[] = "custom_joystick_test";
  char option[] = "--custom-joystick";
  char device[] = "/dev/input/js7";
  char* arguments[] = {executable, option, device};

  (void)param::helper(3, arguments);

  expect_true(param::custom_joystick == "/dev/input/js7",
              "CLI must retain the explicitly selected custom joystick");
}

}  // namespace

int main() {
  test_verified_beitong_mapping_and_edges();
  test_disconnect_forces_every_output_neutral();
  test_missing_device_fails_closed();
  test_custom_joystick_cli_selects_requested_device();
  std::cout << "all custom joystick tests passed\n";
  return 0;
}
