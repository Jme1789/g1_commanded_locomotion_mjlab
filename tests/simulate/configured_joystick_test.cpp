#include <cerrno>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gamepad/configured_joystick.h"
#include "gamepad/gamepad_profile.h"
#include "gamepad/logical_mapper.h"
#include "joystick/joystick.h"

namespace {

using gamepad::DeviceDescriptor;
using gamepad::DeviceIdentity;
using gamepad::EventBatch;
using gamepad::GamepadConfigError;
using gamepad::JoystickEventSource;
using gamepad::RawSnapshot;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                const std::string& message) {
  Expect(std::fabs(actual - expected) <= tolerance,
         message + ": got " + std::to_string(actual));
}

RawSnapshot Neutral(std::size_t axes = 8, std::size_t buttons = 12) {
  RawSnapshot raw{std::vector<std::int16_t>(axes, 0),
                  std::vector<std::uint8_t>(buttons, 0)};
  if (axes > 5) {
    raw.axes[4] = -32767;
    raw.axes[5] = -32767;
  }
  return raw;
}

RawSnapshot AllMappedPressed() {
  RawSnapshot raw = Neutral();
  raw.axes[0] = 16384;
  raw.axes[1] = -16384;
  raw.axes[2] = 8192;
  raw.axes[3] = -8192;
  raw.axes[4] = 32767;
  raw.axes[5] = 32767;
  raw.axes[7] = -32768;
  for (auto& button : raw.buttons) {
    button = 1;
  }
  return raw;
}

class FakeEventSource final : public JoystickEventSource {
 public:
  struct PendingBatch {
    std::vector<RawSnapshot> changes;
    bool connected{true};
  };

  explicit FakeEventSource(std::deque<PendingBatch> batches)
      : batches_(std::move(batches)) {}

  void Push(std::vector<RawSnapshot> changes, bool connected = true) {
    batches_.push_back(PendingBatch{std::move(changes), connected});
  }

  EventBatch Drain() override {
    if (batches_.empty()) {
      throw std::runtime_error("fake source exhausted");
    }
    PendingBatch batch = std::move(batches_.front());
    batches_.pop_front();
    Expect(!batch.changes.empty(), "fake batch needs a final snapshot");
    return EventBatch{std::move(batch.changes.back()), batch.connected};
  }

 private:
  std::deque<PendingBatch> batches_;
};

class ScriptedRead {
 public:
  enum class Kind { kEvent, kWouldBlock, kEof, kError };
  struct Result {
    Kind kind;
    JoystickEvent event{};
    int error{0};
  };

  explicit ScriptedRead(std::deque<Result> results)
      : results_(std::move(results)) {}

  ssize_t operator()(int, void* destination, std::size_t size) {
    Expect(!results_.empty(), "reader script exhausted");
    const Result result = results_.front();
    results_.pop_front();
    switch (result.kind) {
      case Kind::kEvent:
        Expect(size == sizeof(JoystickEvent), "reader requested wrong event size");
        *static_cast<JoystickEvent*>(destination) = result.event;
        return static_cast<ssize_t>(sizeof(JoystickEvent));
      case Kind::kWouldBlock:
      case Kind::kError:
        errno = result.error;
        return -1;
      case Kind::kEof:
        return 0;
    }
    throw std::runtime_error("unknown scripted result");
  }

 private:
  std::deque<Result> results_;
};

JoystickEvent AxisEvent(std::uint8_t number, std::int16_t value) {
  return JoystickEvent{1U, value, JS_EVENT_AXIS, number};
}

JoystickEvent ButtonEvent(std::uint8_t number, std::int16_t value) {
  return JoystickEvent{2U, value, JS_EVENT_BUTTON, number};
}

DeviceDescriptor Descriptor(std::size_t axes = 4, std::size_t buttons = 3) {
  return DeviceDescriptor{"/dev/input/js-test",
                          DeviceIdentity{"1234", "5678", "Test Pad", std::nullopt},
                          axes, buttons, std::nullopt};
}

std::unique_ptr<Joystick> MakeJoystick(
    std::deque<ScriptedRead::Result> script) {
  auto reader = std::make_shared<ScriptedRead>(std::move(script));
  return std::make_unique<Joystick>(
      123, false,
      [reader](int fd, void* destination, std::size_t size) {
        return (*reader)(fd, destination, size);
      });
}

void TestSampleContract() {
  const JoystickEvent expected = AxisEvent(1, 1234);
  auto joystick = MakeJoystick(
      {{ScriptedRead::Kind::kEvent, expected, 0},
       {ScriptedRead::Kind::kWouldBlock, {}, EAGAIN}});
  JoystickEvent actual{};
  Expect(joystick->sample(&actual), "sample must report one complete event");
  Expect(actual.number == 1 && actual.value == 1234 && actual.isAxis(),
         "sample must preserve the jstest event payload");
  Expect(!joystick->sample(&actual), "sample must remain false at EAGAIN");
}

void TestLinuxDrainAndBounds() {
  for (const int would_block : std::vector<int>{EAGAIN, EWOULDBLOCK}) {
    auto joystick = MakeJoystick(
        {{ScriptedRead::Kind::kEvent, AxisEvent(1, 111), 0},
         {ScriptedRead::Kind::kEvent, ButtonEvent(2, 1), 0},
         {ScriptedRead::Kind::kWouldBlock, {}, would_block}});
    gamepad::LinuxJoystickEventSource source(Descriptor(), std::move(joystick));
    const EventBatch batch = source.Drain();
    Expect(batch.connected, "EAGAIN must end a connected drain");
    Expect(batch.snapshot.axes.size() == 4 &&
               batch.snapshot.buttons.size() == 3,
           "raw storage must match detected capabilities exactly");
    Expect(batch.snapshot.axes[1] == 111 && batch.snapshot.buttons[2] == 1,
           "drain must retain every pending event in its final snapshot");
  }

  auto joystick = MakeJoystick(
      {{ScriptedRead::Kind::kEvent, AxisEvent(4, 222), 0},
       {ScriptedRead::Kind::kWouldBlock, {}, EAGAIN}});
  gamepad::LinuxJoystickEventSource source(Descriptor(), std::move(joystick));
  try {
    (void)source.Drain();
  } catch (const GamepadConfigError& error) {
    Expect(std::string(error.what()).find("axis") != std::string::npos,
           "out-of-range diagnostic must identify the axis");
    return;
  }
  throw std::runtime_error("out-of-range joystick event must be rejected");
}

void TestLinuxDisconnectResults() {
  const std::vector<ScriptedRead::Result> endings = {
      {ScriptedRead::Kind::kEof, {}, 0},
      {ScriptedRead::Kind::kError, {}, ENODEV},
      {ScriptedRead::Kind::kError, {}, EIO},
  };
  for (const auto& ending : endings) {
    auto joystick = MakeJoystick({ending});
    gamepad::LinuxJoystickEventSource source(Descriptor(), std::move(joystick));
    const EventBatch batch = source.Drain();
    Expect(!batch.connected, "EOF/ENODEV/EIO must report disconnection");
    Expect(batch.snapshot.axes.size() == 4 &&
               batch.snapshot.buttons.size() == 3,
           "disconnect must retain capability-sized storage");
  }
}

void TestCoherentMappingAndSetterCounts(
    const std::filesystem::path& profile_path) {
  const auto profile = gamepad::LoadGamepadProfile(profile_path);
  RawSnapshot intermediate = Neutral();
  intermediate.axes[0] = -32768;
  RawSnapshot final = AllMappedPressed();
  auto source = std::make_unique<FakeEventSource>(
      std::deque<FakeEventSource::PendingBatch>{
          {{Neutral()}, true}, {{intermediate, final}, true}});
  std::shared_ptr<unitree::common::UnitreeJoystick> joystick =
      std::make_shared<gamepad::ConfiguredJoystick>(profile, std::move(source));
  joystick->update();

  joystick->down(1);
  joystick->left(1);
  joystick->right(1);
  joystick->F1(1);
  joystick->F2(1);
  joystick->update();

  Expect(joystick->back.on_pressed && joystick->start.on_pressed &&
             joystick->LS.on_pressed && joystick->RS.on_pressed &&
             joystick->LB.on_pressed && joystick->RB.on_pressed &&
             joystick->A.on_pressed && joystick->B.on_pressed &&
             joystick->X.on_pressed && joystick->Y.on_pressed &&
             joystick->up.on_pressed,
         "each mapped button setter must run exactly once");
  Expect(joystick->down.on_released && joystick->left.on_released &&
             joystick->right.on_released && joystick->F1.on_released &&
             joystick->F2.on_released,
         "each neutral button setter must run exactly once");
  ExpectNear(joystick->lx(), (16384.0F / 32767.0F) * 0.03F, 1.0e-6F,
             "lx must receive one smoothed final-snapshot update");
  ExpectNear(joystick->ly(), (16384.0F / 32767.0F) * 0.03F, 1.0e-6F,
             "ly must preserve mapper inversion and Axis smoothing");
  ExpectNear(joystick->rx(), (8192.0F / 32767.0F) * 0.03F, 1.0e-6F,
             "rx must receive one setter call");
  ExpectNear(joystick->ry(), (8192.0F / 32767.0F) * 0.03F, 1.0e-6F,
             "ry must receive one setter call");
  ExpectNear(joystick->LT(), 0.03F, 1.0e-6F,
             "LT must retain Unitree Axis smoothing");
  ExpectNear(joystick->RT(), 0.03F, 1.0e-6F,
             "RT must retain Unitree Axis smoothing");
}

void TestTriggerAndButtonEdges(const std::filesystem::path& profile_path) {
  const auto profile = gamepad::LoadGamepadProfile(profile_path);
  auto source = std::make_unique<FakeEventSource>(
      std::deque<FakeEventSource::PendingBatch>{{{Neutral()}, true}});
  auto* source_ptr = source.get();
  std::shared_ptr<unitree::common::UnitreeJoystick> joystick =
      std::make_shared<gamepad::ConfiguredJoystick>(profile, std::move(source));
  joystick->update();

  RawSnapshot lt_up = Neutral();
  lt_up.axes[5] = 32767;
  lt_up.axes[7] = -32768;
  int up_edges = 0;
  for (int update = 0; !joystick->LT.pressed && update < 100; ++update) {
    source_ptr->Push({lt_up});
    joystick->update();
    up_edges += joystick->up.on_pressed ? 1 : 0;
  }
  Expect(joystick->LT.pressed,
         "repeated LT samples must cross inherited threshold");
  Expect(up_edges == 1, "held D-pad up must emit exactly one edge");

  for (int update = 0; joystick->LT.pressed && update < 100; ++update) {
    source_ptr->Push({Neutral()});
    joystick->update();
  }
  Expect(!joystick->LT.pressed, "released LT samples must cross back to neutral");

  RawSnapshot rt_a = Neutral();
  rt_a.axes[4] = 32767;
  rt_a.buttons[0] = 1;
  int a_edges = 0;
  for (int update = 0; !joystick->RT.pressed && update < 100; ++update) {
    source_ptr->Push({rt_a});
    joystick->update();
    a_edges += joystick->A.on_pressed ? 1 : 0;
  }
  Expect(joystick->RT.pressed,
         "repeated RT samples must cross inherited threshold");
  Expect(a_edges == 1, "held A must emit exactly one edge");
}

void TestButtonTriggerRepeatedFramesEmitOneEdge(
    const std::filesystem::path& profile_path) {
  auto profile = gamepad::LoadGamepadProfile(profile_path);
  profile.triggers.at("lt") = gamepad::ButtonTriggerBinding{6, 0.5F};

  auto source = std::make_unique<FakeEventSource>(
      std::deque<FakeEventSource::PendingBatch>{{{Neutral()}, true}});
  auto* source_ptr = source.get();
  auto joystick = std::make_shared<gamepad::ConfiguredJoystick>(
      std::move(profile), std::move(source));
  joystick->update();

  RawSnapshot pressed = Neutral();
  pressed.buttons[6] = 1;
  int press_edges = 0;
  for (int update = 0; update < 100; ++update) {
    source_ptr->Push({pressed, pressed});
    joystick->update();
    press_edges += joystick->LT.on_pressed ? 1 : 0;
  }

  int release_edges = 0;
  for (int update = 0; update < 100; ++update) {
    source_ptr->Push({Neutral(), Neutral()});
    joystick->update();
    release_edges += joystick->LT.on_released ? 1 : 0;
  }

  Expect(press_edges == 1,
         "repeated button-trigger frames must emit one press edge");
  Expect(release_edges == 1,
         "repeated button-trigger release frames must emit one release edge");
  Expect(!joystick->LT.pressed,
         "button trigger must finish in released state");
}

void TestProfileTriggerThresholdControlsPublishedBits(
    const std::filesystem::path& profile_path) {
  auto profile = gamepad::LoadGamepadProfile(profile_path);
  std::get<gamepad::AxisTriggerBinding>(profile.triggers.at("lt")).threshold =
      0.25F;
  std::get<gamepad::AxisTriggerBinding>(profile.triggers.at("rt")).threshold =
      0.75F;
  auto source = std::make_unique<FakeEventSource>(
      std::deque<FakeEventSource::PendingBatch>{{{Neutral()}, true}});
  auto* source_ptr = source.get();
  std::shared_ptr<unitree::common::UnitreeJoystick> joystick =
      std::make_shared<gamepad::ConfiguredJoystick>(
          std::move(profile), std::move(source));
  joystick->update();

  RawSnapshot below_thresholds = Neutral();
  below_thresholds.axes[5] = -19660;
  below_thresholds.axes[4] = 6553;
  for (int update = 0; update < 100; ++update) {
    source_ptr->Push({below_thresholds});
    joystick->update();
    const auto published = joystick->combine();
    Expect(!joystick->LT.pressed &&
               published.RF_RX.btn.components.L2 == 0,
           "LT below profile threshold must stay false in state and DDS");
    Expect(!joystick->RT.pressed &&
               published.RF_RX.btn.components.R2 == 0,
           "RT below profile threshold must stay false in state and DDS");
  }

  RawSnapshot above_thresholds = Neutral();
  above_thresholds.axes[5] = -6553;
  above_thresholds.axes[4] = 19660;
  int lt_edges = 0;
  int rt_edges = 0;
  for (int update = 0; update < 100; ++update) {
    source_ptr->Push({above_thresholds});
    joystick->update();
    const auto published = joystick->combine();
    Expect(joystick->LT.pressed ==
               (published.RF_RX.btn.components.L2 != 0),
           "LT state and published bit must cross together");
    Expect(joystick->RT.pressed ==
               (published.RF_RX.btn.components.R2 != 0),
           "RT state and published bit must cross together");
    lt_edges += joystick->LT.on_pressed ? 1 : 0;
    rt_edges += joystick->RT.on_pressed ? 1 : 0;
  }
  const auto published = joystick->combine();
  Expect(joystick->LT.pressed && published.RF_RX.btn.components.L2 != 0,
         "LT above profile threshold must eventually publish true");
  Expect(joystick->RT.pressed && published.RF_RX.btn.components.R2 != 0,
         "RT above profile threshold must eventually publish true");
  Expect(lt_edges == 1 && rt_edges == 1,
         "each digital trigger target must emit one inherited edge");
}

void TestDisconnectFailsClosedOnce(
    const std::filesystem::path& profile_path) {
  const auto profile = gamepad::LoadGamepadProfile(profile_path);
  auto source = std::make_unique<FakeEventSource>(
      std::deque<FakeEventSource::PendingBatch>{
          {{AllMappedPressed()}, true},
          {{Neutral()}, false},
          {{Neutral()}, false}});
  std::shared_ptr<unitree::common::UnitreeJoystick> joystick =
      std::make_shared<gamepad::ConfiguredJoystick>(profile, std::move(source));
  joystick->update();

  joystick->down(1);
  joystick->left(1);
  joystick->right(1);
  joystick->F1(1);
  joystick->F2(1);
  joystick->lx.smooth = 0.11F;
  joystick->ly.smooth = 0.12F;
  joystick->rx.smooth = 0.13F;
  joystick->ry.smooth = 0.14F;
  joystick->LT.smooth = 0.15F;
  joystick->RT.smooth = 0.16F;

  const auto all_buttons_neutral = [&joystick]() {
    return joystick->back() == 0 && joystick->start() == 0 &&
           joystick->LS() == 0 && joystick->RS() == 0 &&
           joystick->LB() == 0 && joystick->RB() == 0 &&
           joystick->A() == 0 && joystick->B() == 0 &&
           joystick->X() == 0 && joystick->Y() == 0 &&
           joystick->up() == 0 && joystick->down() == 0 &&
           joystick->left() == 0 && joystick->right() == 0 &&
           joystick->F1() == 0 && joystick->F2() == 0;
  };
  const auto all_buttons_released_once = [&joystick]() {
    return joystick->back.on_released && joystick->start.on_released &&
           joystick->LS.on_released && joystick->RS.on_released &&
           joystick->LB.on_released && joystick->RB.on_released &&
           joystick->A.on_released && joystick->B.on_released &&
           joystick->X.on_released && joystick->Y.on_released &&
           joystick->up.on_released && joystick->down.on_released &&
           joystick->left.on_released && joystick->right.on_released &&
           joystick->F1.on_released && joystick->F2.on_released;
  };
  const auto all_axes_neutral = [&joystick]() {
    return joystick->lx() == 0.0F && joystick->ly() == 0.0F &&
           joystick->rx() == 0.0F && joystick->ry() == 0.0F &&
           joystick->LT() == 0.0F && joystick->RT() == 0.0F;
  };
  const auto smoothing_restored = [&joystick]() {
    return joystick->lx.smooth == 0.11F && joystick->ly.smooth == 0.12F &&
           joystick->rx.smooth == 0.13F && joystick->ry.smooth == 0.14F &&
           joystick->LT.smooth == 0.15F && joystick->RT.smooth == 0.16F;
  };

  std::ostringstream errors;
  auto* original = std::cerr.rdbuf(errors.rdbuf());
  joystick->update();
  const bool neutral_once =
      all_buttons_neutral() && all_buttons_released_once() &&
      all_axes_neutral() && smoothing_restored() &&
      !joystick->LT.pressed && !joystick->RT.pressed &&
      joystick->combine().RF_RX.btn.components.L2 == 0 &&
      joystick->combine().RF_RX.btn.components.R2 == 0;
  const std::string first_error = errors.str();
  joystick->update();
  const bool unchanged_after_latch =
      all_buttons_neutral() && all_buttons_released_once() &&
      all_axes_neutral() && smoothing_restored();
  std::cerr.rdbuf(original);

  Expect(neutral_once,
         "first disconnect must force every inherited output exactly neutral");
  Expect(!first_error.empty(), "disconnect must report one error");
  Expect(unchanged_after_latch,
         "later disconnect polls must not repeat the neutral setter update");
  Expect(errors.str() == first_error,
         "later disconnect polls must not emit another error");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Expect(argc == 2, "expected profile fixture path");
    TestSampleContract();
    TestLinuxDrainAndBounds();
    TestLinuxDisconnectResults();
    TestCoherentMappingAndSetterCounts(argv[1]);
    TestTriggerAndButtonEdges(argv[1]);
    TestButtonTriggerRepeatedFramesEmitOneEdge(argv[1]);
    TestDisconnectFailsClosedOnce(argv[1]);
    TestProfileTriggerThresholdControlsPublishedBits(argv[1]);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
