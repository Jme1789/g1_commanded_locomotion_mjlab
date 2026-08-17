#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gamepad/device_discovery.h"
#include "gamepad/gamepad_profile.h"
#include "gamepad/logical_mapper.h"

namespace {

using gamepad::ActiveSelection;
using gamepad::AxisDpadBinding;
using gamepad::AxisTriggerBinding;
using gamepad::ButtonBinding;
using gamepad::ButtonTriggerBinding;
using gamepad::DeviceDescriptor;
using gamepad::DeviceIdentity;
using gamepad::GamepadConfigError;
using gamepad::GamepadProfile;
using gamepad::RawSnapshot;
using gamepad::StickBinding;
using gamepad::UnsupportedBinding;

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

void WriteText(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot write test fixture " + path.string());
  }
  stream << text;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot read test fixture " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

std::string ReplaceOnce(std::string text, const std::string& from,
                        const std::string& to) {
  const auto position = text.find(from);
  if (position == std::string::npos) {
    throw std::runtime_error("test fixture token not found: " + from);
  }
  text.replace(position, from.size(), to);
  return text;
}

void ExpectConfigError(const std::function<void()>& operation,
                       const std::string& field_path) {
  try {
    operation();
  } catch (const GamepadConfigError& error) {
    Expect(std::string(error.what()).rfind(field_path + ":", 0) == 0,
           "diagnostic must contain full path " + field_path + ": " + error.what());
    return;
  }
  throw std::runtime_error("expected GamepadConfigError for " + field_path);
}

GamepadProfile LoadVariantProfile(const std::filesystem::path& fixture,
                                  const std::filesystem::path& scratch) {
  auto yaml = ReadText(fixture);
  yaml = ReplaceOnce(yaml, "right_y: {axis: 3, center: 0, min: -32767, max: 32767, invert: true, deadzone: 0.05}",
                     "right_y: {unsupported: true}");
  yaml = ReplaceOnce(yaml, "lt: {source: axis, index: 5, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: null}",
                     "lt: {source: axis, index: 5, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: {index: 2, observed_within_ms: 80}}");
  yaml = ReplaceOnce(yaml, "rt: {source: axis, index: 4, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: null}",
                     "rt: {source: button, index: 2, threshold: 0.5}");
  yaml = ReplaceOnce(yaml, "x: {source: button, index: 3}",
                     "x: {unsupported: true}");
  yaml = ReplaceOnce(yaml, "left: {source: axis, index: 6, direction: negative, threshold: 0.5}",
                     "left: {source: button, index: 5}");
  yaml = ReplaceOnce(yaml, "right: {source: axis, index: 6, direction: positive, threshold: 0.5}",
                     "right: {unsupported: true}");
  const auto path = scratch / "variant.yaml";
  WriteText(path, yaml);
  return gamepad::LoadGamepadProfile(path);
}

void TestProfileVariantsAndStrictSchema(const std::filesystem::path& fixture,
                                        const std::filesystem::path& scratch) {
  const auto profile = LoadVariantProfile(fixture, scratch);
  Expect(profile.schema_version == 1, "schema version must round-trip");
  Expect(profile.device.vendor_id == "20bc" && profile.device.product_id == "5159",
         "four-character hexadecimal identity must round-trip");
  Expect(std::holds_alternative<UnsupportedBinding>(profile.sticks.at("right_y")),
         "unsupported stick must round-trip");
  const auto& lt = std::get<AxisTriggerBinding>(profile.triggers.at("lt"));
  Expect(lt.correlated_button.has_value() && lt.correlated_button->index == 2 &&
             lt.correlated_button->observed_within_ms == 80,
         "axis trigger correlation must round-trip");
  Expect(std::holds_alternative<ButtonTriggerBinding>(profile.triggers.at("rt")),
         "button trigger must round-trip");
  Expect(std::holds_alternative<UnsupportedBinding>(profile.buttons.at("x")),
         "unsupported button must round-trip");
  Expect(std::holds_alternative<ButtonBinding>(profile.dpad.at("left")) &&
             std::holds_alternative<UnsupportedBinding>(profile.dpad.at("right")),
         "button and unsupported D-pad variants must round-trip");

  const auto original = ReadText(fixture);
  const auto optional_correlation = scratch / "optional_correlation.yaml";
  WriteText(optional_correlation,
            ReplaceOnce(original, ", correlated_button: null}", "}"));
  const auto no_correlation = gamepad::LoadGamepadProfile(optional_correlation);
  Expect(!std::get<AxisTriggerBinding>(no_correlation.triggers.at("lt"))
              .correlated_button.has_value(),
         "omitted trigger correlation must default to none");

  const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> invalid = {
      {"schema", {"schema_version: 1", "schema_version: 2"}},
      {"unknown", {"schema_version: 1", "schema_version: 1\nunknown_root: true"}},
      {"id", {"vendor_id: 20bc", "vendor_id: ABCD"}},
      {"empty_name", {"name: BEITONG BTP-KP20D", "name: '   '"}},
      {"negative_index", {"left_x: {axis: 0", "left_x: {axis: -1"}},
      {"range", {"center: 0, min: -32767", "center: -32767, min: -32767"}},
      {"nan", {"deadzone: 0.05}", "deadzone: .nan}"}},
      {"threshold", {"direction: negative, threshold: 0.5}", "direction: negative, threshold: 0.0}"}},
      {"equal_trigger", {"released: -32767, pressed: 32767", "released: 32767, pressed: 32767"}},
      {"numeric_string", {"axis: 0", "axis: '0'"}},
      {"numeric_id", {"product_id: '5159'", "product_id: 5159"}},
      {"canonical_string_index", {"left_x: {axis: 0", "left_x: {axis: !!str 0"}},
      {"canonical_string_bool", {"invert: false", "invert: !!str false"}},
      {"canonical_float_index", {"left_x: {axis: 0", "left_x: {axis: !!float 0"}},
      {"numeric_name", {"name: BEITONG BTP-KP20D", "name: 123"}},
      {"boolean_serial", {"serial: null", "serial: false"}},
      {"rounded_threshold", {"direction: negative, threshold: 0.5}",
                              "direction: negative, threshold: 1.00000001}"}},
      {"unknown_nested", {"axis: 0, center:", "mystery: 4, axis: 0, center:"}},
  };
  const std::vector<std::string> paths = {
      "schema_version", "unknown_root", "device.vendor_id", "device.name",
      "sticks.left_x.axis", "sticks.left_x", "sticks.left_x.deadzone",
      "dpad.up.threshold", "triggers.lt", "sticks.left_x.axis",
      "device.product_id", "sticks.left_x.axis", "sticks.left_x.invert",
      "sticks.left_x.axis", "device.name", "device.serial",
      "dpad.up.threshold", "sticks.left_x.mystery",
  };
  for (std::size_t i = 0; i < invalid.size(); ++i) {
    const auto path = scratch / (invalid[i].first + ".yaml");
    WriteText(path, ReplaceOnce(original, invalid[i].second.first, invalid[i].second.second));
    ExpectConfigError([&path]() { (void)gamepad::LoadGamepadProfile(path); }, paths[i]);
  }

  const auto tagged_product = scratch / "tagged_product.yaml";
  WriteText(tagged_product,
            ReplaceOnce(original, "product_id: '5159'", "product_id: !!str 5159"));
  Expect(gamepad::LoadGamepadProfile(tagged_product).device.product_id == "5159",
         "canonical YAML string tag must preserve a legal hexadecimal ID");

  const auto near_one_deadzone = scratch / "near_one_deadzone.yaml";
  WriteText(near_one_deadzone,
            ReplaceOnce(original, "deadzone: 0.05}", "deadzone: 0.999999999}"));
  const auto near_one = gamepad::LoadGamepadProfile(near_one_deadzone);
  Expect(std::get<StickBinding>(near_one.sticks.at("left_x")).deadzone < 1.0F,
         "legal deadzone near one must remain representable below one");
}

void TestPyYamlScalarContract(const std::filesystem::path& fixture,
                              const std::filesystem::path& scratch) {
  const std::string original = ReadText(fixture);
  struct IntegerCase {
    const char* name;
    const char* scalar;
    std::size_t expected;
  };
  const std::vector<IntegerCase> integers = {
      {"octal", "010", 8},
      {"hex", "0x08", 8},
      {"binary", "0b1000", 8},
      {"underscore", "8_0", 80},
      {"sexagesimal", "1:20", 80},
  };
  for (const auto& item : integers) {
    const auto path = scratch / ("scalar_integer_" + std::string(item.name) +
                                 ".yaml");
    WriteText(path, ReplaceOnce(original, "left_x: {axis: 0",
                                "left_x: {axis: " +
                                    std::string(item.scalar)));
    const auto profile = gamepad::LoadGamepadProfile(path);
    const auto axis =
        std::get<StickBinding>(profile.sticks.at("left_x")).axis;
    Expect(axis == item.expected,
           std::string(item.name) + " integer must resolve to " +
               std::to_string(item.expected) + ", got " +
               std::to_string(axis));
  }

  struct BoolCase {
    const char* name;
    const char* scalar;
  };
  for (const auto& item :
       std::vector<BoolCase>{{"yes", "yes"}, {"on", "ON"}}) {
    const auto path =
        scratch / ("scalar_bool_" + std::string(item.name) + ".yaml");
    WriteText(path, ReplaceOnce(original, "invert: false",
                                "invert: " + std::string(item.scalar)));
    const auto profile = gamepad::LoadGamepadProfile(path);
    Expect(std::get<StickBinding>(profile.sticks.at("left_x")).invert,
           std::string(item.scalar) + " must resolve to true");
  }

  const auto null_path = scratch / "scalar_canonical_null.yaml";
  WriteText(null_path,
            ReplaceOnce(original, "serial: null", "serial: !!null null"));
  Expect(!gamepad::LoadGamepadProfile(null_path).device.serial,
         "canonical null must be accepted for optional serial");

  const auto tagged_string = scratch / "scalar_canonical_string.yaml";
  WriteText(tagged_string,
            ReplaceOnce(original, "product_id: '5159'",
                        "product_id: !!str 5159"));
  const auto tagged_profile = gamepad::LoadGamepadProfile(tagged_string);
  Expect(tagged_profile.device.vendor_id == "20bc" &&
             tagged_profile.device.product_id == "5159",
         "plain 20bc and canonical string 5159 must remain strings");

  const auto tagged_decimal_float = scratch / "scalar_canonical_float.yaml";
  WriteText(tagged_decimal_float,
            ReplaceOnce(original, "deadzone: 0.05}",
                        "deadzone: !!float 0e0}"));
  const auto decimal_profile =
      gamepad::LoadGamepadProfile(tagged_decimal_float);
  Expect(std::get<StickBinding>(decimal_profile.sticks.at("left_x"))
                 .deadzone == 0.0F,
         "canonical decimal exponent must remain a legal float");

  struct CanonicalFloatCase {
    const char* name;
    const char* scalar;
    float expected;
    bool expect_negative_zero;
  };
  const std::vector<CanonicalFloatCase> canonical_floats = {
      {"signed_underscore", "-0_0.0e+0", -0.0F, true},
      {"sexagesimal", "+0:00.5", 0.5F, false},
      {"underflow", "1e-9999", 0.0F, false},
      {"signed_underflow", "-1e-9999", -0.0F, true},
      {"signed_sexagesimal_component", "0:+0.5", 0.5F, false},
      {"whitespace_sexagesimal_components", "' +0 : +0.5 '", 0.5F, false},
      {"decimal_exponent_underscore", "+1_0e-2", 0.1F, false},
      {"quoted_whitespace", "' 0.5 '", 0.5F, false},
  };
  for (const auto& item : canonical_floats) {
    const auto path =
        scratch / ("scalar_canonical_float_" + std::string(item.name) + ".yaml");
    WriteText(path, ReplaceOnce(original, "deadzone: 0.05}",
                                "deadzone: !!float " +
                                    std::string(item.scalar) + "}"));
    const auto profile = gamepad::LoadGamepadProfile(path);
    const float actual =
        std::get<StickBinding>(profile.sticks.at("left_x")).deadzone;
    ExpectNear(actual, item.expected, 1.0e-6F,
               std::string(item.scalar) + " must remain a legal float");
    if (item.expected == 0.0F) {
      Expect(std::signbit(actual) == item.expect_negative_zero,
             std::string(item.scalar) +
                 " must preserve the expected zero sign");
    }
  }

  struct RejectedCase {
    const char* name;
    const char* from;
    const char* to;
    const char* field;
  };
  const std::vector<RejectedCase> rejected = {
      {"canonical_nan_payload", "deadzone: 0.05}",
       "deadzone: !!float nan(payload)}", "sticks.left_x.deadzone"},
      {"canonical_hex_zero", "deadzone: 0.05}",
       "deadzone: !!float 0x0}", "sticks.left_x.deadzone"},
      {"canonical_hex_exponent", "deadzone: 0.05}",
       "deadzone: !!float 0x1p-4}", "sticks.left_x.deadzone"},
      {"canonical_overflow", "deadzone: 0.05}",
       "deadzone: !!float 1e9999}", "sticks.left_x.deadzone"},
      {"canonical_infinity", "deadzone: 0.05}",
       "deadzone: !!float .inf}", "sticks.left_x.deadzone"},
      {"canonical_nan", "deadzone: 0.05}",
       "deadzone: !!float .nan}", "sticks.left_x.deadzone"},
      {"unresolved_exponent", "deadzone: 0.05}", "deadzone: 0e0}",
       "sticks.left_x.deadzone"},
      {"mixed_bool_case", "invert: false", "invert: yEs",
       "sticks.left_x.invert"},
      {"timestamp_name", "name: BEITONG BTP-KP20D", "name: 2026-08-10",
       "device.name"},
      {"binary_serial", "serial: null", "serial: !!binary SGVsbG8=",
       "device.serial"},
      {"integer_overflow", "left_x: {axis: 0",
       "left_x: {axis: 9223372036854775808", "sticks.left_x.axis"},
      {"sexagesimal_overflow", "left_x: {axis: 0",
       "left_x: {axis: 256204778801521550:08", "sticks.left_x.axis"},
  };
  for (const auto& item : rejected) {
    const auto path =
        scratch / ("scalar_rejected_" + std::string(item.name) + ".yaml");
    WriteText(path, ReplaceOnce(original, item.from, item.to));
    ExpectConfigError(
        [&path]() { (void)gamepad::LoadGamepadProfile(path); }, item.field);
  }
}

void TestCrossFieldValidation(const std::filesystem::path& fixture,
                              const std::filesystem::path& scratch) {
  const auto original = ReadText(fixture);
  const auto unsupported_required = scratch / "unsupported_required.yaml";
  WriteText(unsupported_required,
            ReplaceOnce(original,
                        "left_x: {axis: 0, center: 0, min: -32767, max: 32767, invert: false, deadzone: 0.05}",
                        "left_x: {unsupported: true}"));
  ExpectConfigError(
      [&unsupported_required]() { (void)gamepad::LoadGamepadProfile(unsupported_required); },
      "sticks.left_x");

  const auto unsupported_trigger = scratch / "unsupported_trigger.yaml";
  WriteText(unsupported_trigger,
            ReplaceOnce(original,
                        "lt: {source: axis, index: 5, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: null}",
                        "lt: {unsupported: true}"));
  ExpectConfigError(
      [&unsupported_trigger]() { (void)gamepad::LoadGamepadProfile(unsupported_trigger); },
      "triggers.lt");

  const auto missing = scratch / "missing.yaml";
  WriteText(missing, ReplaceOnce(original, "  right_y: {axis: 3, center: 0, min: -32767, max: 32767, invert: true, deadzone: 0.05}\n", ""));
  ExpectConfigError([&missing]() { (void)gamepad::LoadGamepadProfile(missing); },
                    "sticks.right_y");

  const auto duplicate = scratch / "duplicate.yaml";
  WriteText(duplicate, ReplaceOnce(original, "b: {source: button, index: 1}",
                                   "b: {source: button, index: 0}"));
  ExpectConfigError([&duplicate]() { (void)gamepad::LoadGamepadProfile(duplicate); },
                    "buttons.b");

  auto profile = gamepad::LoadGamepadProfile(fixture);
  profile.dpad.at("down") = UnsupportedBinding{};
  DeviceDescriptor too_small{scratch / "js7", profile.device, 7, 12, std::nullopt};
  ExpectConfigError([&profile, &too_small]() { gamepad::ValidateProfile(profile, too_small); },
                    "dpad.up.index");

  const auto complete = gamepad::LoadGamepadProfile(fixture);
  DeviceDescriptor no_axes{scratch / "js8", complete.device, 0, 12, std::nullopt};
  ExpectConfigError([&complete, &no_axes]() { gamepad::ValidateProfile(complete, no_axes); },
                    "sticks.left_x.axis");
}

void TestActiveSelection(const std::filesystem::path& fixture,
                         const std::filesystem::path& scratch) {
  const auto active_root = scratch / "active_cases";
  const auto profile_path = active_root / "profiles/beitong.yaml";
  WriteText(profile_path, ReadText(fixture));
  const std::string identity =
      "device:\n  vendor_id: 20bc\n  product_id: '5159'\n"
      "  name: BEITONG BTP-KP20D\n  serial: null\n";
  const auto active = active_root / "active.yaml";
  WriteText(active, "schema_version: 1\nprofile: profiles/beitong.yaml\n" + identity);
  const ActiveSelection selection = gamepad::LoadActiveSelection(active);
  Expect(selection.profile == std::filesystem::path("profiles/beitong.yaml"),
         "active selection must preserve normalized relative profile path");

  const std::vector<std::pair<std::string, std::string>> bad_paths = {
      {"/tmp/profile.yaml", "profile"},
      {"profiles/../outside.yaml", "profile"},
      {"profiles//beitong.yaml", "profile"},
      {"other/beitong.yaml", "profile"},
      {"profiles/beitong.yml", "profile"},
  };
  for (std::size_t index = 0; index < bad_paths.size(); ++index) {
    const auto bad = active_root / ("bad_active_" + std::to_string(index) + ".yaml");
    WriteText(bad, "schema_version: 1\nprofile: " + bad_paths[index].first + "\n" + identity);
    ExpectConfigError([&bad]() { (void)gamepad::LoadActiveSelection(bad); },
                      bad_paths[index].second);
  }

  const auto mismatch = active_root / "mismatch.yaml";
  WriteText(mismatch,
            "schema_version: 1\nprofile: profiles/beitong.yaml\ndevice:\n"
            "  vendor_id: 20bc\n  product_id: '5159'\n  name: Different Pad\n  serial: null\n");
  ExpectConfigError([&mismatch]() { (void)gamepad::LoadActiveSelection(mismatch); },
                    "device");
}

gamepad::DiscoveryRoots MakeDiscoveryRoots(const std::filesystem::path& base,
                                            bool create_by_id = true) {
  gamepad::DiscoveryRoots roots{base / "dev", base / "sys", base / "by-id"};
  std::filesystem::create_directories(roots.dev_input);
  std::filesystem::create_directories(roots.sys_class_input);
  if (create_by_id) {
    std::filesystem::create_directories(roots.by_id);
  }
  return roots;
}

void WriteDiscoveryIdentity(const gamepad::DiscoveryRoots& roots,
                            const std::string& js_name,
                            const std::string& vendor = "20bc",
                            const std::string& product = "5159") {
  const auto device = roots.sys_class_input / js_name / "device";
  WriteText(device / "id/vendor", vendor + "\n");
  WriteText(device / "id/product", product + "\n");
  WriteText(device / "name", "Test Pad\n");
}

void TestDiscoveryFailsClosed(const std::filesystem::path& scratch) {
  {
    const auto roots = MakeDiscoveryRoots(scratch / "discovery_invalid_id");
    WriteText(roots.dev_input / "js0", "");
    WriteDiscoveryIdentity(roots, "js0", "xyz", "5159");
    ExpectConfigError(
        [&roots]() { (void)gamepad::EnumerateLinuxJoysticks(roots); },
        "device.vendor_id");
  }

  {
    const auto roots = MakeDiscoveryRoots(scratch / "discovery_long_js");
    WriteText(roots.dev_input / ("js" + std::string(128, '9')), "");
    ExpectConfigError(
        [&roots]() { (void)gamepad::EnumerateLinuxJoysticks(roots); },
        "dev_input");
  }

  {
    const auto roots = MakeDiscoveryRoots(scratch / "discovery_bad_serial");
    WriteText(roots.dev_input / "js0", "");
    WriteDiscoveryIdentity(roots, "js0");
    const auto uniq = roots.sys_class_input / "js0/device/uniq";
    std::filesystem::create_symlink("uniq", uniq);
    ExpectConfigError(
        [&roots]() { (void)gamepad::EnumerateLinuxJoysticks(roots); },
        "device.serial");
  }

  {
    const auto roots =
        MakeDiscoveryRoots(scratch / "discovery_bad_by_id", false);
    WriteText(roots.dev_input / "js0", "");
    WriteDiscoveryIdentity(roots, "js0");
    std::filesystem::create_symlink("by-id", roots.by_id);
    ExpectConfigError(
        [&roots]() { (void)gamepad::EnumerateLinuxJoysticks(roots); },
        "by_id");
  }

  {
    const auto base = scratch / "discovery_rebind";
    const auto roots = MakeDiscoveryRoots(base);
    const auto backing_a = base / "backing-a";
    const auto backing_b = base / "backing-b";
    const auto js0 = roots.dev_input / "js0";
    WriteText(backing_a, "a");
    WriteText(backing_b, "b");
    std::filesystem::create_symlink(backing_a, js0);
    const auto device = roots.sys_class_input / "js0/device";
    WriteText(device / "id/vendor", "20bc\n");
    WriteText(device / "id/product", "5159\n");
    std::filesystem::create_directories(device);
    const auto name_fifo = device / "name";
    Expect(::mkfifo(name_fifo.c_str(), 0600) == 0,
           "must create deterministic sysfs-name FIFO");

    const pid_t child = ::fork();
    Expect(child >= 0, "must fork deterministic rebinding helper");
    if (child == 0) {
      const int writer = ::open(name_fifo.c_str(), O_WRONLY);
      if (writer < 0 || ::unlink(js0.c_str()) != 0 ||
          ::symlink(backing_b.c_str(), js0.c_str()) != 0) {
        _exit(2);
      }
      constexpr char kName[] = "Test Pad\n";
      const ssize_t written = ::write(writer, kName, sizeof(kName) - 1U);
      ::close(writer);
      _exit(written == static_cast<ssize_t>(sizeof(kName) - 1U) ? 0 : 3);
    }

    std::string diagnostic;
    try {
      (void)gamepad::EnumerateLinuxJoysticks(roots);
    } catch (const GamepadConfigError& error) {
      diagnostic = error.what();
    }
    int status = 0;
    Expect(::waitpid(child, &status, 0) == child,
           "must reap deterministic rebinding helper");
    Expect(WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "rebinding helper must complete successfully");
    Expect(diagnostic.rfind("device.path:", 0) == 0,
           "jsN rebinding must fail as one inconsistent snapshot: " + diagnostic);
  }
}
DeviceDescriptor Descriptor(std::string path, std::string vendor, std::string product,
                            std::string name, std::optional<std::string> serial,
                            std::optional<std::string> by_id = std::nullopt) {
  return DeviceDescriptor{
      std::move(path),
      DeviceIdentity{std::move(vendor), std::move(product), std::move(name), std::move(serial)},
      8,
      12,
      by_id ? std::optional<std::filesystem::path>(*by_id) : std::nullopt,
  };
}

void TestIdentitySelection() {
  const DeviceIdentity wanted{" 20bc ", "5159\t", " BEITONG BTP-KP20D ", std::nullopt};
  std::vector<DeviceDescriptor> devices = {
      Descriptor("/dev/input/js9", "20bc", "5159", "BEITONG BTP-KP20D", std::nullopt,
                 "/dev/input/by-id/usb-pad"),
      Descriptor("/dev/input/js0", "20bc", "5159", "Other Pad", std::nullopt),
  };
  const auto selected = gamepad::SelectUniqueDevice(wanted, devices);
  Expect(selected.path == std::filesystem::path("/dev/input/js9"),
         "jsN and by-id paths must not participate in identity matching");

  const DeviceIdentity serial_wanted{"20bc", "5159", "BEITONG BTP-KP20D", "serial-2"};
  devices.push_back(Descriptor("/dev/input/js4", "20bc", "5159", "BEITONG BTP-KP20D",
                               "serial-2"));
  Expect(gamepad::SelectUniqueDevice(serial_wanted, devices).path ==
             std::filesystem::path("/dev/input/js4"),
         "selected serial must match exactly when present");

  const auto no_match = [&]() {
    (void)gamepad::SelectUniqueDevice(
        DeviceIdentity{"ffff", "ffff", "Missing", std::nullopt}, devices);
  };
  try {
    no_match();
    throw std::runtime_error("zero identity matches must fail");
  } catch (const GamepadConfigError& error) {
    const std::string message = error.what();
    Expect(message.find("js9") != std::string::npos &&
               message.find("js0") != std::string::npos &&
               message.find("js4") != std::string::npos,
           "zero-match diagnostic must list every candidate");
  }

  devices.push_back(Descriptor("/dev/input/js12", "20bc", "5159", " BEITONG BTP-KP20D ",
                               std::nullopt));
  try {
    (void)gamepad::SelectUniqueDevice(wanted, devices);
    throw std::runtime_error("multiple identity matches must fail");
  } catch (const GamepadConfigError& error) {
    const std::string message = error.what();
    Expect(message.find("js9") != std::string::npos && message.find("js12") != std::string::npos,
           "multiple-match diagnostic must list every candidate");
  }
}

void TestMapper(const std::filesystem::path& fixture,
                const std::filesystem::path& scratch) {
  const auto profile = gamepad::LoadGamepadProfile(fixture);
  RawSnapshot neutral{{0, 0, 0, 0, -32767, -32767, 0, 0},
                      std::vector<std::uint8_t>(12, 0)};

  struct DpadAxisCase {
    std::int16_t raw_axis;
    bool expected_up;
    bool expected_down;
  };
  for (const auto& item : std::vector<DpadAxisCase>{
           {-1, false, false},
           {1, false, false},
           {-16383, false, false},
           {16383, false, false},
           {-16384, true, false},
           {16384, false, true},
           {-32768, true, false},
           {32767, false, true},
       }) {
    auto raw = neutral;
    raw.axes.at(7) = item.raw_axis;
    const auto mapped = gamepad::MapLogicalSnapshot(profile, raw);
    Expect(mapped.up == item.expected_up && mapped.down == item.expected_down,
           "D-pad threshold must compare a normalized signed axis for raw " +
               std::to_string(item.raw_axis));
  }

  RawSnapshot lt_up = neutral;
  lt_up.axes.at(5) = 32767;
  lt_up.axes.at(7) = -32767;
  const auto first = gamepad::MapLogicalSnapshot(profile, lt_up);
  Expect(first.lt > 0.5F && first.up, "LT + Up must be representable");

  RawSnapshot rt_a = neutral;
  rt_a.axes.at(4) = 32767;
  rt_a.buttons.at(0) = 1;
  const auto second = gamepad::MapLogicalSnapshot(profile, rt_a);
  Expect(second.rt > 0.5F && second.a, "RT + A must be representable");
  Expect(second.lt == 0.0F && !second.up,
         "pure mapper must not retain the preceding snapshot");

  auto custom_yaml = ReadText(fixture);
  custom_yaml = ReplaceOnce(custom_yaml,
      "left_x: {axis: 0, center: 0, min: -32767, max: 32767, invert: false, deadzone: 0.05}",
      "left_x: {axis: 0, center: 1000, min: -3000, max: 9000, invert: false, deadzone: 0.1}");
  custom_yaml = ReplaceOnce(custom_yaml,
      "left_y: {axis: 1, center: 0, min: -32767, max: 32767, invert: true, deadzone: 0.05}",
      "left_y: {axis: 1, center: 0, min: -1000, max: 3000, invert: true, deadzone: 0.1}");
  custom_yaml = ReplaceOnce(custom_yaml,
      "lt: {source: axis, index: 5, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: null}",
      "lt: {source: axis, index: 5, released: 32767, pressed: -32767, threshold: 0.5, correlated_button: null}");
  custom_yaml = ReplaceOnce(custom_yaml,
      "rt: {source: axis, index: 4, released: -32767, pressed: 32767, threshold: 0.5, correlated_button: null}",
      "rt: {source: button, index: 2, threshold: 0.5}");
  custom_yaml = ReplaceOnce(custom_yaml,
      "right_y: {axis: 3, center: 0, min: -32767, max: 32767, invert: true, deadzone: 0.05}",
      "right_y: {unsupported: true}");
  custom_yaml = ReplaceOnce(custom_yaml, "x: {source: button, index: 3}",
                            "x: {unsupported: true}");
  custom_yaml = ReplaceOnce(custom_yaml,
      "left: {source: axis, index: 6, direction: negative, threshold: 0.5}",
      "left: {source: button, index: 5}");
  custom_yaml = ReplaceOnce(custom_yaml,
      "right: {source: axis, index: 6, direction: positive, threshold: 0.5}",
      "right: {unsupported: true}");
  const auto custom_path = scratch / "mapper.yaml";
  WriteText(custom_path, custom_yaml);
  const auto custom = gamepad::LoadGamepadProfile(custom_path);

  RawSnapshot raw{{-1000, 1500, 32767, 30000, -32767, -32767, 0, -16384},
                  std::vector<std::uint8_t>(12, 0)};
  raw.buttons.at(2) = 1;
  raw.buttons.at(5) = 1;
  auto mapped = gamepad::MapLogicalSnapshot(custom, raw);
  ExpectNear(mapped.lx, -0.5F, 0.0001F, "asymmetric negative stick scaling");
  ExpectNear(mapped.ly, -0.5F, 0.0001F, "stick inversion after asymmetric scaling");
  Expect(mapped.lt == 1.0F && mapped.rt == 1.0F,
         "reversed axis and digital triggers must normalize");
  Expect(mapped.left && !mapped.right && mapped.up,
         "button and axis D-pad bindings must map at threshold");
  Expect(mapped.ry == 0.0F && !mapped.x,
         "unsupported outputs must remain neutral");

  raw.axes.at(0) = 1200;
  raw.axes.at(1) = 5000;
  mapped = gamepad::MapLogicalSnapshot(custom, raw);
  Expect(mapped.lx == 0.0F, "deadzone is applied after scaling");
  Expect(mapped.ly == -1.0F, "stick output is clamped before inversion/deadzone");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error("usage: gamepad_profile_mapper_test FIXTURE SCRATCH");
    }
    const std::filesystem::path fixture(argv[1]);
    const std::filesystem::path scratch(argv[2]);
    TestProfileVariantsAndStrictSchema(fixture, scratch);
    TestPyYamlScalarContract(fixture, scratch);
    TestCrossFieldValidation(fixture, scratch);
    TestActiveSelection(fixture, scratch);
    TestDiscoveryFailsClosed(scratch);
    TestIdentitySelection();
    TestMapper(fixture, scratch);
    std::cout << "all C++ gamepad profile/mapper tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
