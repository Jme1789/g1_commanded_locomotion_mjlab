#include "gamepad/gamepad_profile.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale.h>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace gamepad {
namespace {

using StickVariant = std::variant<StickBinding, UnsupportedBinding>;
using ButtonVariant = std::variant<ButtonBinding, UnsupportedBinding>;

[[noreturn]] void Fail(const std::string& path, const std::string& message) {
  throw GamepadConfigError(path + ": " + message);
}

void RequireMap(const YAML::Node& node, const std::string& path) {
  if (!node || !node.IsMap()) {
    Fail(path, "expected mapping");
  }
}

void CheckKeys(const YAML::Node& node, const std::string& path,
               std::initializer_list<std::string_view> allowed) {
  RequireMap(node, path);
  std::set<std::string> permitted;
  for (const auto key : allowed) {
    permitted.emplace(key);
  }
  std::set<std::string> seen;
  for (const auto& entry : node) {
    if (!entry.first.IsScalar()) {
      Fail(path, "mapping key must be a string");
    }
    const std::string key = entry.first.Scalar();
    const std::string field = path.empty() ? key : path + "." + key;
    if (permitted.count(key) == 0U) {
      Fail(field, "unknown key");
    }
    if (!seen.insert(key).second) {
      Fail(field, "duplicate key");
    }
  }
}

YAML::Node Required(const YAML::Node& node, const std::string& key,
                    const std::string& path) {
  const YAML::Node value = node[key];
  if (!value) {
    Fail(path.empty() ? key : path + "." + key, "field is required");
  }
  return value;
}

enum class ScalarKind {
  kString,
  kInteger,
  kFloat,
  kBoolean,
  kNull,
  kTimestamp,
  kUnknown,
};

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) { return std::tolower(character); });
  return value;
}

bool IsYamlBoolean(const std::string& value) {
  static const std::regex pattern(
      R"((yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF))");
  return std::regex_match(value, pattern);
}

bool IsYamlInteger(const std::string& value) {
  static const std::regex pattern(
      R"([-+]?(0b[0-1_]+|0[0-7_]+|0|[1-9][0-9_]*|0x[0-9a-fA-F_]+|[1-9][0-9_]*(:[0-5]?[0-9])+))");
  return std::regex_match(value, pattern);
}

bool IsYamlFloat(const std::string& value) {
  static const std::regex pattern(
      R"(([-+]?([0-9][0-9_]*\.[0-9_]*([eE][-+][0-9]+)?|\.[0-9][0-9_]*([eE][-+][0-9]+)?|[0-9][0-9_]*(:[0-5]?[0-9])+\.[0-9_]*|\.(inf|Inf|INF))|\.(nan|NaN|NAN)))");
  return std::regex_match(value, pattern);
}

bool IsYamlNull(const std::string& value) {
  return value.empty() || value == "~" || value == "null" || value == "Null" ||
         value == "NULL";
}

bool IsYamlTimestamp(const std::string& value) {
  static const std::regex pattern(
      R"(([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}([Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(\.[0-9]*)?([ \t]*(Z|[-+][0-9]{1,2}(:[0-9]{2})?))?))");
  return std::regex_match(value, pattern);
}

ScalarKind ClassifyScalar(const YAML::Node& node) {
  if (!node) {
    return ScalarKind::kUnknown;
  }
  if (node.IsNull()) {
    return ScalarKind::kNull;
  }
  if (!node.IsScalar()) {
    return ScalarKind::kUnknown;
  }
  const std::string tag = node.Tag();
  if (tag == "!" || tag == "tag:yaml.org,2002:str") {
    return ScalarKind::kString;
  }
  if (tag == "tag:yaml.org,2002:int") {
    return ScalarKind::kInteger;
  }
  if (tag == "tag:yaml.org,2002:float") {
    return ScalarKind::kFloat;
  }
  if (tag == "tag:yaml.org,2002:bool") {
    return ScalarKind::kBoolean;
  }
  if (tag == "tag:yaml.org,2002:null") {
    return ScalarKind::kNull;
  }
  if (tag == "tag:yaml.org,2002:timestamp") {
    return ScalarKind::kTimestamp;
  }
  if (!tag.empty() && tag != "?") {
    return ScalarKind::kUnknown;
  }

  const std::string value = node.Scalar();
  if (IsYamlBoolean(value)) {
    return ScalarKind::kBoolean;
  }
  if (IsYamlFloat(value)) {
    return ScalarKind::kFloat;
  }
  if (IsYamlInteger(value)) {
    return ScalarKind::kInteger;
  }
  if (IsYamlNull(value)) {
    return ScalarKind::kNull;
  }
  if (IsYamlTimestamp(value)) {
    return ScalarKind::kTimestamp;
  }
  return ScalarKind::kString;
}

std::string ParseString(const YAML::Node& node, const std::string& path) {
  if (ClassifyScalar(node) != ScalarKind::kString) {
    Fail(path, "expected string");
  }
  return node.Scalar();
}

std::string WithoutUnderscores(const std::string& value) {
  std::string compact;
  compact.reserve(value.size());
  std::copy_if(value.begin(), value.end(), std::back_inserter(compact),
               [](char character) { return character != '_'; });
  return compact;
}

unsigned int DigitValue(char character) {
  if (character >= '0' && character <= '9') {
    return static_cast<unsigned int>(character - '0');
  }
  if (character >= 'a' && character <= 'f') {
    return 10U + static_cast<unsigned int>(character - 'a');
  }
  if (character >= 'A' && character <= 'F') {
    return 10U + static_cast<unsigned int>(character - 'A');
  }
  return 16U;
}

unsigned long long ParseMagnitude(const std::string& digits, unsigned int base,
                                  unsigned long long limit,
                                  const std::string& path) {
  if (digits.empty()) {
    Fail(path, "expected integer");
  }
  unsigned long long result = 0;
  for (const char character : digits) {
    const unsigned int digit = DigitValue(character);
    if (digit >= base) {
      Fail(path, "expected integer");
    }
    if (result > (limit - digit) / base) {
      Fail(path, "integer is out of range");
    }
    result = result * base + digit;
  }
  return result;
}

long long ParseInteger(const YAML::Node& node, const std::string& path) {
  if (ClassifyScalar(node) != ScalarKind::kInteger) {
    Fail(path, "expected integer");
  }
  std::string value = WithoutUnderscores(node.Scalar());
  bool negative = false;
  if (!value.empty() && (value.front() == '-' || value.front() == '+')) {
    negative = value.front() == '-';
    value.erase(value.begin());
  }
  if (value.empty()) {
    Fail(path, "expected integer");
  }
  const unsigned long long negative_limit =
      static_cast<unsigned long long>(std::numeric_limits<long long>::max()) + 1U;
  const unsigned long long limit =
      negative ? negative_limit
               : static_cast<unsigned long long>(
                     std::numeric_limits<long long>::max());

  unsigned long long magnitude = 0;
  if (value == "0") {
    magnitude = 0;
  } else if (value.rfind("0b", 0U) == 0U) {
    magnitude = ParseMagnitude(value.substr(2), 2U, limit, path);
  } else if (value.rfind("0x", 0U) == 0U) {
    magnitude = ParseMagnitude(value.substr(2), 16U, limit, path);
  } else if (value.front() == '0') {
    magnitude = ParseMagnitude(value, 8U, limit, path);
  } else if (value.find(':') != std::string::npos) {
    std::size_t begin = 0;
    while (begin < value.size()) {
      const std::size_t end = value.find(':', begin);
      const std::string digits = value.substr(begin, end - begin);
      const unsigned long long part =
          ParseMagnitude(digits, 10U, limit, path);
      if (magnitude > (limit - part) / 60U) {
        Fail(path, "integer is out of range");
      }
      magnitude = magnitude * 60U + part;
      if (end == std::string::npos) {
        break;
      }
      begin = end + 1U;
    }
  } else {
    magnitude = ParseMagnitude(value, 10U, limit, path);
  }
  if (!negative) {
    return static_cast<long long>(magnitude);
  }
  if (magnitude == negative_limit) {
    return std::numeric_limits<long long>::min();
  }
  return -static_cast<long long>(magnitude);
}

std::size_t ParseIndex(const YAML::Node& node, const std::string& path) {
  const long long value = ParseInteger(node, path);
  if (value < 0) {
    Fail(path, "index must be non-negative");
  }
  return static_cast<std::size_t>(value);
}

int ParseNonNegativeInt(const YAML::Node& node, const std::string& path) {
  const long long value = ParseInteger(node, path);
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    Fail(path, "must be a non-negative integer");
  }
  return static_cast<int>(value);
}

std::int16_t ParseInt16(const YAML::Node& node, const std::string& path) {
  const long long value = ParseInteger(node, path);
  if (value < std::numeric_limits<std::int16_t>::min() ||
      value > std::numeric_limits<std::int16_t>::max()) {
    Fail(path, "must fit signed 16-bit joystick range");
  }
  return static_cast<std::int16_t>(value);
}

bool IsPythonFloatWhitespace(char character) {
  return character == ' ' || character == '\t' || character == '\n' ||
         character == '\r' || character == '\f' || character == '\v';
}

std::string TrimPythonFloatWhitespace(std::string value) {
  const auto first =
      std::find_if_not(value.begin(), value.end(), IsPythonFloatWhitespace);
  const auto last =
      std::find_if_not(value.rbegin(), value.rend(), IsPythonFloatWhitespace)
          .base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

class NumericCLocale final {
 public:
  NumericCLocale() : locale_(newlocale(LC_NUMERIC_MASK, "C", nullptr)) {}
  ~NumericCLocale() {
    if (locale_ != nullptr) {
      freelocale(locale_);
    }
  }

  NumericCLocale(const NumericCLocale&) = delete;
  NumericCLocale& operator=(const NumericCLocale&) = delete;

  locale_t get() const { return locale_; }

 private:
  locale_t locale_;
};

double ParsePythonFloat(const std::string& value, const std::string& path) {
  const std::string trimmed = TrimPythonFloatWhitespace(value);
  static const std::regex pattern(
      R"([-+]?(([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?|inf(inity)?|nan))");
  if (!std::regex_match(trimmed, pattern)) {
    Fail(path, "expected number");
  }

  static const NumericCLocale numeric_locale;
  if (numeric_locale.get() == nullptr) {
    Fail(path, "numeric parser is unavailable");
  }
  char* end = nullptr;
  const double parsed = strtod_l(trimmed.c_str(), &end, numeric_locale.get());
  if (end !=
      trimmed.c_str() + static_cast<std::ptrdiff_t>(trimmed.size())) {
    Fail(path, "expected number");
  }
  return parsed;
}

double ParseDouble(const YAML::Node& node, const std::string& path) {
  const ScalarKind kind = ClassifyScalar(node);
  if (kind != ScalarKind::kInteger && kind != ScalarKind::kFloat) {
    Fail(path, "expected number");
  }
  if (kind == ScalarKind::kInteger) {
    return static_cast<double>(ParseInteger(node, path));
  }

  std::string value = LowerAscii(WithoutUnderscores(node.Scalar()));
  bool negative = false;
  if (!value.empty() && (value.front() == '-' || value.front() == '+')) {
    negative = value.front() == '-';
    value.erase(value.begin());
  }
  double parsed = 0.0;
  if (value == ".inf") {
    parsed = std::numeric_limits<double>::infinity();
  } else if (value == ".nan") {
    parsed = std::numeric_limits<double>::quiet_NaN();
  } else if (value.find(':') != std::string::npos) {
    std::vector<double> digits;
    std::size_t begin = 0;
    while (true) {
      const std::size_t end = value.find(':', begin);
      const std::string part = value.substr(begin, end - begin);
      digits.push_back(ParsePythonFloat(part, path));
      if (end == std::string::npos) {
        break;
      }
      begin = end + 1U;
    }
    std::reverse(digits.begin(), digits.end());
    double base = 1.0;
    for (const double digit : digits) {
      parsed += digit * base;
      if (!std::isfinite(parsed)) {
        Fail(path, "must be finite");
      }
      base *= 60.0;
    }
  } else {
    parsed = ParsePythonFloat(value, path);
  }
  if (negative) {
    parsed = -parsed;
  }
  if (!std::isfinite(parsed)) {
    Fail(path, "must be finite");
  }
  return parsed;
}

float BoundedFloat(double value, double lower, bool lower_inclusive,
                   double upper, bool upper_inclusive,
                   const std::string& path) {
  const bool below = lower_inclusive ? value < lower : value <= lower;
  const bool above = upper_inclusive ? value > upper : value >= upper;
  if (below || above) {
    const std::string interval =
        std::string(lower_inclusive ? "[" : "(") + std::to_string(lower) +
        ", " + std::to_string(upper) + (upper_inclusive ? "]" : ")");
    Fail(path, "must be in " + interval);
  }
  float result = static_cast<float>(value);
  const float lower_float = static_cast<float>(lower);
  const float upper_float = static_cast<float>(upper);
  if (!lower_inclusive && result <= lower_float) {
    result = std::nextafter(lower_float, std::numeric_limits<float>::infinity());
  }
  if (!upper_inclusive && result >= upper_float) {
    result = std::nextafter(upper_float, -std::numeric_limits<float>::infinity());
  }
  return result;
}

bool ParseBool(const YAML::Node& node, const std::string& path) {
  if (ClassifyScalar(node) != ScalarKind::kBoolean) {
    Fail(path, "expected boolean");
  }
  const std::string lower = LowerAscii(node.Scalar());
  if (lower == "true" || lower == "yes" || lower == "on") {
    return true;
  }
  if (lower == "false" || lower == "no" || lower == "off") {
    return false;
  }
  Fail(path, "expected boolean");
}

void RequireSchemaOne(const YAML::Node& node, const std::string& path) {
  if (ParseInteger(node, path) != 1) {
    Fail(path, "only schema version 1 is supported");
  }
}

std::string Trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

bool IsLowerHexId(const std::string& value) {
  return value.size() == 4U &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isdigit(character) != 0 ||
                  (character >= static_cast<unsigned char>('a') &&
                   character <= static_cast<unsigned char>('f'));
         });
}

DeviceIdentity ParseIdentity(const YAML::Node& node, const std::string& path) {
  CheckKeys(node, path, {"vendor_id", "product_id", "name", "serial"});
  const YAML::Node vendor = Required(node, "vendor_id", path);
  const YAML::Node product = Required(node, "product_id", path);
  DeviceIdentity identity;
  identity.vendor_id = ParseString(vendor, path + ".vendor_id");
  identity.product_id = ParseString(product, path + ".product_id");
  identity.name = ParseString(Required(node, "name", path), path + ".name");
  const YAML::Node serial = Required(node, "serial", path);
  if (ClassifyScalar(serial) != ScalarKind::kNull) {
    identity.serial = ParseString(serial, path + ".serial");
  }
  if (!IsLowerHexId(identity.vendor_id)) {
    Fail(path + ".vendor_id", "expected four lowercase hexadecimal characters");
  }
  if (!IsLowerHexId(identity.product_id)) {
    Fail(path + ".product_id", "expected four lowercase hexadecimal characters");
  }
  if (Trim(identity.name).empty()) {
    Fail(path + ".name", "must not be empty");
  }
  return identity;
}

bool ParseUnsupported(const YAML::Node& node, const std::string& path) {
  CheckKeys(node, path, {"unsupported"});
  if (!ParseBool(Required(node, "unsupported", path), path + ".unsupported")) {
    Fail(path + ".unsupported", "must be true");
  }
  return true;
}

StickVariant ParseStick(const YAML::Node& node, const std::string& path) {
  RequireMap(node, path);
  if (node["unsupported"]) {
    ParseUnsupported(node, path);
    return UnsupportedBinding{};
  }
  CheckKeys(node, path, {"axis", "center", "min", "max", "invert", "deadzone"});
  const std::string deadzone_path = path + ".deadzone";
  const double deadzone =
      ParseDouble(Required(node, "deadzone", path), deadzone_path);
  StickBinding binding{
      ParseIndex(Required(node, "axis", path), path + ".axis"),
      ParseInt16(Required(node, "center", path), path + ".center"),
      ParseInt16(Required(node, "min", path), path + ".min"),
      ParseInt16(Required(node, "max", path), path + ".max"),
      ParseBool(Required(node, "invert", path), path + ".invert"),
      BoundedFloat(deadzone, 0.0, true, 1.0, false, deadzone_path),
  };
  if (!(binding.minimum < binding.center && binding.center < binding.maximum)) {
    Fail(path, "stick range must contain center");
  }
  return binding;
}

std::optional<CorrelatedButton> ParseCorrelation(const YAML::Node& node,
                                                 const std::string& path) {
  if (ClassifyScalar(node) == ScalarKind::kNull) {
    return std::nullopt;
  }
  CheckKeys(node, path, {"index", "observed_within_ms"});
  return CorrelatedButton{
      ParseIndex(Required(node, "index", path), path + ".index"),
      ParseNonNegativeInt(Required(node, "observed_within_ms", path),
                          path + ".observed_within_ms"),
  };
}

TriggerBinding ParseTrigger(const YAML::Node& node, const std::string& path) {
  RequireMap(node, path);
  if (node["unsupported"]) {
    ParseUnsupported(node, path);
    return UnsupportedBinding{};
  }
  const std::string source =
      ParseString(Required(node, "source", path), path + ".source");
  if (source == "axis") {
    CheckKeys(node, path,
              {"source", "index", "released", "pressed", "threshold",
               "correlated_button"});
    const YAML::Node correlation = node["correlated_button"];
    const std::string threshold_path = path + ".threshold";
    const double threshold =
        ParseDouble(Required(node, "threshold", path), threshold_path);
    AxisTriggerBinding binding{
        ParseIndex(Required(node, "index", path), path + ".index"),
        ParseInt16(Required(node, "released", path), path + ".released"),
        ParseInt16(Required(node, "pressed", path), path + ".pressed"),
        BoundedFloat(threshold, 0.0, true, 1.0, true, threshold_path),
        correlation
            ? ParseCorrelation(correlation, path + ".correlated_button")
            : std::nullopt,
    };
    if (binding.released == binding.pressed) {
      Fail(path, "trigger released and pressed values must differ");
    }
    return binding;
  }
  if (source == "button") {
    CheckKeys(node, path, {"source", "index", "threshold"});
    const YAML::Node threshold_node = node["threshold"];
    const double threshold =
        threshold_node ? ParseDouble(threshold_node, path + ".threshold") : 0.5;
    if (threshold != 0.5) {
      Fail(path + ".threshold", "button trigger threshold must be 0.5");
    }
    return ButtonTriggerBinding{
        ParseIndex(Required(node, "index", path), path + ".index"),
        static_cast<float>(threshold)};
  }
  Fail(path + ".source", "expected axis or button");
}

ButtonVariant ParseButton(const YAML::Node& node, const std::string& path) {
  RequireMap(node, path);
  if (node["unsupported"]) {
    ParseUnsupported(node, path);
    return UnsupportedBinding{};
  }
  CheckKeys(node, path, {"source", "index"});
  if (ParseString(Required(node, "source", path), path + ".source") != "button") {
    Fail(path + ".source", "expected button");
  }
  return ButtonBinding{ParseIndex(Required(node, "index", path), path + ".index")};
}

DpadBinding ParseDpad(const YAML::Node& node, const std::string& path) {
  RequireMap(node, path);
  if (node["unsupported"]) {
    ParseUnsupported(node, path);
    return UnsupportedBinding{};
  }
  const std::string source =
      ParseString(Required(node, "source", path), path + ".source");
  if (source == "button") {
    CheckKeys(node, path, {"source", "index"});
    return ButtonBinding{ParseIndex(Required(node, "index", path), path + ".index")};
  }
  if (source != "axis") {
    Fail(path + ".source", "expected axis or button");
  }
  CheckKeys(node, path, {"source", "index", "direction", "threshold"});
  const std::string direction =
      ParseString(Required(node, "direction", path), path + ".direction");
  DpadDirection parsed_direction;
  if (direction == "negative") {
    parsed_direction = DpadDirection::kNegative;
  } else if (direction == "positive") {
    parsed_direction = DpadDirection::kPositive;
  } else {
    Fail(path + ".direction", "expected negative or positive");
  }
  const std::string threshold_path = path + ".threshold";
  const double threshold =
      ParseDouble(Required(node, "threshold", path), threshold_path);
  const float mapped_threshold =
      BoundedFloat(threshold, 0.0, false, 1.0, true, threshold_path);
  return AxisDpadBinding{
      ParseIndex(Required(node, "index", path), path + ".index"),
      parsed_direction,
      mapped_threshold,
  };
}

template <typename Binding, typename Parser>
std::map<std::string, Binding> ParseControlMap(
    const YAML::Node& node, const std::string& path,
    const std::vector<std::string>& expected, Parser parser) {
  RequireMap(node, path);
  std::set<std::string> permitted(expected.begin(), expected.end());
  std::set<std::string> seen;
  std::map<std::string, Binding> parsed;
  for (const auto& entry : node) {
    if (!entry.first.IsScalar()) {
      Fail(path, "control name must be a string");
    }
    const std::string name = entry.first.Scalar();
    const std::string field = path + "." + name;
    if (permitted.count(name) == 0U) {
      Fail(field, "unknown control");
    }
    if (!seen.insert(name).second) {
      Fail(field, "duplicate control");
    }
    parsed.emplace(name, parser(entry.second, field));
  }
  for (const auto& name : expected) {
    if (seen.count(name) == 0U) {
      Fail(path + "." + name, "control is required");
    }
  }
  return parsed;
}

bool IsUnsupported(const StickVariant& binding) {
  return std::holds_alternative<UnsupportedBinding>(binding);
}

bool IsUnsupported(const TriggerBinding& binding) {
  return std::holds_alternative<UnsupportedBinding>(binding);
}

bool IsUnsupported(const ButtonVariant& binding) {
  return std::holds_alternative<UnsupportedBinding>(binding);
}

bool IsUnsupported(const DpadBinding& binding) {
  return std::holds_alternative<UnsupportedBinding>(binding);
}

bool IdentityEqual(const DeviceIdentity& left, const DeviceIdentity& right) {
  return left.vendor_id == right.vendor_id && left.product_id == right.product_id &&
         left.name == right.name && left.serial == right.serial;
}

void ValidateProfileImpl(const GamepadProfile& profile,
                         const DeviceDescriptor* descriptor) {
  const auto require_supported = [](bool unsupported, const std::string& path) {
    if (unsupported) {
      Fail(path, "required control is unsupported");
    }
  };
  require_supported(IsUnsupported(profile.sticks.at("left_x")), "sticks.left_x");
  require_supported(IsUnsupported(profile.sticks.at("left_y")), "sticks.left_y");
  require_supported(IsUnsupported(profile.sticks.at("right_x")), "sticks.right_x");
  require_supported(IsUnsupported(profile.triggers.at("lt")), "triggers.lt");
  require_supported(IsUnsupported(profile.triggers.at("rt")), "triggers.rt");
  require_supported(IsUnsupported(profile.buttons.at("a")), "buttons.a");
  require_supported(IsUnsupported(profile.buttons.at("b")), "buttons.b");
  require_supported(IsUnsupported(profile.buttons.at("rb")), "buttons.rb");
  require_supported(IsUnsupported(profile.dpad.at("up")), "dpad.up");

  std::map<std::size_t, std::string> analog_axes;
  std::map<std::size_t, std::string> button_sources;
  std::map<std::size_t, std::vector<std::pair<std::string, DpadDirection>>> dpad_axes;
  const auto check_axis = [descriptor](std::size_t index, const std::string& path) {
    if (descriptor != nullptr && index >= descriptor->axis_count) {
      Fail(path, "axis index is outside detected capability " +
                     std::to_string(descriptor->axis_count));
    }
  };
  const auto check_button = [descriptor](std::size_t index, const std::string& path) {
    if (descriptor != nullptr && index >= descriptor->button_count) {
      Fail(path, "button index is outside detected capability " +
                     std::to_string(descriptor->button_count));
    }
  };
  const auto add_analog = [&analog_axes, &check_axis](
                              std::size_t index, const std::string& path,
                              const std::string& index_path) {
    const auto existing = analog_axes.find(index);
    if (existing != analog_axes.end()) {
      Fail(path, "duplicate analog axis " + std::to_string(index) +
                     " already used by " + existing->second);
    }
    analog_axes.emplace(index, path);
    check_axis(index, index_path);
  };
  const auto add_button = [&button_sources, &check_button](std::size_t index,
                                                           const std::string& path) {
    const auto existing = button_sources.find(index);
    if (existing != button_sources.end()) {
      Fail(path, "duplicate button " + std::to_string(index) +
                     " already used by " + existing->second);
    }
    button_sources.emplace(index, path);
    check_button(index, path + ".index");
  };

  for (const auto& [name, binding] : profile.sticks) {
    if (const auto* stick = std::get_if<StickBinding>(&binding)) {
      const std::string path = "sticks." + name;
      add_analog(stick->axis, path, path + ".axis");
    }
  }
  for (const auto& [name, binding] : profile.triggers) {
    const std::string path = "triggers." + name;
    if (const auto* axis = std::get_if<AxisTriggerBinding>(&binding)) {
      add_analog(axis->index, path, path + ".index");
      if (axis->correlated_button) {
        check_button(axis->correlated_button->index,
                     path + ".correlated_button.index");
      }
    } else if (const auto* button = std::get_if<ButtonTriggerBinding>(&binding)) {
      add_button(button->index, path);
    }
  }
  for (const auto& [name, binding] : profile.buttons) {
    if (const auto* button = std::get_if<ButtonBinding>(&binding)) {
      add_button(button->index, "buttons." + name);
    }
  }
  for (const auto& [name, binding] : profile.dpad) {
    const std::string path = "dpad." + name;
    if (const auto* button = std::get_if<ButtonBinding>(&binding)) {
      add_button(button->index, path);
      continue;
    }
    const auto* axis = std::get_if<AxisDpadBinding>(&binding);
    if (axis == nullptr) {
      continue;
    }
    check_axis(axis->index, path + ".index");
    auto& uses = dpad_axes[axis->index];
    for (const auto& [other_name, other_direction] : uses) {
      const bool opposite_pair =
          (name == "up" && other_name == "down") ||
          (name == "down" && other_name == "up") ||
          (name == "left" && other_name == "right") ||
          (name == "right" && other_name == "left");
      if (!opposite_pair || axis->direction == other_direction) {
        Fail(path, "duplicate or incompatible dpad axis direction " +
                       std::to_string(axis->index));
      }
    }
    uses.emplace_back(name, axis->direction);
  }
}

YAML::Node LoadYaml(const std::filesystem::path& path) {
  try {
    return YAML::LoadFile(path.string());
  } catch (const YAML::Exception& error) {
    throw GamepadConfigError(path.string() + ": invalid YAML: " + error.what());
  }
}

}  // namespace

GamepadProfile LoadGamepadProfile(const std::filesystem::path& path) {
  const YAML::Node root = LoadYaml(path);
  CheckKeys(root, "", {"schema_version", "device", "sticks", "triggers", "buttons", "dpad"});
  RequireSchemaOne(Required(root, "schema_version", ""), "schema_version");
  GamepadProfile profile{
      1,
      ParseIdentity(Required(root, "device", ""), "device"),
      ParseControlMap<StickVariant>(
          Required(root, "sticks", ""), "sticks",
          {"left_x", "left_y", "right_x", "right_y"}, ParseStick),
      ParseControlMap<TriggerBinding>(Required(root, "triggers", ""), "triggers",
                                      {"lt", "rt"}, ParseTrigger),
      ParseControlMap<ButtonVariant>(
          Required(root, "buttons", ""), "buttons",
          {"a", "b", "x", "y", "lb", "rb", "start", "back", "left_stick",
           "right_stick"},
          ParseButton),
      ParseControlMap<DpadBinding>(Required(root, "dpad", ""), "dpad",
                                   {"up", "down", "left", "right"}, ParseDpad),
  };
  ValidateProfileImpl(profile, nullptr);
  return profile;
}

ActiveSelection LoadActiveSelection(const std::filesystem::path& path) {
  const YAML::Node root = LoadYaml(path);
  CheckKeys(root, "", {"schema_version", "profile", "device"});
  RequireSchemaOne(Required(root, "schema_version", ""), "schema_version");
  const std::string profile_text =
      ParseString(Required(root, "profile", ""), "profile");
  const std::filesystem::path profile_path(profile_text);
  if (profile_path.is_absolute() || profile_text.empty()) {
    Fail("profile", "must be a relative path");
  }
  for (const auto& component : profile_path) {
    if (component == ".." || component == ".") {
      Fail("profile", "must not contain . or .. components");
    }
  }
  const auto normalized = profile_path.lexically_normal();
  if (normalized.generic_string() != profile_text || normalized.begin() == normalized.end() ||
      *normalized.begin() != "profiles" || normalized.extension() != ".yaml") {
    Fail("profile", "must be normalized profiles/*.yaml path");
  }
  const auto parent = path.parent_path().lexically_normal();
  const auto resolved = (parent / normalized).lexically_normal();
  const auto relative = resolved.lexically_relative(parent);
  if (relative.empty() || *relative.begin() == "..") {
    Fail("profile", "escapes active selection directory");
  }
  const DeviceIdentity identity =
      ParseIdentity(Required(root, "device", ""), "device");
  const GamepadProfile profile = LoadGamepadProfile(resolved);
  if (!IdentityEqual(identity, profile.device)) {
    Fail("device", "active identity does not match profile identity");
  }
  return ActiveSelection{1, normalized, identity};
}

void ValidateProfile(const GamepadProfile& profile,
                     const DeviceDescriptor& descriptor) {
  ValidateProfileImpl(profile, &descriptor);
}

}  // namespace gamepad
