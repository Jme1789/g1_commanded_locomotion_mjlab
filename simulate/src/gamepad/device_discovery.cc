#include "gamepad/device_discovery.h"

#include <fcntl.h>
#include <linux/joystick.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace gamepad {
namespace {

struct NodeFingerprint {
  dev_t device;
  ino_t inode;
  dev_t raw_device;
  mode_t type;
};

class ScopedFd {
 public:
  explicit ScopedFd(int value) : value_(value) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ~ScopedFd() {
    if (value_ >= 0) {
      ::close(value_);
    }
  }

  int get() const { return value_; }

 private:
  int value_;
};

std::string Trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

bool IsMissing(const std::error_code& error) {
  return error == std::errc::no_such_file_or_directory;
}

std::string ReadOpened(std::ifstream& stream,
                       const std::filesystem::path& path,
                       const std::string& field) {
  std::string contents;
  char buffer[4096];
  while (stream) {
    stream.read(buffer, sizeof(buffer));
    contents.append(buffer, static_cast<std::size_t>(stream.gcount()));
  }
  if (!stream.eof()) {
    throw GamepadConfigError("device." + field + ": incomplete read from " +
                             path.string());
  }
  return Trim(contents);
}

std::string ReadRequired(const std::filesystem::path& path,
                         const std::string& field) {
  std::ifstream stream(path);
  if (!stream) {
    throw GamepadConfigError("device." + field + ": cannot read " + path.string());
  }
  const std::string value = ReadOpened(stream, path, field);
  if (value.empty()) {
    throw GamepadConfigError("device." + field + ": value is empty");
  }
  return value;
}

std::optional<std::string> ReadOptional(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::status(path, error);
  if (error) {
    if (IsMissing(error)) {
      return std::nullopt;
    }
    throw GamepadConfigError("device.serial: cannot inspect " + path.string() +
                             ": " + error.message());
  }
  if (!std::filesystem::exists(status)) {
    return std::nullopt;
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw GamepadConfigError("device.serial: expected readable regular file " +
                             path.string());
  }
  std::ifstream stream(path);
  if (!stream) {
    throw GamepadConfigError("device.serial: cannot read " + path.string());
  }
  const std::string value = ReadOpened(stream, path, "serial");
  return value.empty() ? std::nullopt : std::optional<std::string>(value);
}

bool IsJoystickName(const std::string& name) {
  return name.size() > 2U && name.rfind("js", 0U) == 0U &&
         std::all_of(name.begin() + 2, name.end(), [](unsigned char character) {
           return std::isdigit(character) != 0;
         });
}

unsigned long long ParseJoystickNumber(const std::filesystem::path& path) {
  const std::string suffix = path.filename().string().substr(2);
  try {
    std::size_t used = 0;
    const unsigned long long value = std::stoull(suffix, &used, 10);
    if (used != suffix.size()) {
      throw std::invalid_argument("partial joystick suffix");
    }
    return value;
  } catch (const std::exception& error) {
    throw GamepadConfigError("dev_input: invalid joystick node " + path.string() +
                             ": " + error.what());
  }
}

bool IsLowerHexId(const std::string& value) {
  return value.size() == 4U &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isdigit(character) != 0 ||
                  (character >= static_cast<unsigned char>('a') &&
                   character <= static_cast<unsigned char>('f'));
         });
}

DeviceIdentity ReadIdentity(const std::filesystem::path& sys_device) {
  std::string vendor = ReadRequired(sys_device / "id/vendor", "vendor_id");
  std::string product = ReadRequired(sys_device / "id/product", "product_id");
  std::transform(vendor.begin(), vendor.end(), vendor.begin(),
                 [](unsigned char character) { return std::tolower(character); });
  std::transform(product.begin(), product.end(), product.begin(),
                 [](unsigned char character) { return std::tolower(character); });
  if (!IsLowerHexId(vendor)) {
    throw GamepadConfigError(
        "device.vendor_id: expected four hexadecimal characters");
  }
  if (!IsLowerHexId(product)) {
    throw GamepadConfigError(
        "device.product_id: expected four hexadecimal characters");
  }
  return DeviceIdentity{
      vendor,
      product,
      ReadRequired(sys_device / "name", "name"),
      ReadOptional(sys_device / "uniq"),
  };
}

bool SameIdentity(const DeviceIdentity& left, const DeviceIdentity& right) {
  return left.vendor_id == right.vendor_id &&
         left.product_id == right.product_id && left.name == right.name &&
         left.serial == right.serial;
}

NodeFingerprint FingerprintFd(int descriptor,
                              const std::filesystem::path& path) {
  struct stat status {};
  if (::fstat(descriptor, &status) != 0) {
    throw GamepadConfigError("device.path: cannot inspect opened joystick " +
                             path.string() + ": " + std::strerror(errno));
  }
  return NodeFingerprint{
      status.st_dev,
      status.st_ino,
      status.st_rdev,
      static_cast<mode_t>(status.st_mode & S_IFMT),
  };
}

NodeFingerprint FingerprintPath(const std::filesystem::path& path,
                                const std::string& field) {
  struct stat status {};
  if (::stat(path.c_str(), &status) != 0) {
    throw GamepadConfigError(field + ": cannot inspect " + path.string() +
                             ": " + std::strerror(errno));
  }
  return NodeFingerprint{
      status.st_dev,
      status.st_ino,
      status.st_rdev,
      static_cast<mode_t>(status.st_mode & S_IFMT),
  };
}

bool SameFingerprint(const NodeFingerprint& left,
                     const NodeFingerprint& right) {
  return left.device == right.device && left.inode == right.inode &&
         left.raw_device == right.raw_device && left.type == right.type;
}

void RequireCurrentPath(const std::filesystem::path& path,
                        const NodeFingerprint& opened) {
  if (!SameFingerprint(FingerprintPath(path, "device.path"), opened)) {
    throw GamepadConfigError(
        "device.path: joystick node changed during enumeration " + path.string());
  }
}

std::optional<std::filesystem::path> FindById(
    const NodeFingerprint& opened,
    const std::filesystem::path& by_id_root) {
  std::error_code error;
  const auto status = std::filesystem::status(by_id_root, error);
  if (error) {
    if (IsMissing(error)) {
      return std::nullopt;
    }
    throw GamepadConfigError("by_id: cannot inspect " + by_id_root.string() +
                             ": " + error.message());
  }
  if (!std::filesystem::exists(status)) {
    return std::nullopt;
  }
  if (!std::filesystem::is_directory(status)) {
    throw GamepadConfigError("by_id: expected directory " + by_id_root.string());
  }

  std::vector<std::filesystem::path> candidates;
  for (std::filesystem::directory_iterator iterator(by_id_root, error), end;
       !error && iterator != end; iterator.increment(error)) {
    candidates.push_back(iterator->path());
  }
  if (error) {
    throw GamepadConfigError("by_id: cannot enumerate " + by_id_root.string() +
                             ": " + error.message());
  }
  std::sort(candidates.begin(), candidates.end());
  for (const auto& candidate : candidates) {
    const auto fingerprint = FingerprintPath(candidate, "by_id");
    if (SameFingerprint(fingerprint, opened)) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::pair<std::size_t, std::size_t> ReadCounts(
    int descriptor, const std::filesystem::path& path) {
  unsigned char axes = 0;
  unsigned char buttons = 0;
  if (::ioctl(descriptor, JSIOCGAXES, &axes) < 0) {
    throw GamepadConfigError("device.capabilities: cannot query joystick axes " +
                             path.string() + ": " + std::strerror(errno));
  }
  if (::ioctl(descriptor, JSIOCGBUTTONS, &buttons) < 0) {
    throw GamepadConfigError(
        "device.capabilities: cannot query joystick buttons " + path.string() +
        ": " + std::strerror(errno));
  }
  return {axes, buttons};
}

std::string CandidateText(const DeviceDescriptor& descriptor) {
  std::ostringstream stream;
  stream << descriptor.path << " [" << Trim(descriptor.identity.vendor_id) << ':'
         << Trim(descriptor.identity.product_id) << " name="
         << Trim(descriptor.identity.name) << " serial=";
  if (descriptor.identity.serial) {
    stream << *descriptor.identity.serial;
  } else {
    stream << "<none>";
  }
  stream << ']';
  return stream.str();
}

bool IdentityMatches(const DeviceIdentity& selected,
                     const DeviceIdentity& candidate) {
  return Trim(selected.vendor_id) == Trim(candidate.vendor_id) &&
         Trim(selected.product_id) == Trim(candidate.product_id) &&
         Trim(selected.name) == Trim(candidate.name) &&
         (!selected.serial ||
          (candidate.serial && *selected.serial == *candidate.serial));
}

}  // namespace

DiscoveryRoots DiscoveryRoots::System() {
  return DiscoveryRoots{"/dev/input", "/sys/class/input", "/dev/input/by-id"};
}

std::vector<DeviceDescriptor> EnumerateLinuxJoysticks(
    const DiscoveryRoots& roots) {
  std::error_code error;
  std::vector<std::pair<unsigned long long, std::filesystem::path>> paths;
  for (std::filesystem::directory_iterator iterator(roots.dev_input, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const auto path = iterator->path();
    if (IsJoystickName(path.filename().string())) {
      paths.emplace_back(ParseJoystickNumber(path), path);
    }
  }
  if (error) {
    throw GamepadConfigError("dev_input: cannot enumerate " +
                             roots.dev_input.string() + ": " + error.message());
  }
  std::sort(paths.begin(), paths.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });

  std::vector<DeviceDescriptor> result;
  for (const auto& [number, path] : paths) {
    (void)number;
    const int raw_descriptor =
        ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (raw_descriptor < 0) {
      throw GamepadConfigError("device.path: cannot open joystick " +
                               path.string() + ": " + std::strerror(errno));
    }
    const ScopedFd descriptor(raw_descriptor);
    const NodeFingerprint opened = FingerprintFd(descriptor.get(), path);
    const auto sys_device =
        roots.sys_class_input / path.filename() / "device";
    const DeviceIdentity before = ReadIdentity(sys_device);
    RequireCurrentPath(path, opened);
    const auto by_id_path = FindById(opened, roots.by_id);
    RequireCurrentPath(path, opened);
    const auto counts = ReadCounts(descriptor.get(), path);
    RequireCurrentPath(path, opened);
    const DeviceIdentity after = ReadIdentity(sys_device);
    RequireCurrentPath(path, opened);
    if (!SameIdentity(before, after)) {
      throw GamepadConfigError(
          "device.identity: identity changed during enumeration " +
          path.string());
    }
    if (by_id_path &&
        !SameFingerprint(FingerprintPath(*by_id_path, "by_id"), opened)) {
      throw GamepadConfigError(
          "by_id: link changed during enumeration " + by_id_path->string());
    }
    result.push_back(DeviceDescriptor{
        path,
        before,
        counts.first,
        counts.second,
        by_id_path,
    });
  }
  return result;
}

DeviceDescriptor SelectUniqueDevice(
    const DeviceIdentity& identity,
    const std::vector<DeviceDescriptor>& descriptors) {
  std::vector<const DeviceDescriptor*> matches;
  for (const auto& descriptor : descriptors) {
    if (IdentityMatches(identity, descriptor.identity)) {
      matches.push_back(&descriptor);
    }
  }
  if (matches.size() == 1U) {
    return *matches.front();
  }
  std::ostringstream message;
  message << "device: expected exactly one identity match, found " << matches.size()
          << "; candidates:";
  for (const auto& descriptor : descriptors) {
    message << "\n- " << CandidateText(descriptor);
  }
  throw GamepadConfigError(message.str());
}

}  // namespace gamepad
