// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright Drew Noakes 2013-2016

#include <cerrno>
#include <utility>
#include "joystick.h"

Joystick::Joystick()
{
  openPath("/dev/input/js0");
}

Joystick::Joystick(int joystickNumber)
{
  std::stringstream sstm;
  sstm << "/dev/input/js" << joystickNumber;
  openPath(sstm.str());
}

Joystick::Joystick(std::string devicePath)
{
  openPath(devicePath);
}

Joystick::Joystick(std::string devicePath, bool blocking)
{
  openPath(devicePath, blocking);
}

Joystick::Joystick(int fd, bool owns_fd, ReadOperation read_operation)
    : _fd(fd), owns_fd_(owns_fd), read_operation_(std::move(read_operation))
{
}

void Joystick::openPath(std::string devicePath, bool blocking)
{
  // Open the device using either blocking or non-blocking
  _fd = open(devicePath.c_str(), blocking ? O_RDONLY : O_RDONLY | O_NONBLOCK);
  if (_fd < 0)
  {
    last_error_ = errno;
  }
}

bool Joystick::sample(JoystickEvent *event)
{
  return readEvent(event) == ReadStatus::Event;
}

Joystick::ReadStatus Joystick::readEvent(JoystickEvent *event)
{
  const ssize_t bytes = read_operation_(_fd, event, sizeof(*event));
  if (bytes == static_cast<ssize_t>(sizeof(*event)))
  {
    last_error_ = 0;
    return ReadStatus::Event;
  }
  if (bytes == 0)
  {
    last_error_ = 0;
    return ReadStatus::Disconnected;
  }
  if (bytes < 0)
  {
    last_error_ = errno;
    if (errno == EAGAIN || errno == EWOULDBLOCK)
    {
      return ReadStatus::WouldBlock;
    }
    if (errno == ENODEV || errno == EIO)
    {
      return ReadStatus::Disconnected;
    }
    return ReadStatus::Error;
  }
  last_error_ = EIO;
  return ReadStatus::Error;
}

int Joystick::lastError() const
{
  return last_error_;
}

bool Joystick::isFound()
{
  return _fd >= 0;
}

Joystick::~Joystick()
{
  if (owns_fd_ && _fd >= 0)
  {
    close(_fd);
  }
}

std::ostream &operator<<(std::ostream &os, const JoystickEvent &e)
{
  os << "type=" << static_cast<int>(e.type)
     << " number=" << static_cast<int>(e.number)
     << " value=" << static_cast<int>(e.value);
  return os;
}
