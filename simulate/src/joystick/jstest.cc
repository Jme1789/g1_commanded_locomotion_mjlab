#include <unistd.h>
#include <iostream>
#include "joystick.h"

int main()
{
  // Create an instance of Joystick
  Joystick joystick("/dev/input/js0");

  // Ensure that it was found and that we can use it
  if (!joystick.isFound())
  {
    printf("open failed.\n");
    exit(1);
  }

  while (true)
  {
    JoystickEvent event{};
    if (joystick.sample(&event))
    {
      std::cout << event << std::endl;
    }

    // Restrict rate
    usleep(10000);
  }
  return 0;
}
