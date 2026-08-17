# Deployment

## Responsibility boundary

Two processes are used in sim-to-sim:

1. `unitree_mujoco` owns the simulator and the physical Linux joystick.
2. `g1_ctrl` owns the G1 FSM, policy inference, and low-level command output.

Always start the simulator first. The controller waits for robot state over DDS
and cannot connect to a simulator that is not running.

## Prepare the gamepad profile

The released simulator intentionally does not commit a machine-specific
`active.yaml`.

~~~bash
ls -l /dev/input/js*
python -m scripts.gamepad_calibrator
~~~

Open `http://127.0.0.1:8766`, inspect the live axis/button transitions, map the
tested controller, save, and activate it. The resulting
`simulate/config/gamepads/active.yaml` is local and ignored by Git.

The C++ selector matches joystick identity, not a USB-A/USB-C physical port.
If the same controller exposes a different vendor/product/name in another
mode, create or activate a matching profile. The real-robot
`CustomJoystick` path uses the tested BEITONG event layout directly.

## Build sim-to-sim

Dependencies include CMake, a C++17 compiler, Boost program_options, yaml-cpp,
fmt, GLFW, Unitree SDK2, and CycloneDDS.

~~~bash
cmake -S simulate -B simulate/build
cmake --build simulate/build --target unitree_mujoco jstest -j2

cmake -S deploy/robots/g1 -B deploy/robots/g1/build
cmake --build deploy/robots/g1/build --target g1_ctrl -j2
~~~

## Run sim-to-sim

Terminal 1:

~~~bash
./simulate/build/unitree_mujoco
~~~

Terminal 2, after the MuJoCo window and DDS initialization are ready:

~~~bash
./deploy/robots/g1/build/g1_ctrl --network=lo
~~~

Keep the controller in the foreground. Keyboard transitions use that terminal's
input, while joystick input is forwarded by the simulator.

## Controller sequence

1. Press Start to enter FixStand.
2. Press RT+A to enter Velocity.
3. Select swing height with right-stick Y.
4. Select step length with right-stick X.
5. Press a D-pad direction for one phase-latched step.
6. Use LB/RB for yaw; hold LT for the fixed 1.5x yaw multiplier.
7. Use LT+B to return to Passive where configured.

A fall confirmed during Velocity enters Fallen damping protection. To request
optional recovery, release A fully and then hold A for one continuous second.

## Real G1 package

The repository contains source and preselected policy/configuration artifacts,
not a portable x86 executable. Build on the aarch64 G1 so the binary links
against the robot's installed Unitree SDK2/CycloneDDS environment:

~~~bash
./build_on_g1.sh
~~~

The script:

- refuses to build on a non-aarch64 host;
- verifies the release manifest;
- builds `deploy/robots/g1/build/g1_ctrl`;
- checks dynamic-library resolution with `ldd`.

Start with the robot network interface and joystick event path:

~~~bash
ip -br link
ls -l /dev/input/js*
./start_g1.sh eth0 /dev/input/js0
~~~

Use the interface connected to the G1 control network; do not assume it is
always `eth0`.

## Optional GetUp model

Fallen damping protection is bundled and remains usable without GetUp weights.
The referenced external GetUp ONNX is deliberately excluded because this
release does not have explicit permission to redistribute that model.

If you independently obtain a compatible model and have the right to use it,
place it at:

~~~text
deploy/robots/g1/config/policy/getup/amp_reference/exported/policy.onnx
~~~

It must match the observation/action contract in the adjacent
`params/deploy.yaml`. Missing, malformed, non-finite, or dimensionally
incompatible input/model state causes GetUp to fail safely back to Fallen.

## Troubleshooting

- **No joystick candidate:** verify `/dev/input/js0`, user membership in
  `input`, and the active profile identity.
- **Controller waits forever:** start MuJoCo first and use `--network=lo` for
  sim-to-sim.
- **Library not found:** inspect `ldd deploy/robots/g1/build/g1_ctrl` and the
  SDK/CycloneDDS/ONNX Runtime paths.
- **Joystick reconnects or changes mode:** stabilize the controller's hardware
  mode before launch, regenerate the profile, and restart both processes.
- **Clock-skew warnings on G1:** synchronize host/robot time, then rebuild from
  a clean build directory; stale future timestamps can prevent reliable
  incremental builds.
