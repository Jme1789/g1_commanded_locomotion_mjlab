# Safety

This repository controls a high-power humanoid robot. Simulation success does
not establish hardware safety.

## Before any physical run

- Use a suspension rig or support frame for the first test of every new binary,
  policy, gain set, controller, or joystick.
- Clear people and obstacles from the reachable area.
- Keep a working hardware emergency stop within immediate reach.
- Verify joint order, action dimension, gains, default pose, and policy hashes.
- Confirm the correct DDS network interface and that no competing low-level
  controller is running.
- Verify that the third-party joystick is stable in one hardware mode and that
  `/dev/input/js0` is readable.
- Keep `g1_ctrl` in a foreground terminal so logs and termination are visible.
- Start from Passive, then FixStand, and enter Velocity only after the robot is
  mechanically supported and state feedback is sane.

## Runtime boundaries

- The D-pad controller emits one phase-latched command and requires release
  before retriggering. Do not defeat its timeout or release interlock.
- Swing height and step length are policy commands, not guaranteed physical
  clearances.
- Fallen applies damping; it is not an emergency stop.
- GetUp is optional and must never be tested without support. A missing or
  invalid model keeps the controller in Fallen.
- A disconnected joystick is neutralized, but the real joystick reader does
  not hot-reconnect. Restart the controller after reconnection.
- Do not test stairs, edges, or uneven terrain until standing, short motion,
  stop, fall detection, and Passive transitions have all been validated.

## Stop conditions

Immediately stop and return to Passive or use the emergency stop if any of the
following occurs:

- unexpected joint direction or order;
- oscillation, foot scissoring, or rapidly growing yaw;
- non-finite state/action diagnostics;
- loss of DDS state updates;
- repeated joystick reconnects or mode switching;
- controller/model dimension or hash mismatch;
- contact with a stair edge that destabilizes the unsupported robot.

## Research limitations

The selected policy was trained in simulation. Friction, actuator delay,
compliance, backlash, payload, battery state, state-estimation error, and
contact geometry can differ on hardware. The included training metrics and
visual Play are evidence of a development result, not a warranty, validation
certificate, or assurance of safe autonomous stair traversal.
