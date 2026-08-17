# Architecture

## Project boundary

This repository is the first completed phase of a G1-only commanded locomotion
stack. It covers training, deterministic Play, gamepad input, MuJoCo sim-to-sim,
the G1 finite-state controller, and aarch64 deployment inputs.

Stepping-stone or plum-blossom-pole locomotion is deliberately outside this
boundary. That work may need a different command representation, observation
contract, terrain generator, policy architecture, and evaluation protocol, so
it should begin in a separate repository rather than replacing this release.

## Training data flow

~~~mermaid
flowchart LR
  TASK[G1 task configuration] --> CMD[twist + swing height + step length]
  CMD --> OBS[actor and critic observations]
  OBS --> PPO[PPO actor-critic]
  PPO --> ACT[29 joint-position actions]
  ACT --> MJW[MuJoCo-Warp environments]
  MJW --> REW[tracking, lift, reach, stability rewards]
  REW --> PPO
  PPO --> PT[selected PT checkpoint]
  PT --> PLAY[deterministic Play]
  PT --> EXPORT[velocity ONNX export]
~~~

The selected policy contract is:

| Signal | Dimension | Meaning |
| --- | ---: | --- |
| actor observation | 100 | deployable proprioception, commands, gait phase, previous action |
| critic observation | 302 | actor inputs plus privileged training-only state |
| action | 29 | G1 joint-position targets |
| twist command | 3 | forward/lateral velocity and yaw rate |
| swing-height command | 1 | low, medium, or high foot-lift intent |
| step-length command | 1 | short, medium, or long forward-step intent |

The scalar height and length commands are part of the neural-network input.
They are not MuJoCo-only controls and are exported into the deployment
observation contract.

## Runtime split

~~~mermaid
flowchart LR
  PAD[Linux joystick] --> CAL[loopback mapping console]
  CAL --> PROFILE[active.yaml]
  PROFILE --> SIM[unitree_mujoco]
  SIM <-->|Unitree DDS topics| CTRL[g1_ctrl]
  POLICY[velocity policy.onnx] --> CTRL
  CTRL --> FSM[Passive / FixStand / Velocity / Fallen / GetUp]
  CTRL --> HW[physical G1]
~~~

The mapping console only observes Linux joystick events and writes a profile.
It does not call MuJoCo APIs, start the controller, or issue robot commands.

In simulation, `unitree_mujoco` owns the physical joystick and publishes a
Unitree wireless-controller state over DDS. `g1_ctrl` consumes that state and
the simulated robot state. On hardware, `g1_ctrl` uses `CustomJoystick` to
read the same tested third-party controller from `/dev/input/js0`.

## Command processing

The final control mapping intentionally separates state selection from motion:

- right-stick X selects the step-length state: 0.20, 0.30, or 0.40 m;
- right-stick Y selects the swing-height state: 0.05, 0.10, or 0.20 m;
- D-pad selects one of four phase-latched single-step directions;
- LB/RB request yaw and LT applies a fixed 1.5x yaw multiplier.

Moving the right stick alone does not move the robot. The selected height and
length are inserted into the next policy observation while a D-pad or yaw
motion command is active.

The phase-latched single-step controller is implemented in the G1 control
layer. It observes the policy gait phase, emits a bounded velocity command for
one phase advance, and requires release before another step. It does not use a
MuJoCo-specific API and therefore has the same control semantics in sim-to-sim
and on the robot.

## State machine

~~~mermaid
stateDiagram-v2
  [*] --> Passive
  Passive --> FixStand: Start
  FixStand --> Velocity: RT + A
  Velocity --> Fallen: confirmed fall
  Fallen --> GetUp: release A, then hold A for 1 s
  GetUp --> Velocity: stable upright
  GetUp --> Fallen: failure or timeout
  FixStand --> Passive: LT + B
  Velocity --> Passive: LT + B
  Fallen --> Passive: LT + B
~~~

Fallen applies damping output and is available without the optional GetUp
model. GetUp is lazy-loaded and fails safely back to Fallen when its model is
absent or invalid.

## Source map

| Path | Responsibility |
| --- | --- |
| `src/tasks/velocity/config/g1/` | G1 task and PPO configurations |
| `src/tasks/velocity/mdp/` | command, observation, reward, and termination terms |
| `scripts/train.py` | CLI training entry point and tuning-profile integration |
| `scripts/play_forward.py` | deterministic command-level Play |
| `src/training_console/` | local training/terminal/TensorBoard control plane |
| `src/gamepad_calibrator/` | local joystick inspection and mapping UI |
| `simulate/src/gamepad/` | strict C++ profile parser, discovery, and logical mapper |
| `deploy/include/isaaclab/` | policy environment and external command pipeline |
| `deploy/robots/g1/` | G1 FSM, policies, configuration, and executable |
| `artifacts/g1-commanded-locomotion-v1/` | selected checkpoint and compact evidence |
