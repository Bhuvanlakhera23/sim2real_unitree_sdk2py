📜 Go2 Locomotion Policy Contract (Sim ↔ Real)

Author: Bhuvan Lakhera
Robot: Unitree Go2 (12-DoF)
Policy Type: RL locomotion policy (position-target output)
Control Mode: Joint position PD
Execution Rate: 50 Hz (policy), ≥500 Hz inner loop (sim / firmware)

1. Purpose of This Document

This document defines the non-negotiable interface contract between:

a trained locomotion policy

the simulation environment

the real robot deployment stack

Any violation of this contract invalidates the policy and will result in unstable, biased, or unsafe behavior.

This is not a config file.
This is the ground truth specification.

2. Policy I/O Definition (Immutable)
2.1 Action Space (STRICT)
Property	Value
num_actions	12
Action range	[-1, 1] (normalized)
Semantic meaning	Δ joint position offsets
Output type	float32

Joint order (MUST NOT CHANGE):

[hip, thigh, calf] × [FL, FR, RL, RR]


Final target computation:

target_q = default_angles + action * action_scale


⚠️ The policy does not output torques.
⚠️ The policy does not know about PD gains.

2.2 Observation Space (STRICT)
Property	Value
num_obs	48
Observation order	Fixed
Normalization	Mandatory

Observation layout (index-accurate):

0–2    : base linear velocity (body frame) × lin_vel_scale
3–5    : base angular velocity (body frame) × ang_vel_scale
6–8    : gravity vector (projected)
9–11   : command [vx, vy, yaw_rate] × cmd_scale
12–23  : (q - default_angles) × dof_pos_scale
24–35  : dq × dof_vel_scale
36–47  : previous action


❌ Adding/removing/reordering entries breaks the policy
❌ Changing frames (world ↔ body) breaks the policy

3. Normalization Constants (Frozen)

These values are part of the trained model, not tunables.

Parameter	Value
lin_vel_scale	2.0
ang_vel_scale	0.25
dof_pos_scale	1.0
dof_vel_scale	0.05
action_scale	0.25

🔒 These must be identical in:

training

MuJoCo simulation

hardware deployment

4. Timing Contract
4.1 Policy Timing (Immutable)
Property	Value
Policy rate	50 Hz
Control period	0.02 s
Execution	Deterministic

The policy assumes:

fixed-rate execution

no skipped steps

no variable dt

4.2 Inner Control Loop (Environment-dependent)
Environment	Inner Loop
MuJoCo	dt = 0.002 s, decimation = 10
Hardware	Firmware loop (~500 Hz)

✔ Inner loop may differ
❌ Policy rate must not

5. Default Pose (Critical Reference)
default_angles:
  0.0, 0.80, -1.50,
  0.0, 0.80, -1.50,
  0.0, 1.00, -1.50,
  0.0, 1.00, -1.50


This pose is:

the zero-action equilibrium

the reference for all observations

assumed by the policy at reset

⚠️ Changing this requires retraining.

6. PD Control Layer (OUTSIDE Policy)

The policy is PD-agnostic.

PD gains:

may differ between sim and real

may be tuned for safety

must remain stable and overdamped

6.1 What PD Gains Affect
Effect	PD layer
Tracking stiffness	✅
Oscillation damping	✅
Torque magnitude	✅
Policy behavior	❌ (indirect only)
7. Command Interface
7.1 Command Semantics
cmd = [vx, vy, yaw_rate]


Body-frame

Continuous

Assumed smooth

7.2 Command Scaling
Environment	cmd_scale
Simulation	[2.0, 2.0, 0.25]
Hardware	[0.9, 0.4, 0.25]

✔ Scaling may differ
❌ Command ordering may not

8. Allowed Modifications (Safe)

You may safely change:

PD gains

command limits

simulation timestep

terrain

domain randomization

logging / plotting

camera behavior

9. Forbidden Modifications (Policy-Breaking)

❌ Changing observation order
❌ Changing normalization constants
❌ Changing action semantics
❌ Changing default joint angles
❌ Mixing torque and position control
❌ Running policy at variable rate

Any of the above invalidates all results.

10. Versioning Rule

Each trained policy must be accompanied by:

this contract

the exact training config

the exact observation definition

Policy ≠ file
Policy = file + contract

11. One-Line Summary (for collaborators)

“If it touches observations, normalization, action semantics, or timing — retrain.
If it touches PD, commands, or physics — tune.”