```
# Go2 Low-Level Control Stack (Simulation + Deployment)

> ⚠️ **Research Sandbox — Use at Your Own Risk**  
> This directory is provided as a **research and experimentation sandbox**.  
> The author does **not take responsibility** for hardware damage, falls, or unsafe operation.  
> Always operate the robot in a **harnessed, open environment**.

---

## 1. Overview

This directory contains a **hardware-aligned low-level control pipeline** for the **Unitree Go2**, designed for **sim-to-real locomotion research**.

It includes:

- MuJoCo-based simulation with policy inference
- Legacy Isaac Gym–based validation scripts
- Safe hardware deployment via **joint position PD control**
- CSV-based logging for debugging and gait inspection

This stack is intended for **researchers and engineers**, not for production or consumer use.

---

## 2. Repository Scope & Ownership

This repository contains **two distinct components**:

1. **Unitree SDK2 Python source code and official examples**
   - Located under `unitree_sdk2py/` and `example/`
   - Copyright © Unitree Robotics
   - Largely unmodified vendor code

2. **Custom Go2 sim-to-real research stack**
   - Located under `example/go2/low_level/`
   - Developed specifically for RL-based locomotion research
   - **Not part of the official Unitree SDK**

Only the **Go2 low-level stack** is actively developed and maintained here.

---

## 3. Supported Scope & Limitations

### Supported
- ✅ Unitree **Go2 only**
- ✅ MuJoCo simulation
- ✅ Legacy Isaac Gym validation (see notes below)
- ✅ Joint position PD control on hardware
- ✅ TorchScript policy inference
- ✅ CSV logging for pose and gait inspection

### Not Supported
- ❌ Policy training (Isaac Gym / Isaac Lab training pipelines not included)
- ❌ Torque control on hardware
- ❌ Other robots or morphologies
- ❌ Automatic recovery or fall detection
- ❌ Safety supervisors beyond PD + damping mode

---

## 4. Directory Structure (Relevant Subset)

```

low_level/
├── common/                  # Shared helpers (paths, math, controller utilities)
├── config/
│   ├── sim/v1/              # MuJoCo simulation YAMLs
│   └── deploy/v1/           # Hardware deployment YAMLs
├── simulate/
│   ├── mujoco/v1/           # MuJoCo simulation scripts + assets
│   └── gym/v1/              # Legacy Isaac Gym validation scripts
├── deploy/
│   └── v1/                  # Hardware deployment scripts
├── debug/                   # Policy inspection, mode switching
├── policies/                # Policy artifacts (NOT tracked in git)
├── plots/
│   ├── sim_op/              # Simulation logs & plots
│   └── real_op/             # Hardware logs (if enabled)
├── docs/
│   ├── README.md            # This document
│   └── policy_contract.md   # Immutable policy interface specification

```

---

## 5. Versioning Philosophy

- **v1.0**  
  First working deployment for a given policy interface.

- **v1.1**  
  Same control logic as v1.0, **adds logging only**.

- **v2.x (future)**  
  Reserved for **newly trained policies with new contracts**.

> All `v1.x` scripts are compatible with the **v1 policy interface**,  
> not with a specific policy filename.

---

## 6. Policy Contract (CRITICAL)

The **policy interface is immutable**.

See:  
```

docs/policy_contract.md

```

This document defines:
- Observation layout and normalization
- Action semantics
- Timing assumptions
- Default pose reference

Any mismatch between **code** and **contract** invalidates the policy.

---

## 7. Policy Semantics (Summary)

- **Observation dimension:** 48  
- **Action dimension:** 12  
- **Action meaning:** Δ joint position offsets  
- **Control mode:** Joint position PD (policy is PD-agnostic)

Final target computation (sim & real):

```

target_q = default_angles + action * action_scale

```

### Important Notes
- The policy **does not output torques**
- PD gains are applied **outside** the policy
- Hardware deployment **does not enforce action clipping**
  - Safety relies on `action_scale` and PD gains
- A neutral-action bias is subtracted at deployment time
  to compensate for posture bias in torque-trained policies

---

## 8. Control Architecture (Sim & Real)

```

Observations (48D)
↓
RL Policy (MLP, TorchScript)
↓
Optional safety transforms (deployment-specific)
↓
Joint target positions
↓
PD controller
↓
Motor torques (implicit)

```

- No direct torque commands are issued
- PD loop exists **outside** the policy
- PD gains are configurable via YAML

---

## 9. MuJoCo Simulation

### Primary Script
```

simulate/mujoco/v1/sim_walk_v1.0.py

```

### Experimental (Adds Logging Only)
```

simulate/mujoco/v1/sim_walk_v1.1.py

````

### Run
```
cd example/go2/low_level/simulate/mujoco/v1
python3 sim_walk_v1.0.py go2_sim_v1.0.yaml
````

### Notes on Simulation Behavior

* MuJoCo simulation includes **auxiliary stabilizers**
  (roll, yaw-rate, lateral velocity, yaw integral)
* These stabilizers are **not part of the policy**
* They are applied post-policy for numerical stability and inspection
* Hardware deployment does **not** use these stabilizers

Expected behavior:

* Symmetric standing posture
* Forward walking gait
* Mild lateral drift to the right over long horizons

This drift reflects **policy bias**, not a bug.

---

## 10. Isaac Gym Simulation (Legacy)

```
simulate/gym/v1/gym_walk_v1.0.py
```

Purpose:

* Legacy validation of a torque-trained policy
* PD-based initialization and stabilization
* **Not used for deployment**

⚠️ This script exists for **historical comparison only**
and should not be treated as the primary simulation pipeline.

---

## 11. Hardware Deployment (⚠️ Safety-Critical)

### Primary Script

```
deploy/v1/dep_walk_v1.0.py
```

### Experimental (Adds Logging)

```
deploy/v1/dep_walk_v1.1.py
```

---

### 11.1 Network Interface Selection

Identify the correct interface before deployment:

```
ifconfig
```

Example:

```
eno1: inet 192.168.123.222  netmask 255.255.255.0
```

❌ Using the wrong interface will cause DDS initialization failure.

---

### 11.2 Required Robot State (MANDATORY)

> **The robot must be SITTING before deployment.**

The deployment script:

* switches the robot into low-level mode
* zeros motor torques
* ramps into the default pose

🚫 Never start deployment from a standing robot
unless it is fully harnessed and lifted.

---

### 11.3 Run Deployment

```
cd example/go2/low_level/deploy/v1
python3 dep_walk_v1.0.py eno1 go2_dep_v1.0.yaml
```

---

### 11.4 Safe Shutdown

1. **Preferred**
   Press **SELECT** on the Unitree controller
   → robot enters damping mode and sits down

2. **Emergency**

   ```
   CTRL+C
   ```

Always:

* keep the robot harnessed
* operate in open space
* stand clear of the legs

---

## 12. Logging & Debugging

* `v1.1` scripts generate CSV logs
* Logs are used to:

  * inspect pose symmetry
  * analyze gait patterns
  * debug PD–policy interaction
* Logging does **not** alter control behavior

---

## 13. Known Sim-to-Real Gaps

Intentional mismatches include:

* Policy trained without hardware awareness
* MuJoCo provides idealized contacts and symmetry
* Hardware requires:

  * higher PD stiffness
  * careful damping
  * compensation for asymmetries

As a result:

* Standing is easier in simulation
* Hardware requires conservative tuning

Future versions (`v2.x`) will address this via
**hardware-informed training**.

---

## 14. Final Notes

* YAML files define **control behavior**
* Python scripts bind policy, control, and transport
* Policies are **external artifacts** and not tracked in git
* This stack prioritizes **safety, clarity, and inspectability**
  over raw performance

### Reference / Deprecated Scripts

* `deploy_policy.py`
  Deprecated, kept only for historical reference

* `go2_stand_example.py`
  Original Unitree SDK example, not part of this stack

* `path_utils.py`
  Centralized, deterministic path resolution utility

```

---
