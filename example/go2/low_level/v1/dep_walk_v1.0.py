#!/usr/bin/env python3
"""
Bare Bones Deployment Script for Go2 using a pre-trained policy:
- No stabilization or state estimation; direct policy inference.
- Joint position commands with PD control.
- No logging or data saving.

CLI USAGE:
python3 dep_walk_v1.0.py eno1 go2_dep_v1.0.yaml
"""

import sys, os, time, argparse, numpy as np, torch, yaml, subprocess

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_ as LowCmdGo,
    unitree_go_msg_dds__LowState_ as LowStateGo,
)

# -----------------------------------------------------------------------------
# Ensure low_level root is on PYTHONPATH
# -----------------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
LOW_LEVEL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(THIS_FILE), "..", "..")
)

if LOW_LEVEL_ROOT not in sys.path:
    sys.path.insert(0, LOW_LEVEL_ROOT)

# -----------------------------------------------------------------------------
# Canonical path helpers
# -----------------------------------------------------------------------------
from common.path_utils import (
    get_policy_path,
    get_deploy_config_path,
    get_project_root,
)

# -----------------------------------------------------------------------------
# Local helpers
# -----------------------------------------------------------------------------
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go
from common.remote_controller import RemoteController, KeyMap
from common.rotation_helper import get_gravity_orientation


class Go2Deployer:
    """Raw policy deployment, training-consistent."""

    # SIM → HW joint index map (legged_gym → Unitree)
    SIM_TO_HW = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    def __init__(self, net_if: str, cfg_name: str):

        # ------------------------------------------------------------------
        # Mode switch (MANDATORY)
        # ------------------------------------------------------------------
        project_root = get_project_root(__file__)
        mode_switch_path = os.path.join(project_root, "debug", "mode_switch.py")
        subprocess.run([sys.executable, mode_switch_path, net_if], check=True)

        # ------------------------------------------------------------------
        # Load deployment config (versioned, deterministic)
        # ------------------------------------------------------------------
        cfg_path = get_deploy_config_path(__file__, "v1", cfg_name)
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.control_dt = float(self.cfg.get("control_dt", 0.02))

        # ------------------------------------------------------------------
        # Policy
        # ------------------------------------------------------------------
        self.policy_path = get_policy_path(
            __file__,
            self.cfg.get("policy", "policy_v1.pt")
        )

        self.policy = torch.jit.load(self.policy_path)

        # ------------------------------------------------------------------
        # Dimensions
        # ------------------------------------------------------------------
        self.num_actions = int(self.cfg.get("num_actions", 12))
        self.num_obs = int(self.cfg.get("num_obs", 48))

        # ------------------------------------------------------------------
        # Gains & posture
        # ------------------------------------------------------------------
        self.kps = np.array(self.cfg["kps"], dtype=np.float32)
        self.kds = np.array(self.cfg["kds"], dtype=np.float32)

        self.default_angles = np.array(self.cfg["default_angles"], dtype=np.float32)
        self.default_hw = self.default_angles[self.SIM_TO_HW]

        # ------------------------------------------------------------------
        # Scales
        # ------------------------------------------------------------------
        self.lin_vel_scale = float(self.cfg.get("lin_vel_scale", 2.0))
        self.ang_vel_scale = float(self.cfg.get("ang_vel_scale", 0.25))
        self.dof_pos_scale = float(self.cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(self.cfg.get("dof_vel_scale", 0.05))
        self.action_scale = float(self.cfg.get("action_scale", 0.25))
        self.cmd_scale = np.array(
            self.cfg.get("cmd_scale", [1.0, 1.0, 0.25]),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # DDS setup
        # ------------------------------------------------------------------
        ChannelFactoryInitialize(0, net_if)

        self.low_cmd = LowCmdGo()
        self.low_state = LowStateGo()

        self.pub = ChannelPublisher("rt/lowcmd", type(self.low_cmd))
        self.pub.Init()

        self.sub = ChannelSubscriber("rt/lowstate", type(self.low_state))
        self.sub.Init(self._on_lowstate, 10)

        init_cmd_go(self.low_cmd, weak_motor=[])

        # ------------------------------------------------------------------
        # Buffers
        # ------------------------------------------------------------------
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        self.remote = RemoteController()

        # ------------------------------------------------------------------
        # Policy neutral offset (standing bias)
        # ------------------------------------------------------------------
        dummy_obs = np.zeros(self.num_obs, dtype=np.float32)
        with torch.no_grad():
            self.policy_neutral_offset = (
                self.policy(torch.from_numpy(dummy_obs).unsqueeze(0))
                .cpu()
                .numpy()
                .squeeze()
            )

        self._wait_for_state()

    # ------------------------------------------------------------------
    def _on_lowstate(self, msg: LowStateGo):
        self.low_state = msg
        self.remote.set(self.low_state.wireless_remote)

    def _wait_for_state(self):
        print("[INFO] Waiting for robot state...")
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("[INFO] Connected to Go2 state stream.")

    def _send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # ------------------------------------------------------------------
    def zero_torque_state(self):
        print("[STATE] Zero Torque — Press START to continue.")
        while self.remote.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self._send_cmd()
            time.sleep(self.control_dt)

    def move_to_default(self):
        print("[STATE] Moving to default position (2 s ramp)...")
        steps = int(2.0 / self.control_dt)
        init_q = np.array(
            [self.low_state.motor_state[i].q for i in range(self.num_actions)],
            dtype=np.float32,
        )
        for step in range(steps):
            alpha = step / steps
            q_targets = (1 - alpha) * init_q + alpha * self.default_hw
            for i in range(self.num_actions):
                self.low_cmd.motor_cmd[i].q = q_targets[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self._send_cmd()
            time.sleep(self.control_dt)
        print("[STATE] Default stance reached ✅")

    def hold_default(self):
        print("[STATE] Holding default position — Press A to start policy.")
        while self.remote.button[KeyMap.A] != 1:
            for i in range(self.num_actions):
                self.low_cmd.motor_cmd[i].q = self.default_hw[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self._send_cmd()
            time.sleep(self.control_dt)
        print("[STATE] A pressed — running policy...")

    # ------------------------------------------------------------------
    def run_once(self):
        for i in range(self.num_actions):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        self.qj = self.qj[self.SIM_TO_HW]
        self.dqj = self.dqj[self.SIM_TO_HW]

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        g_vec = get_gravity_orientation(quat).astype(np.float32)

        cmd = np.array(
            [self.remote.ly, -self.remote.lx, -self.remote.rx],
            dtype=np.float32,
        )
        cmd_scaled = cmd * self.cmd_scale

        self.obs = np.concatenate(
            [
                np.zeros(3) * self.lin_vel_scale,
                ang_vel * self.ang_vel_scale,
                g_vec,
                cmd_scaled,
                (self.qj - self.default_angles) * self.dof_pos_scale,
                self.dqj * self.dof_vel_scale,
                self.action,
            ],
            dtype=np.float32,
        )[: self.num_obs]

        with torch.no_grad():
            act = (
                self.policy(torch.from_numpy(self.obs).unsqueeze(0))
                .cpu()
                .numpy()
                .squeeze()
            )

        act -= 0.6 * self.policy_neutral_offset
        self.action = act

        target = (self.default_angles + act * self.action_scale)[self.SIM_TO_HW]

        for i in range(self.num_actions):
            self.low_cmd.motor_cmd[i].q = float(target[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kps[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

        self._send_cmd()
        time.sleep(self.control_dt)

    # ------------------------------------------------------------------
    def shutdown(self):
        create_damping_cmd(self.low_cmd)
        self._send_cmd()
        print("[INFO] Damping engaged — shutdown complete.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("net_if", type=str)
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    d = Go2Deployer(args.net_if, args.config)

    try:
        d.zero_torque_state()
        d.move_to_default()
        d.hold_default()
        while d.remote.button[KeyMap.select] != 1:
            d.run_once()
    finally:
        d.shutdown()
