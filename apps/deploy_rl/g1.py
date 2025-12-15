#!/usr/bin/env python3

import os
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.timer import Timer
import math
import yaml

from sensor_msgs.msg import Joy
from unitree_go.msg import SportModeState
from unitree_hg.msg import LowCmd, LowState, IMUState, MotorState

from huro_py.crc_hg import Crc
from huro_py.utils import quat_rotate_inverse

G1_NUM_MOTOR = 29

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


class G1PolicyRunner(Node):
    def __init__(self):
        super().__init__("g1_policy_runner")

        self.control_freq_hz = 50  # 10ms
        self.control_dt = 1.0 / self.control_freq_hz
        self.timer_dt_ms = int((self.control_dt) * 1000)
        self.time = 0.0
        self.init_duration_s = 3.0

        self.mode_ = Mode.PR
        self.mode_machine = 0

        self.run_policy = False
        self.motors_on = 1

        share = get_package_share_directory("huro")
        # Load yaml config
        yaml_name = "g1_actuators.yaml"
        yaml_path = os.path.join(share, "resources", "models", "g1", yaml_name)
        with open(yaml_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        # Load policy model params
        obs_norm_name = "g1_actor_obs_normalizer.pt"
        policy_name = "g1_actor.pt"

        policy_path = os.path.join(share, "resources", "models", "g1", policy_name)
        obs_norm_path = os.path.join(share, "resources", "models", "g1", obs_norm_name)
        self.device = torch.device("cpu")
        for path in [policy_path, obs_norm_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found at: {path}")

        self.obs_norm = torch.jit.load(obs_norm_path, map_location=self.device)
        self.obs_norm.eval()
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.get_logger().info("Policy loaded successfully")

        self.odom_state = SportModeState()
        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(len(JOINT_NAMES))]
        self.joystick = Joy()
        self.prev_actions = np.zeros(len(JOINT_NAMES))

        # Publishers
        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)

        # Subscribers
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )
        self.odommodestate_sub = self.create_subscription(
            SportModeState, "/odommodestate", self.odom_handler, 10
        )
        self.joystick_sub = self.create_subscription(Joy, "/joy", self.joy_handler, 10)

        self.timer = self.create_timer(self.control_dt, self.control)

    def control(self):
        low_cmd = LowCmd()
        self.time += self.control_dt

        low_cmd.mode_pr = self.mode_
        low_cmd.mode_machine = self.mode_machine

        if self.time < self.init_duration_s or not self.run_policy:
            for name in JOINT_NAMES:
                idx = self.cfg[name]["index"]
                q_init = self.cfg[name]["default_position"]
                ratio = self.clamp(self.time / self.init_duration_s, 0.0, 1.0)
                cmd = low_cmd.motor_cmd[idx]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - ratio) * self.motor[idx].q + ratio * q_init
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = self.cfg[name]["stiffness"]
                cmd.kd = self.cfg[name]["damping"]
        else:  # If A (or cross) is pressed, run policy
            # Get observations
            obs = self.get_obs()
            # Infer policy
            with torch.no_grad():
                obs = self.get_obs()
                obs = self.obs_norm(obs)
                actions = self.policy(obs)
            actions = actions.squeeze(0).cpu().numpy()
            self.prev_actions = actions.copy()
            # Command robot
            for name in JOINT_NAMES:
                idx = self.cfg[name]["index"]
                q_init = self.cfg[name]["default_position"]
                action_scale = self.cfg[name]["action_scale"]
                cmd = low_cmd.motor_cmd[idx]
                cmd.mode = self.motors_on
                cmd.q = q_init + action_scale * actions[idx]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = self.cfg[name]["stiffness"]
                cmd.kd = self.cfg[name]["damping"]

        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

    def get_obs(self):
        # Base linear velocity [3]
        # Base angular velocity [3]
        # Proj grav [4]
        # Joint pos [29]
        # Joint vel [29]
        # Actions [29]
        # Command [3]

        # proj_j = self.projected_gravity()
        quat = np.array(
            [
                self.imu.quaternion[0],  # w
                self.imu.quaternion[1],  # x
                self.imu.quaternion[2],  # y
                self.imu.quaternion[3],  # z
            ]
        )
        gravity_world = np.array([0.0, 0.0, -1.0])
        proj_g = quat_rotate_inverse(quat, gravity_world)

        q = np.zeros(len(JOINT_NAMES))
        dq = np.zeros(len(JOINT_NAMES))
        for name in JOINT_NAMES:
            idx = self.cfg[name]["index"]
            q[idx] = self.motor[idx].q
            dq[idx] = self.motor[idx].dq

        command = np.array([0.0, 0.0, 0.0])

        obs = torch.cat(
            [
                torch.tensor(self.odom_state.velocity, dtype=torch.float),
                torch.tensor(self.imu.gyroscope, dtype=torch.float),
                torch.tensor(proj_g, dtype=torch.float),
                torch.tensor(q, dtype=torch.float),
                torch.tensor(dq, dtype=torch.float),
                torch.tensor(self.prev_actions, dtype=torch.float),
                torch.tensor(command, dtype=torch.float),
            ],
            axis=0,
        )

        return obs

    def low_state_handler(self, msg: LowState):
        self.mode_machine = msg.mode_machine
        self.imu = msg.imu_state
        for name in JOINT_NAMES:
            idx = self.cfg[name]["index"]
            self.motor[idx] = msg.motor_state[idx]

    def odom_handler(self, msg: SportModeState):
        self.odom_state = msg

    def joy_handler(self, msg: Joy):
        self.joystick = msg
        if msg.buttons[0] == 1:
            self.run_policy = True

    def clamp(self, value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value


def main(args=None):
    rclpy.init(args=args)
    node = G1PolicyRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
