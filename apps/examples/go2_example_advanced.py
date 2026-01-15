#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
import math

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, IMUState, MotorState

from huro_py.crc_go import Crc

# Custom PRorAB enum or constant
# from hucebot_g1_ros.msg import MotorMode  # Assuming PRorAB is defined here

# from hucebot_g1_ros.config import GO2_NUM_MOTOR, Kp, Kd
# from hucebot_g1_ros.motor_crc_hg import get_crc  # Assuming Python version available

GO2_NUM_MOTOR = 12

Kp = [
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
    60.0,
]

Kd = [
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
]

targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]


class MoveExample(Node):
    def __init__(self):
        super().__init__("move_example")

        self.control_dt = 0.002  # 2ms (500Hz like the example)
        self.timer_dt_ms = int(self.control_dt * 1000)
        self.time = 0.0
        
        # Phase durations (in number of control steps)
        self.duration_1 = 500  # Move from start to position 1
        self.duration_2 = 500  # Move from position 1 to position 2
        self.duration_3 = 1000  # Hold position 2
        self.duration_4 = 900  # Move from position 2 to position 3
        
        # Phase completion tracking
        self.percent_1 = 0.0
        self.percent_2 = 0.0
        self.percent_3 = 0.0
        self.percent_4 = 0.0
        
        self.startPos = [0.0] * 12
        self.firstRun = True
        self.received_state = False

        self.motors_on = 1

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(GO2_NUM_MOTOR)]

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )

        self.sport_pub = self.create_publisher(Request, "/api/sport/request", 10)
        ROBOT_SPORT_API_ID_STANDDOWN = 1005
        req = Request()
        req.header.identity.api_id = ROBOT_SPORT_API_ID_STANDDOWN
        self.sport_pub.publish(req)

        self.motion_pub = self.create_publisher(
            Request, "/api/motion_switcher/request", 10
        )
        ROBOT_MOTION_SWITCHER_API_RELEASEMODE = 1003
        req = Request()
        req.header.identity.api_id = ROBOT_MOTION_SWITCHER_API_RELEASEMODE
        self.motion_pub.publish(req)

        self.timer = self.create_timer(self.control_dt, self.control)

    def control(self):
        # Wait until we receive actual robot state
        if not self.received_state:
            return
            
        low_cmd = LowCmd()
        low_cmd.head[0] = 0xFE
        low_cmd.head[1] = 0xEF
        low_cmd.gpio = 0

        self.time += self.control_dt

        # Initialize start positions on first run
        if self.firstRun:
            for i in range(GO2_NUM_MOTOR):
                self.startPos[i] = self.motor[i].q
            self.firstRun = False
            self.get_logger().info("Starting position sequence...")
            self.get_logger().info(f"Start positions: {self.startPos}")

        # Phase 1: Move from start position to targetPos_1
        if self.percent_1 < 1.0:
            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1.0)
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - self.percent_1) * self.startPos[i] + self.percent_1 * targetPos_1[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        # Phase 2: Move from targetPos_1 to targetPos_2
        elif self.percent_1 == 1.0 and self.percent_2 < 1.0:
            if self.percent_2 == 0.0:
                self.get_logger().info("Phase 1 complete, moving to position 2...")
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1.0)
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - self.percent_2) * targetPos_1[i] + self.percent_2 * targetPos_2[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        # Phase 3: Hold at targetPos_2
        elif self.percent_1 == 1.0 and self.percent_2 == 1.0 and self.percent_3 < 1.0:
            if self.percent_3 == 0.0:
                self.get_logger().info("Phase 2 complete, holding position 2...")
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1.0)
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = targetPos_2[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        # Phase 4: Move from targetPos_2 to targetPos_3
        elif self.percent_1 == 1.0 and self.percent_2 == 1.0 and self.percent_3 == 1.0 and self.percent_4 < 1.0:
            if self.percent_4 == 0.0:
                self.get_logger().info("Phase 3 complete, moving to position 3...")
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1.0)
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - self.percent_4) * targetPos_2[i] + self.percent_4 * targetPos_3[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        # Final: Hold at targetPos_3
        else:
            if self.percent_4 == 1.0:
                self.get_logger().info("All phases complete! Holding position 3.")
                self.percent_4 = 1.01  # Prevent repeated logging
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = targetPos_3[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

    def low_state_handler(self, msg: LowState):
        # self.get_logger().info(str(self.motors_on))
        self.imu = msg.imu_state
        for i in range(GO2_NUM_MOTOR):
            self.motor[i] = msg.motor_state[i]
        
        # Mark that we've received state data
        if not self.received_state:
            self.received_state = True
            self.get_logger().info("Received robot state, starting control...")

        ## Handle Controller Message
        self.controller_msg = msg.wireless_remote
        if self.controller_msg[3] == 1:
            self.motors_on = 0

    def clamp(self, value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value


def main(args=None):
    rclpy.init(args=args)
    node = MoveExample()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
