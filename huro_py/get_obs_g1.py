#!/usr/bin/env python3

from unitree_go.msg import LowState, SportModeState
from huro.msg import SpaceMouseState

import numpy as np
import yaml
import os
from huro_py.utils import quat_rotate_inverse
NUM_ACTIONS = 12

def get_obs_low_state(lowstate_msg: LowState, spacemouse_msg: SpaceMouseState, height: float, prev_actions: np.array, phase: float, default_pos: list):
    """
    Extract observations from LowState message for RL policy.
    
    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands
    
    Returns:
        obs: numpy array of shape (49,) with observation vector
        
    Observation structure (49 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU) 
    - obs[3:7]   : Gravity direction (from IMU)
    - obs[7:10]  : Command velocity (x, y, yaw)
    - obs[10]    : Height command
    - obs[11:23] : Joint positions relative to default (12 joints)
    - obs[23:35] : Joint velocities (12 joints)
    - obs[35:47] : Previous actions (12 values)
    """
    
    
    # MAPPING ROBOT -> POLICY
    
    motor_states = lowstate_msg.motor_state[:12]
    
    current_joint_pos_sdk = np.array([motor_states[i].q for i in range(NUM_ACTIONS)])
    current_joint_vel_sdk = np.array([motor_states[i].dq for i in range(NUM_ACTIONS)])

    # FILLING OBS VECTOR


    obs = np.zeros(47)
    
    # lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
        
    # Base linear velocity (obs[0:3])
    
    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = np.array([
        lowstate_msg.imu_state.gyroscope[0],
        lowstate_msg.imu_state.gyroscope[1],
        lowstate_msg.imu_state.gyroscope[2]
    ]) * ang_vel_scale
    # Computing projected gravity from IMU sensor
    quat = np.array([
        lowstate_msg.imu_state.quaternion[0],  # w
        lowstate_msg.imu_state.quaternion[1],  # x
        lowstate_msg.imu_state.quaternion[2],  # y
        lowstate_msg.imu_state.quaternion[3]   # z
    ])
    # Normalize quaternion to prevent drift
  
    
    gravity_world = np.array([0.0, 0.0, -1.0])

    gravity_b = quat_rotate_inverse(quat,gravity_world)
    # gravity_b[0] *= 2.0
    # gravity_b[1] *= 2.0
    obs[3:6] = gravity_b
    # Command velocity (obs[9:12]) - default to zero (forward, lateral, yaw rate)
    obs[6:9] = [spacemouse_msg.twist.angular.y / 2, -spacemouse_msg.twist.angular.x / 2, spacemouse_msg.twist.angular.z / 2]
    # Fill joint positions (obs[13:25]) in policy order
    obs[9:21] = (current_joint_pos_sdk - np.array(default_pos)) * dof_pos_scale
    # Fill joint velocities (obs[25:37]) in policy order
    obs[21:33] = current_joint_vel_sdk * dof_vel_scale
    # Previous actions (obs[37:49]) - default to zero
    obs[33:45] = prev_actions
    obs[45] = np.sin(2.0 * np.pi * phase)
    obs[46] = np.cos(2.0 * np.pi * phase)
        
    return obs
