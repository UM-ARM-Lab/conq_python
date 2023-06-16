from typing import Callable

import numpy as np
from bosdyn import geometry
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder

from conq.exceptions import DetectionError
from conq.manipulation import blocking_arm_command
from regrasping_demo.get_detections import GetRetryResult


def look_at_command(robot_state_client, x, y, z, roll=0., pitch=np.pi / 2, yaw=0., duration=0.5):
    """
    Move the arm to a pose relative to the body

    Args:
        robot_state_client: RobotStateClient
        x: x position in meters in front of the body center
        y: y position in meters to the left of the body center
        z: z position in meters above the body center
        roll: roll in radians
        pitch: pitch in radians
        yaw: yaw in radians
        duration: duration in seconds
    """
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    hand_pos_in_body = geometry_pb2.Vec3(x=x, y=y, z=z)

    euler = geometry.EulerZXY(roll=roll, pitch=pitch, yaw=yaw)
    quat_hand = euler.to_quaternion()

    body_in_odom = get_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    hand_in_body = geometry_pb2.SE3Pose(position=hand_pos_in_body, rotation=quat_hand)

    hand_in_odom = body_in_odom * math_helpers.SE3Pose.from_proto(hand_in_body)

    arm_command = RobotCommandBuilder.arm_pose_command(
        hand_in_odom.x, hand_in_odom.y, hand_in_odom.z, hand_in_odom.rot.w, hand_in_odom.rot.x,
        hand_in_odom.rot.y, hand_in_odom.rot.z, ODOM_FRAME_NAME, duration)
    return arm_command


def look_at_scene(command_client, robot_state_client, x=0.56, y=0.1, z=0.55, pitch=0., yaw=0., dx=0., dy=0., dpitch=0.):
    look_cmd = look_at_command(robot_state_client, x + dx, y + dy, z,
                               0, pitch + dpitch, yaw,
                               duration=0.5)
    blocking_arm_command(command_client, look_cmd)


def get_point_f_retry(command_client, robot_state_client, image_client, get_point_f: Callable,
                      y, z, pitch, yaw, **get_point_f_kwargs) -> GetRetryResult:
    dx = 0
    dy = 0
    dpitch = 0
    while True:
        look_at_scene(command_client, robot_state_client, y=y, z=z, pitch=pitch, yaw=yaw, dx=dx, dy=dy, dpitch=dpitch)
        try:
            return get_point_f(image_client, **get_point_f_kwargs)
        except DetectionError:
            dx = np.random.randn() * 0.05
            dy = np.random.randn() * 0.05
            dpitch = np.random.randn() * 0.08
