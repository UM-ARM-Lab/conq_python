from typing import Callable

import numpy as np
from bosdyn import geometry
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder

from conq.clients import Clients
from conq.manipulation import add_follow_with_body, blocking_arm_command


def hand_pose_cmd(clients: Clients, x, y, z, roll=0., pitch=np.pi / 2, yaw=0., duration=0.5):
    """
    Move the arm to a pose relative to the body

    Args:
        clients: Clients
        x: x position in meters in front of the body center
        y: y position in meters to the left of the body center
        z: z position in meters above the body center
        roll: roll in radians
        pitch: pitch in radians
        yaw: yaw in radians
        duration: duration in seconds
    """
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot

    hand_pos_in_body = geometry_pb2.Vec3(x=x, y=y, z=z)

    euler = geometry.EulerZXY(roll=roll, pitch=pitch, yaw=yaw)
    quat_hand = euler.to_quaternion()

    body_in_odom = get_a_tform_b(transforms, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    hand_in_body = geometry_pb2.SE3Pose(position=hand_pos_in_body, rotation=quat_hand)

    hand_in_odom = body_in_odom * math_helpers.SE3Pose.from_proto(hand_in_body)

    arm_command = RobotCommandBuilder.arm_pose_command(
        hand_in_odom.x, hand_in_odom.y, hand_in_odom.z, hand_in_odom.rot.w, hand_in_odom.rot.x,
        hand_in_odom.rot.y, hand_in_odom.rot.z, VISION_FRAME_NAME, duration)
    return arm_command


def hand_delta_in_body_frame(clients: Clients, dx, dy, dz, follow=True):
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    hand_in_body = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    hand_pos_in_body = hand_in_body.position
    cmd = hand_pose_cmd(clients, hand_pos_in_body.x + dx, hand_pos_in_body.y + dy, hand_pos_in_body.z + dz)
    if follow:
        cmd = add_follow_with_body(cmd)
    blocking_arm_command(clients, cmd)


def randomized_look(clients: Clients, callback: Callable, x, y, z, pitch, yaw):
    dx = 0
    dy = 0
    dpitch = 0
    done = False
    while not done:
        look_cmd = hand_pose_cmd(clients, x + dx, y + dy, z, 0, pitch + dpitch, yaw,
                                 duration=1)
        blocking_arm_command(clients, look_cmd)
        done = callback()
        dx = np.random.randn() * 0.05
        dy = np.random.randn() * 0.05
        dpitch = np.random.randn() * 0.08
