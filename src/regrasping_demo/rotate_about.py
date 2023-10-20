import time

import numpy as np
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_se2_a_tform_b, VISION_FRAME_NAME, HAND_FRAME_NAME, \
    GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_for_trajectory_cmd

from conq.cameras_utils import rot_2d
from conq.clients import Clients


def rotate_around_point_in_hand_frame(clients: Clients, pos: np.ndarray, angle: float):
    """
    Moves the body by `angle` degrees, and translate so that `pos` stays in the same place.

    Assumptions:
     - the hand and the body have aligned X axes

    Args:
        clients: clients
        pos: 2D position of the point to rotate around, in the hand frame
        angle: angle in radians to rotate the body by, around the Z axis
    """
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    hand_in_odom = get_se2_a_tform_b(transforms, VISION_FRAME_NAME, HAND_FRAME_NAME)
    hand_in_body = get_se2_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    body_in_hand = hand_in_body.inverse()  # NOTE: querying frames in opposite order returns None???
    body_pt_in_hand = np.array([body_in_hand.x, body_in_hand.y])
    rotated_body_pos_in_hand = rot_2d(angle) @ body_pt_in_hand + pos
    rotated_body_in_hand = math_helpers.SE2Pose(rotated_body_pos_in_hand[0], rotated_body_pos_in_hand[1],
                                                angle + body_in_hand.angle)
    goal_in_odom = hand_in_odom * rotated_body_in_hand
    se2_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=goal_in_odom.to_proto(),
                                                                 frame_name=VISION_FRAME_NAME,
                                                                 locomotion_hint=spot_command_pb2.HINT_CRAWL)
    se2_synchro_cmd = RobotCommandBuilder.build_synchro_command(se2_cmd)
    se2_cmd_id = clients.command.robot_command(lease=None, command=se2_synchro_cmd,
                                               end_time_secs=time.time() + 999)
    block_for_trajectory_cmd(clients.command, se2_cmd_id)
