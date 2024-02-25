# BOSTON DYNAMICS API
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2

import numpy as np
from bosdyn import geometry
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.estop import EstopClient
from bosdyn.api import arm_command_pb2, estop_pb2, robot_command_pb2, synchronized_command_pb2

from conq.clients import Clients

def build_arm_target_from_vision(clients, visual_pose):
    x_hand,y_hand,z_hand,roll_hand, pitch_hand, yaw_hand = visual_pose
    # 2. TRANSFORM from Hand pose to Grav_aligned_body pose
    # TODO: Convert euler to rot
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    # Transformation of hand w.r.t grav_aligned
    BODY_T_HAND = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    
    xyz = geometry_pb2.Vec3(x = x_hand, y=y_hand,z=z_hand)
    euler = geometry.EulerZXY(roll = roll_hand, pitch = pitch_hand, yaw = yaw_hand)
    quat = euler.to_quaternion()
    HAND_T_OBJECT = geometry_pb2.SE3Pose(position = xyz, rotation = quat)
    BODY_T_OBJECT = BODY_T_HAND * math_helpers.SE3Pose.from_proto(HAND_T_OBJECT)

    #print("Object pose in Body frame",BODY_T_OBJECT)
    arm_command_pose = (BODY_T_OBJECT.position.x,BODY_T_OBJECT.position.y,BODY_T_OBJECT.position.z,BODY_T_OBJECT.rotation.w,BODY_T_OBJECT.rotation.x, BODY_T_OBJECT.rotation.y,BODY_T_OBJECT.rotation.z)
    return arm_command_pose

def make_robot_command(arm_joint_traj):
    """ Adapted from sdk example. Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)

def hand_pose_cmd_to_frame(a_in_frame, x, y, z, roll, pitch, yaw):
    hand_pos_in_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    euler = geometry.EulerZXY(roll=roll, pitch=pitch, yaw=yaw)
    quat_hand = euler.to_quaternion()
    hand_in_body = geometry_pb2.SE3Pose(position=hand_pos_in_body, rotation=quat_hand)
    hand_in_vision = a_in_frame * math_helpers.SE3Pose.from_proto(hand_in_body)
    return hand_in_vision

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)