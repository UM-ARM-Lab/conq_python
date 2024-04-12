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
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives, blocking_stand
from conq.clients import Clients
from bosdyn.api import image_pb2
from bosdyn.client.image import _depth_image_data_to_numpy, _depth_image_get_valid_indices

def build_arm_target_from_vision(clients, visual_pose):
    x_hand,y_hand,z_hand,roll_hand, pitch_hand, yaw_hand = visual_pose
    roll_hand, pitch_hand, yaw_hand = 0,0,0 # FIXME: Make this generalized
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

def unstow_arm(robot, command_client):
    stand_command = RobotCommandBuilder.synchro_stand_command() # params
    unstow = RobotCommandBuilder.arm_ready_command(build_on_command=stand_command)
    unstow_command_id = command_client.robot_command(unstow)
    robot.logger.info('Unstow command issued.')
    block_until_arm_arrives(command_client, unstow_command_id, 3.0)


def stow_arm(robot, command_client):
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    stow_command_id = command_client.robot_command(stow_cmd)
    robot.logger.info('Stow command issued.')
    block_until_arm_arrives(command_client, stow_command_id, 3.0)

def stand(robot, command_client):
    robot.logger.info('Commanding robot to stand...')
    blocking_stand(command_client, timeout_sec=10)
    robot.logger.info('Robot standing.')

#### PERCEPTION UTILS ####

def depth2pcl(image_response, seg_mask = None, min_dist=0, max_dist=1000):
     """
     Convert depth image to point cloud.
     Modified from bosdyn.clients.image
     """
     if image_response.source.image_type != image_pb2.ImageSource.IMAGE_TYPE_DEPTH:
         raise ValueError('requires an image_type of IMAGE_TYPE_DEPTH')
     if image_response.source.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
         raise ValueError('IMAGE_TYPE_DEPTH with an unsupported format, required PIXEL_FORMAT_DEPTH_U16')
     if not image_response.source.HasField('pinhole'):
         raise ValueError('Requires a pinhole camera model')
     
     source_rows = image_response.source.rows
     source_cols = image_response.source.cols
     fx = image_response.source.pinhole.intrinsics.focal_length.x
     fy = image_response.source.pinhole.intrinsics.focal_length.y
     cx = image_response.source.pinhole.intrinsics.principal_point.x
     cy = image_response.source.pinhole.intrinsics.principal_point.y
     depth_scale = image_response.source.depth_scale

     # Convert the proto representation into a numpy array
     depth_array = _depth_image_data_to_numpy(image_response)

     # SEGMENT if seg_mask is not None:
     if seg_mask is not None:
         depth_array[~seg_mask] = 0  # Set non-segmented regions to zero

     # Determine which indices have valid data in the user requested range
     valid_inds = _depth_image_get_valid_indices(depth_array, np.rint(min_dist*depth_scale),
                                                 np.rint(max_dist * depth_scale))
     
     # Compute the valid data
     rows, cols = np.mgrid[0:source_rows, 0:source_cols]
     depth_array = depth_array[valid_inds]
     rows = rows[valid_inds]
     cols = cols[valid_inds]

     # Convert the valid distance data to (x,y,z) values expressed in the sensor frame
     z = depth_array / depth_scale
     x = np.multiply(z, (cols - cx)) / fx
     y = np.multiply(z, (rows - cy)) / fy
     return np.vstack((x,y,z)).T

def pcl_transform(pcl_xyz, image_response, source,target_frame):
    pcl_xyz1 = np.hstack((pcl_xyz,np.ones((pcl_xyz.shape[0],1))))
    # TRANSFORM FROM sensor frame to GRAV_ALIGNED frame
    BODY_T_VISION = get_a_tform_b(
        image_response.shot.transforms_snapshot, target_frame, source).to_matrix() # 4x4

    body_pcl_hand_sensor = np.dot(BODY_T_VISION,pcl_xyz1.T).T # Nx4
    # Remove ones
    body_pcl_hand_sensor = body_pcl_hand_sensor[:,:-1] #/body_pcl_hand_sensor[:,-1][:,np.newaxis]

    return body_pcl_hand_sensor


