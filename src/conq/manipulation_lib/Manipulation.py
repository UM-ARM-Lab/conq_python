# PYTHON
import time
import numpy as np
import rerun as rr


# BOSDYN: API
import bosdyn
from bosdyn.api import arm_command_pb2, estop_pb2, robot_command_pb2, synchronized_command_pb2
from google.protobuf import wrappers_pb2
from bosdyn.util import duration_to_seconds

# BOSDYN: Clients
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME, ODOM_FRAME_NAME, GROUND_PLANE_FRAME_NAME
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
# CONQ: Clients
from conq.clients import Clients
# CONQ: Wrappers
from conq.manipulation import blocking_arm_command, get_is_grasping, add_follow_with_body #, open_gripper
from conq.hand_motion import hand_pose_cmd_in_frame
from conq.manipulation import add_follow_with_body
from conq.hand_motion import hand_pose_cmd, hand_pose_cmd_to_vision
from conq.utils import setup_and_stand

# CONQ: Manipulation 
from conq.manipulation_lib.utils import make_robot_command, hand_pose_cmd_to_frame

def grasped_bool(clients: Clients):
    #FIXME: Needs force based instead of position based
    #FIXME: Replace with rob_state.client.get_rob_state().manipulation_state.is_gripper_holding_item
    is_grasping = get_is_grasping(clients)
    return is_grasping

def move_to_blocking(clients: Clients, pose, duration = 1, follow=False):
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

    Adapted from conq.manipulation:hand_pose_cmd
    """
    
    try:
        print("Arm command received")
        x,y,z,pitch,roll,yaw = pose
        arm_cmd = hand_pose_cmd(clients, x,y,z,roll,pitch,yaw,duration)
        follow = reachability(clients, pose)
        if follow: #TODO: determmined by reachability
            arm_cmd = add_follow_with_body(arm_cmd)
        blocking_arm_command(clients, arm_cmd)
        print("Arm command done")
        return True
    except Exception as e:
        print(e)
        return False
    
    
def move_to_unblocking(clients: Clients, pose, frame_name = GRAV_ALIGNED_BODY_FRAME_NAME, duration = 0.5, follow=False):
    """
    Move the arm to a pose relative to the body

    Args:
        clients: Clients
        x: x position in meters in front of the body center
        y: y position in meters to the left of the body center
        z: z position in meters above the body center
        qw: 
        qx: 
        qy: 
        qz:

        duration: duration in seconds
        frequency: rate of issuing command (useful for visual servoing)

    Adapted from conq.manipulation:hand_pose_cmd
    """
    
    try:
        #print("Arm command received")
        x,y,z,qw,qx,qy,qz = pose
        
        arm_command = RobotCommandBuilder.arm_pose_command(
            x, y, z, qw, qx,qy, qz, frame_name, duration,False)
        
        cmd_id = clients.command.robot_command_async(command = arm_command, end_time_secs=duration, timesync_endpoint=duration+0.25, lease=None)
        
        # TODO: Need to get feedback
        # TODO: Check for reachability

        #print("Arm command done")
        return True
    except Exception as e:
        print(e)
        return False
    
    
def open_gripper(clients: Clients):
    try:
        clients.command.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(1)  # FIXME: how to block on a gripper command?
        return True
    except:
        return False
    
def close_gripper(clients: Clients):
    try:
        clients.command.robot_command(RobotCommandBuilder.claw_gripper_close_command())
        time.sleep(1)  # FIXME: how to block on a gripper command?
        return True
    except:
        return False

def move_joint_trajectory(clients: Clients,joint_targets, max_vel=2.5, max_acc=15, times=[2.0,4.0]):
        """
        Function to execute joint trajectory
        Args:
        - clients: Clients
        - joint_targets: List of [tuple(joint)]
        - max_vel: Maximum velocity for all the joints
        - max_acc: Maximum acceleration constraint for all the joints
        - time (list): Reach trajectory at time step
        """
        print("Entered joint trajectory execution")
        max_vel = wrappers_pb2.DoubleValue(value=max_vel)
        max_acc = wrappers_pb2.DoubleValue(value=max_acc)
        traj_list = []
        duration = 0
        for idx,joints in enumerate(joint_targets):
            sh0,sh1,el0,el1,wr0,wr1 = joints
            traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
                sh0, sh1, el0, el1, wr0, wr1, times[idx])
            traj_list.append(traj_point)
            duration+= times[idx]
        print("Created traj list")
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=traj_list,
                                                            maximum_velocity=max_vel,
                                                            maximum_acceleration=max_acc)
        # Make a RobotCommand
        command = make_robot_command(arm_joint_traj)

        # Send the request
        cmd_id = clients.command.robot_command(command)
        block_until_arm_arrives(clients.command, cmd_id, duration+2.0)

        return cmd_id

# TODO: Move to utils
def print_feedback(feedback_resp, logger):
    """ Helper function to query for ArmJointMove feedback, and print it to the console.
        Returns the time_to_goal value reported in the feedback """
    
    joint_move_feedback = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_joint_move_feedback
    logger.info(f'  planner_status = {joint_move_feedback.planner_status}')
    logger.info(
        f'  time_to_goal = {duration_to_seconds(joint_move_feedback.time_to_goal):.2f} seconds.')

    # Query planned_points to determine target pose of arm
    logger.info('  planned_points:')
    for idx, points in enumerate(joint_move_feedback.planned_points):
        pos = points.position
        pos_str = f'sh0 = {pos.sh0.value:.3f}, sh1 = {pos.sh1.value:.3f}, el0 = {pos.el0.value:.3f}, ' \
                  f'el1 = {pos.el1.value:.3f}, wr0 = {pos.wr0.value:.3f}, wr1 = {pos.wr1.value:.3f}'
        logger.info(f'    {idx}: {pos_str}')
    return duration_to_seconds(joint_move_feedback.time_to_goal)
    
def follow_cart_traj(clients: Clients, pose, duration):
    """
    Input -> pose (list of tuples)
    """
    # TODO: Add constraints in future with configs
    pass

def reachability(client: Clients, pose):
    """
    Check reachibility of for the given pose when robot is in fixed standing pose
    Useful to decide when the robot cannot reach, hence allow robot to move closer
    True: Reachable
    False: Unreachable
    """
    # TODO: Implement reachability following sdk examples: inverse kinematics
    return True

def get_camera_intrinsics(image_proto):
    """
    Get camera instrinsics
    """
    focal_x = image_proto.source.pinhole.intrinsics.focal_length.x
    principal_x = image_proto.source.pinhole.intrinsics.principal_point.x

    focal_y = image_proto.source.pinhole.intrinsics.focal_length.y
    principal_y = image_proto.source.pinhole.intrinsics.principal_point.y

    return [focal_x,focal_y, principal_x, principal_y]

def get_gpe_in_cam(rgb_res, clients: Clients):
    transforms_hand = rgb_res.shot.transforms_snapshot
    transforms_body = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    odon_in_cam = get_a_tform_b(transforms_hand, rgb_res.shot.frame_name_image_sensor, VISION_FRAME_NAME)
    gpe_in_odom = get_a_tform_b(transforms_body, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    gpe_in_cam = odon_in_cam * gpe_in_odom
    return gpe_in_cam

