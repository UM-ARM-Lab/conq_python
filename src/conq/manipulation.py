"""
Functions that make it easier to manipulate objects.
This primarily uses the manipulation_client, state_client, and command_client 
This includes functionality such as:
    - grasping a point in an image
"""

import time
from typing import List
import numpy as np
from bosdyn.api import manipulation_api_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.robot_command import block_until_arm_arrives, RobotCommandBuilder
from bosdyn.api import geometry_pb2
from conq.clients import Clients
from bosdyn.client import math_helpers


def blocking_arm_command(clients: Clients, cmd, timeout_sec=3):
    block_until_arm_arrives(clients.command, clients.command.robot_command(cmd), timeout_sec)
    # FIXME: why is this needed???
    clients.command.robot_command(RobotCommandBuilder.stop_command())


def block_for_manipulation_api_command(clients, cmd_response, period=0.25):
    """
    Block until the manipulation API command is done.

    Args:
        clients: Clients
        cmd_response: ManipulationApiCommandResponse
        period: float, seconds to wait between checking the command status
    """
    while True:
        time.sleep(period)
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)
        response = clients.manipulation.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)
        print(f'Current state: {state_name}')

        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            break

    print('Finished.')


def open_gripper(clients: Clients):
    clients.command.robot_command(RobotCommandBuilder.claw_gripper_open_command())
    time.sleep(1)  # FIXME: how to block on a gripper command?


HIGH_FORCE_THRESHOLD = 16
FORCE_BUFFER_SIZE = 15
MIN_NUM_FORCE_MEASUREMENTS = 5


def force_measure(clients: Clients, force_buffer: List):
    state = clients.state.get_robot_state()
    manip_state = state.manipulator_state
    force_reading = manip_state.estimated_end_effector_force_in_hand
    total_force = np.sqrt(force_reading.x ** 2 + force_reading.y ** 2 + force_reading.z ** 2)

    # circular buffer
    force_buffer.append(total_force)
    if len(force_buffer) > FORCE_BUFFER_SIZE:
        force_buffer.pop(0)
    recent_avg_total_force = float(np.mean(force_buffer))

    import rerun as rr
    rr.log_scalar("force/x", force_reading.x)
    rr.log_scalar("force/y", force_reading.y)
    rr.log_scalar("force/z", force_reading.z)
    rr.log_scalar("force/total", total_force)
    rr.log_scalar("force/recent_avg_total", recent_avg_total_force)
    rr.log_scalar("force/high_force_threshold", HIGH_FORCE_THRESHOLD)

    if recent_avg_total_force > HIGH_FORCE_THRESHOLD and len(force_buffer) > MIN_NUM_FORCE_MEASUREMENTS:
        print(f"large force detected! {recent_avg_total_force:.2f}")
        clients.command.robot_command(RobotCommandBuilder.stop_command())
        return True
    return False


def grasp_point_in_image(clients: Clients, image_res, pick_vec):
    """
    Args:
        clients: Clients object containing all clients for the robot
        image_res: ImageResponse obejct
        pick_vec: geometry_pb2.Vec2
    Returns:
        True if grasp succeeded, False otherwise
    """
    pick_cmd = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image_res.shot.transforms_snapshot,
        frame_name_image_sensor=image_res.shot.frame_name_image_sensor,
        camera_model=image_res.source.pinhole)

    return do_grasp_cmd(clients, pick_cmd)

# TODO: this is a duplicate of grasp_point_in_image
# WANT: functions that don't require a Clients object for everything. 
# We should break down our functions by what client they're interacting with.
# this function needs a manipulation client to move, and also a state client for feedback 
# (though arguably the feedback could be provided somewhere else, like in a higher-level state machine)
def grasp_point_in_image_basic(manipulation_client,state_client, image_response, pixel_xy, timeout=10):
    """
    Args:
        manipulation_client: ManipulationApiClient
        image_response: ImageResponse obejct
        pixel_xy: [x, y] in image coordinates
    Returns:
        True if grasp succeeded, False otherwise
    """
    # Construct the request
    pick_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
    pick_cmd = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
        frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
        camera_model=image_response.source.pinhole)
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick_cmd)
    
    # Send the request
    cmd_response = manipulation_client.manipulation_api_command(manipulation_api_request=grasp_request)

    # Get feedback and block until command is done
    t0 = time.time()
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)
        
        feedback_response = manipulation_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)
        
        print(f"Current State: {manipulation_api_pb2.ManipulationFeedbackState.Name(feedback_response.current_state)}")    

        if (feedback_response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
            or feedback_response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
            or feedback_response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
            or time.time() - t0 >= timeout):
            break
        
        time.sleep(0.25)

    if (feedback_response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED 
        and is_grasping(state_client)):
            print("Grasp Succeeded")
            return True
    else:
        print("Grasp failed")
        return False


def do_grasp_cmd(clients: Clients, pick_cmd: manipulation_api_pb2.PickObjectInImage, timeout=10):
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick_cmd)
    cmd_response = clients.manipulation.manipulation_api_command(manipulation_api_request=grasp_request)

    # execute grasp
    t0 = time.time()
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = clients.manipulation.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        print(manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            break
        elif response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            break
        elif response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION:
            break
        elif time.time() - t0 >= timeout:
            break

        time.sleep(1)

    is_grasping = get_is_grasping(clients)
    if is_grasping:
        print('Grasp succeeded')
        return True

    print('Grasp failed')
    open_gripper(clients)
    return False

# TODO: redundant to get_is_grasping
def is_grasping(state_client, percent_open_threshold=5):
    # Now check if we're actually holding something
    robot_state = state_client.get_robot_state()
    open_enough = robot_state.manipulator_state.gripper_open_percentage > percent_open_threshold
    # is_gripper_holding_item is NOT reliable
    is_grasping = robot_state.manipulator_state.is_gripper_holding_item and open_enough

    return is_grasping

def get_is_grasping(clients: Clients, percent_open_threshold=5):
    # Now check if we're actually holding something
    robot_state = clients.state.get_robot_state()
    open_enough = robot_state.manipulator_state.gripper_open_percentage > percent_open_threshold
    # is_gripper_holding_item is NOT reliable
    is_grasping = robot_state.manipulator_state.is_gripper_holding_item and open_enough
    return is_grasping


def hand_delta_z(clients: Clients, dz: float):
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    hand_in_body = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    raise_cmd = RobotCommandBuilder.arm_pose_command(hand_in_body.x, hand_in_body.y, hand_in_body.z + dz,
                                                     hand_in_body.rot.w, hand_in_body.rot.x, hand_in_body.rot.y,
                                                     hand_in_body.rot.z,
                                                     GRAV_ALIGNED_BODY_FRAME_NAME,
                                                     0.5)
    blocking_arm_command(clients, raise_cmd)


def add_follow_with_body(arm_command):
    """ Takes an arm command and combines it with a command to move the body to follow the arm """
    follow_cmd = RobotCommandBuilder.follow_arm_command()
    arm_and_follow_cmd = RobotCommandBuilder.build_synchro_command(follow_cmd, arm_command)
    return arm_and_follow_cmd

# TODO: Move this to a separate file that has to do with the command client
def move_gripper_to_pose(command_client, state_client, position_xyz, orientation_quat):
    """
    Move gripper to a given pose (in meters, wrt gravity-aligned robot frame)
    position_xyz: [x,y,z]
    orientation_quat: [w,x,y,z]

    """
    position_vec = geometry_pb2.Vec3(x=position_xyz[0], y=position_xyz[1], z=position_xyz[2])
    orientation = geometry_pb2.Quaternion(w=orientation_quat[0],
                                            x=orientation_quat[1],
                                            y=orientation_quat[2],
                                            z=orientation_quat[3])

    pose = geometry_pb2.SE3Pose(position=position_vec, rotation=orientation)

    robot_state = state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(pose)

    seconds = 2 # duration in seconds

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
        odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

    # Send the request
    cmd_id = command_client.robot_command(arm_command)
    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(command_client, cmd_id, 3.0)
