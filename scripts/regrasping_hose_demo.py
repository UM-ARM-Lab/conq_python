import argparse
import sys
import time
from typing import List, Callable

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rerun as rr
from bosdyn import geometry
from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b, \
    HAND_FRAME_NAME
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, block_for_trajectory_cmd)
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import wrappers_pb2

from bosdyn.client.lease import LeaseKeepAlive
from regrasping_demo.detect_regrasp_point import DetectionError, min_angle_to_x_axis
from regrasping_demo.get_detections import GetRetryResult, get_hose_and_head_point, get_hose_and_regrasp_point, get_mess
from conq.utils import blocking_arm_command, block_for_manipulation_api_command, setup_and_stand, rot_2d, open_gripper

HIGH_FORCE_THRESHOLD = 16
FORCE_BUFFER_SIZE = 15


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


def force_measure(state_client, command_client, force_buffer: List):
    state = state_client.get_robot_state()
    manip_state = state.manipulator_state
    force_reading = manip_state.estimated_end_effector_force_in_hand
    total_force = np.sqrt(force_reading.x ** 2 + force_reading.y ** 2 + force_reading.z ** 2)

    # circular buffer
    force_buffer.append(total_force)
    if len(force_buffer) > FORCE_BUFFER_SIZE:
        force_buffer.pop(0)
    recent_avg_total_force = float(np.mean(force_buffer))

    rr.log_scalar("force/x", force_reading.x)
    rr.log_scalar("force/y", force_reading.y)
    rr.log_scalar("force/z", force_reading.z)
    rr.log_scalar("force/total", total_force)
    rr.log_scalar("force/recent_avg_total", recent_avg_total_force)

    if recent_avg_total_force > HIGH_FORCE_THRESHOLD and len(force_buffer) == FORCE_BUFFER_SIZE:
        print(f"large force detected! {recent_avg_total_force:.2f}")
        command_client.robot_command(RobotCommandBuilder.stop_command())
        return True
    return False


def pose_in_start_frame(initial_transforms, x, y, angle):
    """
    The start frame is where the robot starts.

    Args:
        initial_transforms: The initial transforms of the robot, this should be created at the beginning.
        x: The x position of the pose in the start frame.
        y: The y position of the pose in the start frame.
        angle: The angle of the pose in the start frame.
    """
    pose_in_body = math_helpers.SE2Pose(x=x, y=y, angle=angle)
    pose_in_odom = get_se2_a_tform_b(initial_transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME) * pose_in_body
    return pose_in_odom


def look_at_scene(command_client, robot_state_client, x=0.56, y=0.1, z=0.55, pitch=0., yaw=0., dx=0., dy=0., dpitch=0.):
    look_cmd = look_at_command(robot_state_client, x + dx, y + dy, z,
                               0, pitch + dpitch, yaw,
                               duration=0.5)
    blocking_arm_command(command_client, look_cmd)


def drag_rope_to_goal(robot_state_client, command_client, initial_transforms, x, y, angle):
    """
    Move the robot to a pose relative to the body while dragging the hose
    """
    force_buffer = []

    # Create the se2 trajectory for the dragging motion
    walk_cmd_id = walk_to_pose_in_initial_frame(command_client, initial_transforms, x=x, y=y, angle=angle,
                                                block=False, crawl=True)

    # loop to check forces
    while True:
        feedback = command_client.robot_command_feedback(walk_cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach goal.")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at goal.")
            return True
        if force_measure(robot_state_client, command_client, force_buffer):
            time.sleep(1)  # makes the video look better in my opinion
            print("High force detected. Failed to reach goal.")
            return False
        time.sleep(0.25)


def walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0., y=0., angle=0., block=True, crawl=False):
    """
    Non-blocking, returns the command id
    """
    goal_pose_in_odom = pose_in_start_frame(initial_transforms, x=x, y=y, angle=angle)
    if crawl:
        locomotion_hint = spot_command_pb2.HINT_CRAWL
    else:
        locomotion_hint = spot_command_pb2.HINT_AUTO
    se2_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=goal_pose_in_odom.to_proto(),
                                                                 frame_name=ODOM_FRAME_NAME,
                                                                 locomotion_hint=locomotion_hint)
    se2_synchro_commnd = RobotCommandBuilder.build_synchro_command(se2_cmd)
    se2_cmd_id = command_client.robot_command(lease=None, command=se2_synchro_commnd, end_time_secs=time.time() + 999)
    if block:
        block_for_trajectory_cmd(command_client, se2_cmd_id)
    return se2_cmd_id


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


def do_grasp(robot, manipulation_api_client, image_res, pick_vec):
    pick_cmd = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec, transforms_snapshot_for_camera=image_res.shot.transforms_snapshot,
        frame_name_image_sensor=image_res.shot.frame_name_image_sensor,
        camera_model=image_res.source.pinhole)

    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick_cmd)
    cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=grasp_request)

    # execute grasp
    t0 = time.time()
    while time.time() - t0 < 15:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        print('Current state: ',
              manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            break

        time.sleep(1)
    robot.logger.info('Finished grasp.')


def walk_to_then_grasp(robot, robot_state_client, image_client, command_client, manipulation_api_client,
                       get_point_f: Callable, first_get_point_kwargs=None,
                       second_get_point_kwargs=None):
    if first_get_point_kwargs is None:
        first_get_point_kwargs = {}
    if second_get_point_kwargs is None:
        second_get_point_kwargs = {}

    walk_to_res = get_point_f_retry(command_client, robot_state_client, image_client, get_point_f,
                                    y=0., z=0.4,
                                    pitch=np.deg2rad(25), yaw=0,
                                    **first_get_point_kwargs)

    # NOTE: if we are going to use the body cameras, which are rotated, we also need to rotate the image coordinates
    # First just walk to in front of that point
    offset_distance = wrappers_pb2.FloatValue(value=1.00)
    # TODO: what we use ordered_hose_points to walk to a pose?
    #  the position should be relative to walk_vec we
    walk_to_cmd = manipulation_api_pb2.WalkToObjectInImage(
        pixel_xy=walk_to_res.best_vec2,
        transforms_snapshot_for_camera=walk_to_res.image_res.shot.transforms_snapshot,
        frame_name_image_sensor=walk_to_res.image_res.shot.frame_name_image_sensor,
        camera_model=walk_to_res.image_res.source.pinhole,
        offset_distance=offset_distance)
    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to_cmd)
    walk_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=walk_to_request)
    block_for_manipulation_api_command(robot, manipulation_api_client, walk_response)

    # Before calling do_grasp, re-orient so the hand stays in place but the body rotates to align Y with the hose
    align_with_hose(command_client, get_point_f, image_client, robot_state_client, second_get_point_kwargs)

    pick_res = get_point_f_retry(command_client, robot_state_client, image_client,
                                 get_point_f, y=0., z=0.5, pitch=np.deg2rad(85), yaw=0)
    do_grasp(robot, manipulation_api_client, pick_res.image_res, pick_res.best_vec2)


def align_with_hose(command_client, get_point_f, image_client, robot_state_client, get_point_kwargs):
    pick_res = get_point_f_retry(command_client, robot_state_client, image_client,
                                 get_point_f, y=0., z=0.5, pitch=np.deg2rad(85), yaw=0, **get_point_kwargs)
    hose_points = pick_res.hose_points
    best_idx = pick_res.best_idx

    # Compute the angle of the hose around the given point using finite differencing
    if best_idx == 0:
        angle1 = min_angle_to_x_axis(hose_points[best_idx] - hose_points[best_idx + 1])
        angle2 = min_angle_to_x_axis(hose_points[best_idx + 1] - hose_points[best_idx + 2])
    elif best_idx == len(hose_points) - 1:
        angle1 = min_angle_to_x_axis(hose_points[best_idx] - hose_points[best_idx - 1])
        angle2 = min_angle_to_x_axis(hose_points[best_idx - 1] - hose_points[best_idx - 2])
    else:
        angle1 = min_angle_to_x_axis(hose_points[best_idx] - hose_points[best_idx - 1])
        angle2 = min_angle_to_x_axis(hose_points[best_idx] - hose_points[best_idx + 1])

    angle = (angle1 + angle2) / 2
    # The angles to +X in pixel space are "flipped" because images are stored with Y increasing downwards
    angle = -angle

    if abs(angle) < np.deg2rad(10):
        print("Not rotating because angle is small")
        return pick_res, angle

    # This is the point we want to rotate around
    best_px = hose_points[best_idx]

    # convert to camera frame and ignore the Z. Assumes the camera is pointed straight down.
    best_pt_in_cam = np.array(pixel_to_camera_space(pick_res.image_res, best_px[0], best_px[1], depth=1.0))[:2]
    # Camera frame is not the same as the frame of the hand, so we do a hacky conversion
    best_pt_in_hand = np.array([-best_pt_in_cam[1], -best_pt_in_cam[0]])

    rotate_around_point_in_hand_frame(command_client, robot_state_client, best_pt_in_hand, angle)
    return pick_res, angle


def rotate_around_point_in_hand_frame(command_client, robot_state_client, pos: np.ndarray, angle: float):
    """
    Moves the body by `angle` degrees, and translate so that `pos` stays in the same place.

    Assumptions:
     - the hand and the body have aligned X axes

    Args:
        command_client: command client
        robot_state_client: robot state client
        pos: 2D position of the point to rotate around, in the hand frame
        angle: angle in radians to rotate the body by, around the Z axis
    """
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    hand_in_odom = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME)
    hand_in_body = get_se2_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    body_in_hand = hand_in_body.inverse()  # NOTE: querying frames in opposite order returns None???
    body_pt_in_hand = np.array([body_in_hand.x, body_in_hand.y])
    rotated_body_pos_in_hand = rot_2d(angle) @ body_pt_in_hand + pos
    rotated_body_in_hand = math_helpers.SE2Pose(rotated_body_pos_in_hand[0], rotated_body_pos_in_hand[1],
                                                angle + body_in_hand.angle)
    goal_in_odom = hand_in_odom * rotated_body_in_hand
    se2_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=goal_in_odom.to_proto(),
                                                                 frame_name=ODOM_FRAME_NAME,
                                                                 locomotion_hint=spot_command_pb2.HINT_CRAWL)
    se2_synchro_commnd = RobotCommandBuilder.build_synchro_command(se2_cmd)
    se2_cmd_id = command_client.robot_command(lease=None, command=se2_synchro_commnd,
                                              end_time_secs=time.time() + 999)
    block_for_trajectory_cmd(command_client, se2_cmd_id)


def reset_before_regrasp(command_client, initial_transforms):
    open_gripper(command_client)
    blocking_arm_command(command_client, RobotCommandBuilder.arm_stow_command())
    walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0.0, y=0.0, angle=0.0)


def raise_hand(command_client, robot_state_client, dz):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    hand_in_body = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    raise_cmd = RobotCommandBuilder.arm_pose_command(hand_in_body.x, hand_in_body.y, hand_in_body.z + dz,
                                                     hand_in_body.rot.w, hand_in_body.rot.x, hand_in_body.rot.y,
                                                     hand_in_body.rot.z,
                                                     GRAV_ALIGNED_BODY_FRAME_NAME,
                                                     0.5)
    blocking_arm_command(command_client, raise_cmd)


def arm_pull_rope(config):
    bosdyn.client.util.setup_logging(config.verbose)

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('RopePullClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, such as the" \
                                    " estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    lease_client.take()

    # Video recording
    from conq.video_recording import VideoRecorder
    device_num = 4
    vr = VideoRecorder(device_num, 'video/')
    vr.start_new_recording(f'demo_{int(time.time())}.mp4')
    vr.start_in_thread()

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        command_client = setup_and_stand(robot)

        initial_transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # open the hand, so we can see more with the depth sensor
        open_gripper(command_client)

        # first detect the goal
        mess_x, mess_y = get_point_f_retry(command_client, robot_state_client, image_client, get_mess,
                                           y=-0.05, z=0.5,
                                           pitch=np.deg2rad(20), yaw=-0.3)
        # offset from the mess because it looks better
        pre_mess_x = mess_x - 0.35
        pre_mess_y = mess_y - 0.4

        while True:
            # Grasp the hose to DRAG
            walk_to_then_grasp(robot, robot_state_client, image_client, command_client, manipulation_api_client,
                               get_hose_and_head_point)

            goal_reached = drag_rope_to_goal(robot_state_client, command_client, initial_transforms,
                                             pre_mess_x, pre_mess_y,
                                             np.pi / 2)
            if goal_reached:
                break

            reset_before_regrasp(command_client, initial_transforms)

            # Grasp the hose to get it UNSTUCK
            walk_to_then_grasp(robot, robot_state_client, image_client, command_client, manipulation_api_client,
                               get_hose_and_regrasp_point,
                               first_get_point_kwargs={'ideal_dist_to_obs': 5},
                               second_get_point_kwargs={'ideal_dist_to_obs': 40})

            # Move the arm to get the hose unstuck
            # TODO: use homotopy planner here
            blocking_arm_command(command_client, look_at_command(robot_state_client, 1.0, 0, 0.2))
            blocking_arm_command(command_client, look_at_command(robot_state_client, 1.0, -0.4, 0.2))
            blocking_arm_command(command_client, look_at_command(robot_state_client, 1.0, -0.4, -0.5))

            # Open the gripper
            open_gripper(command_client)
            blocking_arm_command(command_client, RobotCommandBuilder.arm_stow_command())

            # reset before trying again
            walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0, y=0, angle=0)

        # raise arm a bit
        raise_hand(command_client, robot_state_client, 0.1)
        # rotate to give a better view
        walk_to_pose_in_initial_frame(command_client, initial_transforms, x=mess_x + 0.7, y=mess_y + 0.15,
                                      angle=np.deg2rad(180), crawl=True)

        open_gripper(command_client)
        blocking_arm_command(command_client, RobotCommandBuilder.arm_stow_command())

        # Go home, you're done!
        walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0.0, y=0.0, angle=0.0)
        print("Done!")

    vr.stop_in_thread()


def main(argv):
    np.seterr(all='raise')
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    rr.init("rope_pull")
    rr.connect()
    arm_pull_rope(options)


if __name__ == '__main__':
    # Checks to see IP address is specified
    if not main(sys.argv[1:]):
        sys.exit(1)
