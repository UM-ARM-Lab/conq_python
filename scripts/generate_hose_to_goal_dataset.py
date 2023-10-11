#!/usr/bin/env python

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Callable

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rerun as rr
from bosdyn import geometry
from bosdyn.api import manipulation_api_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME, \
    get_a_tform_b, get_se2_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_command import block_for_trajectory_cmd
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import wrappers_pb2

from arm_segmentation.predictor import Predictor
from conq.cameras_utils import rot_2d, pos_in_cam_to_pos_in_hand
from conq.clients import Clients
from conq.data_recorder import ConqDataRecorder
from conq.exceptions import DetectionError
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command, hand_delta_z, grasp_point_in_image, get_is_grasping
from conq.manipulation import open_gripper, force_measure, add_follow_with_body
from conq.rerun_utils import viz_common_frames
from conq.utils import setup_and_stand
from regrasping_demo.detect_regrasp_point import min_angle_to_x_axis
from regrasping_demo.get_detections import DetectionsResult, get_object_on_floor, get_hose_and_head_point

HOME = math_helpers.SE2Pose(0, 0, 0)


def look_at_scene(clients: Clients, x=0.56, y=0.1, z=0.55, pitch=0., yaw=0., dx=0., dy=0., dpitch=0.,
                  dyaw=0.):
    look_cmd = hand_pose_cmd(clients, x + dx, y + dy, z, 0, pitch + dpitch, yaw + dyaw, duration=0.5)
    blocking_arm_command(clients, look_cmd)


def get_point_f_retry(clients: Clients, get_point_f: Callable) -> DetectionsResult:
    # move through a series of hard-coded body/hand poses
    is_grasping = get_is_grasping(clients)
    start_snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    body_yaws = [
        0,
        0,
        np.deg2rad(90),
        np.deg2rad(180),
        np.deg2rad(270),
        0,
    ]
    hand_poses = [
        (0.6, 0, 0.4, 0, np.deg2rad(65), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(90), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(65), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(65), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(65), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(65), 0),
    ]
    if is_grasping:
        for body_yaw in body_yaws[2:]:
            try:
                time.sleep(1)  # avoid motion blur
                return get_point_f()
            except DetectionError:
                body_in_odom = get_se2_a_tform_b(start_snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
                body_in_odom.angle += body_yaw
                body_command = RobotCommandBuilder.synchro_se2_trajectory_command(body_in_odom.to_proto(),
                                                                                  VISION_FRAME_NAME)

                cmds = [body_command]
                cmd = RobotCommandBuilder.build_synchro_command(*cmds)
                clients.command.robot_command(lease=None, command=cmd, end_time_secs=time.time() + 999)
    else:
        for body_yaw, hand_pose in zip(body_yaws, hand_poses):
            try:
                time.sleep(1)  # avoid motion blur
                return get_point_f()
            except DetectionError:
                x, y, z, hand_roll, hand_pitch, hand_yaw = hand_pose
                euler = geometry.EulerZXY(roll=hand_roll, pitch=hand_pitch, yaw=hand_yaw)
                q = euler.to_quaternion()
                arm_command = RobotCommandBuilder.arm_pose_command(x, y, z, q.w, q.x, q.y, q.z,
                                                                   GRAV_ALIGNED_BODY_FRAME_NAME,
                                                                   0.5)
                body_in_odom = get_se2_a_tform_b(start_snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
                body_in_odom.angle += body_yaw
                body_command = RobotCommandBuilder.synchro_se2_trajectory_command(body_in_odom.to_proto(),
                                                                                  VISION_FRAME_NAME)

                cmd = RobotCommandBuilder.build_synchro_command(arm_command, body_command)
                clients.command.robot_command(lease=None, command=cmd, end_time_secs=time.time() + 999)
                blocking_arm_command(clients, arm_command)

    raise DetectionError()


def drag_hand_to_goal(clients: Clients, predictor: Predictor):
    """
    Move the robot to a pose relative to the body while dragging the hose, so the hand arrives at the goal pose
    Ideally here we would plan a collision free path to the goal, but what should the model of the hose be?
    We assume that the end not grasped by the robot is fixed in the world, but we don't know where that is.
    """
    snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    viz_common_frames(snapshot)
    _get_goal = partial(get_object_on_floor, predictor, clients.image, 'mess mat')
    get_goal_res = get_point_f_retry(clients, _get_goal)
    hand_in_body = get_se2_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    offset = hand_in_body.x
    pixel_xy = get_goal_res.best_vec2.to_proto()
    res_image = get_goal_res.rgb_res
    offset_distance = wrappers_pb2.FloatValue(value=offset)
    walk_to = manipulation_api_pb2.WalkToObjectInImage(pixel_xy=pixel_xy,
                                                       transforms_snapshot_for_camera=res_image.shot.transforms_snapshot,
                                                       frame_name_image_sensor=res_image.shot.frame_name_image_sensor,
                                                       camera_model=res_image.source.pinhole,
                                                       offset_distance=offset_distance)
    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
    cmd_response = clients.manipulation.manipulation_api_command(manipulation_api_request=walk_to_request)

    force_buffer = []
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)
        response = clients.manipulation.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)
        print(f'Current state: {state_name}')

        if force_measure(clients, force_buffer):
            time.sleep(1)  # makes the video look better in my opinion
            print("High force detected. Failed to reach goal.")
            return False

        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            break

        time.sleep(0.25)

    return True


def walk_to_pose_in_initial_frame(clients: Clients, initial_transforms, goal, block=True, crawl=False):
    """
    Non-blocking, returns the command id
    """
    goal_pose_in_odom = get_se2_a_tform_b(initial_transforms, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME) * goal
    if crawl:
        locomotion_hint = spot_command_pb2.HINT_CRAWL
    else:
        locomotion_hint = spot_command_pb2.HINT_AUTO
    se2_cmd_id = walk_to_pose_in_odom(clients, goal_pose_in_odom, locomotion_hint, block)
    return se2_cmd_id


def walk_to_pose_in_odom(clients, goal_pose_in_odom, locomotion_hint, block):
    se2_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=goal_pose_in_odom.to_proto(),
                                                                 frame_name=VISION_FRAME_NAME,
                                                                 locomotion_hint=locomotion_hint)
    se2_synchro_commnd = RobotCommandBuilder.build_synchro_command(se2_cmd)
    se2_cmd_id = clients.command.robot_command(lease=None, command=se2_synchro_commnd, end_time_secs=time.time() + 999)
    if block:
        block_for_trajectory_cmd(clients.command, se2_cmd_id)
    return se2_cmd_id


def hand_delta_in_body_frame(clients: Clients, dx, dy, dz, follow=True):
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    hand_in_body = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    hand_pos_in_body = hand_in_body.position
    cmd = hand_pose_cmd(clients, hand_pos_in_body.x + dx, hand_pos_in_body.y + dy, hand_pos_in_body.z + dz)
    if follow:
        cmd = add_follow_with_body(cmd)
    blocking_arm_command(clients, cmd)


def align_with_hose(clients: Clients, get_point_f):
    pick_res = get_point_f_retry(clients, get_point_f)
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

    if abs(angle) < np.deg2rad(15):
        print("Not rotating because angle is small")
        return

    # This is the point we want to rotate around
    best_px = hose_points[best_idx]

    # convert to camera frame and ignore the Z. Assumes the camera is pointed straight down.
    best_pt_in_cam = np.array(pixel_to_camera_space(pick_res.rgb_res, best_px[0], best_px[1], depth=1.0))[:2]
    best_pt_in_hand = pos_in_cam_to_pos_in_hand(best_pt_in_cam)

    rotate_around_point_in_hand_frame(clients, best_pt_in_hand, angle)


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


def retry_grasp_hose(clients: Clients, get_point_f):
    for _ in range(5):
        grasp_res = get_point_f_retry(clients, get_point_f)

        # first just try the automatic grasp
        success = grasp_point_in_image(clients, grasp_res.rgb_res, grasp_res.best_vec2)
        if success:
            return

        # If that fails try aligning with the hose
        look_at_scene(clients, z=0.4, pitch=np.deg2rad(85))
        align_with_hose(clients, get_point_f)
        grasp_res = get_point_f_retry(clients, get_point_f)
        success = grasp_point_in_image(clients, grasp_res.rgb_res, grasp_res.best_vec2)
        if success:
            return


def go_to_goal(predictor, clients):
    while True:
        # Grasp the hose to DRAG
        clients.recorder.add_instruction("find the head of the hose")
        _get_hose_head = partial(get_hose_and_head_point, predictor, clients.image)
        look_at_scene(clients, z=0.4, pitch=np.deg2rad(85))

        clients.recorder.add_instruction("grasp the neck of the hose, near the head")
        retry_grasp_hose(clients, _get_hose_head)

        dz = 0.4
        clients.recorder.add_instruction("lift the hose up")
        hand_delta_z(clients, dz)

        time.sleep(0.5)
        is_grasping = get_is_grasping(clients)
        if not is_grasping:
            continue

        clients.recorder.add_instruction("drag the hose to the goal")
        goal_reached = drag_hand_to_goal(clients, predictor)

        is_grasping = get_is_grasping(clients)
        if not is_grasping:
            continue

        if goal_reached:
            hand_delta_z(clients, -dz * 0.9)
            return True


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    rr.init("continuous_regrasping_hose_demo")
    rr.connect()

    predictor = Predictor('models/hose_regrasping.pth')

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('continuous_regrasping_hose_demo')
    robot = sdk.create_robot('192.168.80.3')
    # robot = sdk.create_robot('10.0.0.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, such as the" \
                                    " estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)

    lease_client.take()

    now = int(time.time())
    root = Path(f"data/regrasping_dataset_{now}")
    recorder = ConqDataRecorder(root, robot_state_client, image_client)

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        command_client = setup_and_stand(robot)

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                          image=image_client, raycast=rc_client, command=command_client, robot=robot,
                          recorder=recorder)

        open_gripper(clients)

        initial_transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        for mode, num_episodes in [('train', 10), ('val', 5)]:
            for episode_idx in range(num_episodes):
                recorder.start_episode(mode)
                go_to_goal(predictor, clients)

                clients.recorder.add_instruction("release and stow the arm")
                open_gripper(clients)
                blocking_arm_command(clients, RobotCommandBuilder.arm_stow_command())

                clients.recorder.add_instruction("walk back to the start")
                walk_to_pose_in_initial_frame(clients, initial_transforms, math_helpers.SE2Pose(0, 0, 0))
                recorder.next_episode()
        recorder.stop()


if __name__ == '__main__':
    main()
