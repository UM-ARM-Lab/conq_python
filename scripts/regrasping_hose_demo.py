import argparse
import sys
import time
from functools import partial
from typing import Callable

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rerun as rr
from bosdyn.api import manipulation_api_pb2, trajectory_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, \
    GROUND_PLANE_FRAME_NAME
from bosdyn.client.frame_helpers import get_se2_a_tform_b, \
    HAND_FRAME_NAME
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_command import (block_for_trajectory_cmd)
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import wrappers_pb2

from arm_segmentation.predictor import Predictor
from conq.cameras_utils import rot_2d, get_color_img, get_depth_img, camera_space_to_pixel, pos_in_cam_to_pos_in_hand
from conq.exceptions import DetectionError, GraspingException
from conq.hand_motion import hand_pose_cmd, hand_delta_in_body_frame
from conq.manipulation import block_for_manipulation_api_command, open_gripper, force_measure, \
    do_grasp, raise_hand
from conq.manipulation import blocking_arm_command
from conq.utils import setup_and_stand
from regrasping_demo import homotopy_planner
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.center_object import center_object_step
from regrasping_demo.detect_regrasp_point import min_angle_to_x_axis, detect_regrasp_point_from_hose
from regrasping_demo.get_detections import GetRetryResult, np_to_vec2, get_hose_and_regrasp_point, \
    get_hose_and_head_point, get_mess
from regrasping_demo.get_detections import save_data
from regrasping_demo.homotopy_planner import get_obstacle_coms


def look_at_scene(command_client, robot_state_client, x=0.56, y=0.1, z=0.55, pitch=0., yaw=0., dx=0., dy=0., dpitch=0.):
    look_cmd = hand_pose_cmd(robot_state_client, x + dx, y + dy, z,
                             0, pitch + dpitch, yaw,
                             duration=0.5)
    blocking_arm_command(command_client, look_cmd)


def get_point_f_retry(command_client, robot_state_client, image_client, predictor, get_point_f: Callable,
                      y, z, pitch, yaw) -> GetRetryResult:
    dx = 0
    dy = 0
    dpitch = 0
    while True:
        look_at_scene(command_client, robot_state_client, y=y, z=z, pitch=pitch, yaw=yaw, dx=dx, dy=dy, dpitch=dpitch)
        try:
            return get_point_f(predictor, image_client, robot_state_client)
        except DetectionError:
            dx = np.random.randn() * 0.05
            dy = np.random.randn() * 0.05
            dpitch = np.random.randn() * 0.08


def pose_in_start_frame(initial_transforms, x, y, yaw):
    """
    The start frame is where the robot starts.

    Args:
        initial_transforms: The initial transforms of the robot, this should be created at the beginning.
        x: The x position of the pose in the start frame.
        y: The y position of the pose in the start frame.
        yaw: The angle of the pose in the start frame.
    """
    pose_in_body = math_helpers.SE2Pose(x=x, y=y, angle=yaw)
    pose_in_odom = get_se2_a_tform_b(initial_transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME) * pose_in_body
    return pose_in_odom


def drag_rope_to_goal(robot_state_client, command_client, initial_transforms, x, y, angle):
    """ Move the robot to a pose relative to the body while dragging the hose """
    force_buffer = []

    # Raise arm a bit
    raise_hand(command_client, robot_state_client, 0.1)

    # Create the se2 trajectory for the dragging motion
    walk_cmd_id = walk_to_pose_in_initial_frame(command_client, initial_transforms, x=x, y=y, yaw=angle,
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


def walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0., y=0., yaw=0., block=True, crawl=False):
    """
    Non-blocking, returns the command id
    """
    goal_pose_in_odom = pose_in_start_frame(initial_transforms, x=x, y=y, yaw=yaw)
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

def walk_to(robot_state_client, image_client, command_client, predictor, get_point_f):
    walk_to_res = get_point_f_retry(command_client, robot_state_client, image_client, predictor, get_point_f,
                                    y=0., z=0.4,
                                    pitch=np.deg2rad(30), yaw=0)
    # This should probably be an offset 
    offset_distance = wrappers_pb2.FloatValue(value=1.00)
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    transforms_cam = walk_to_res.image_res.shot.transforms_snapshot
    frame_name_shot = walk_to_res.image_res.shot.frame_name_image_sensor
    
    walk_to_res_in_body = get_se2_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME) * get_se2_a_tform_b(transforms_cam, ODOM_FRAME_NAME, frame_name_shot)
    se2_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(goal_x_rt_body=walk_to_res_in_body.x, goal_y_rt_body=walk_to_res_in_body.y, goal_heading_rt_body=walk_to_res_in_body.angle, frame_name=ODOM_FRAME_NAME)
    se2_synchro_cmd = RobotCommandBuilder.build_synchro_command(se2_cmd)
    se2_cmd_id = command_client.robot_command(lease=None, command=se2_synchro_cmd, end_time_secs=time.time() + 999)
    block_for_trajectory_cmd(command_client, se2_cmd_id)

#TODO: This function isn't useful anymore? Should we still be aligning after walking to the point?
def align_with_hose(command_client, robot_state_client, image_client, get_point_f):
    pick_res = get_point_f_retry(command_client, robot_state_client, image_client,
                                 get_point_f, y=0., z=0.5, pitch=np.deg2rad(85), yaw=0)
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
    best_pt_in_hand = pos_in_cam_to_pos_in_hand(best_pt_in_cam)

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


def retry_do_grasp(command_client, robot_state_client, manipulation_api_client, image_client, get_point_f):
    for _ in range(3):
        # Try repeatedly to get a valid point in the image to grasp
        pick_res = get_point_f_retry(command_client, robot_state_client, image_client, get_point_f, y=0., z=0.5,
                                     pitch=np.deg2rad(85), yaw=0)
        # Try to do the grasp
        success = do_grasp(command_client, manipulation_api_client, robot_state_client, pick_res.image_res,
                           pick_res.best_vec2)
        if success:
            return

    open_gripper(command_client)
    raise GraspingException("Failed to grasp")


def reset_before_regrasp(command_client, initial_transforms):
    open_gripper(command_client)
    blocking_arm_command(command_client, RobotCommandBuilder.arm_stow_command())
    walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0.0, y=0.0, yaw=0.0)


def center_obstacles(predictor, command_client, robot_state_client, image_client, motion_scale=0.0004):
    rng = np.random.RandomState(0)
    for _ in range(5):
        rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
        depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
        predictions = predictor.predict(rgb_np)
        save_data(rgb_np, depth_np, predictions)

        delta_px = center_object_step(rgb_np, predictions, rng)

        if delta_px is None:
            print("success!")
            break

        # FIXME: generalize this math/transform. This assumes that +x in image (column) is -Y in body, etc.
        # FIXME: should be using camera intrinsics here so motion scale makes more sense
        dx_in_body, dy_in_body = np.array([-delta_px[1], -delta_px[0]]) * motion_scale
        hand_delta_in_body_frame(command_client, robot_state_client, dx_in_body, dy_in_body, dz=0, follow=False)


def main(argv):
    np.seterr(all='raise')
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    args = parser.parse_args(argv)
    rr.init("rope_pull")
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [0.4, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 0.4, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 0.4], color=(0, 0, 255), width_scale=0.02)

    predictor = Predictor()

    bosdyn.client.util.setup_logging(args.verbose)

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('RopePullClient')
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, such as the" \
                                    " estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)

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
        handedness = 1  # 1 for when the mess is on the left, -1 for when the mess is on the right
        mess_yaw = handedness * -np.pi / 2
        mess_x, mess_y = get_point_f_retry(command_client, robot_state_client, image_client,
                                           partial(get_mess, predictor, rc_client),
                                           y=handedness * 0.10, z=0.5, pitch=np.deg2rad(20), yaw=handedness * 0.4,
                                           )
        # # offset from the mess because it looks better
        pre_mess_x = mess_x - 0.5
        pre_mess_y = mess_y + handedness * 0.55
        post_mess_x = mess_x
        post_mess_y = mess_y + handedness * 0.55

        while True:
            # Grasp the hose to DRAG
            walk_to(robot_state_client, image_client, command_client, predictor, get_hose_and_head_point)
            align_with_hose(command_client, robot_state_client, image_client, get_hose_and_head_point)
            retry_do_grasp(command_client, robot_state_client, manipulation_api_client, image_client,
                           get_hose_and_head_point)

            goal_reached = drag_rope_to_goal(robot_state_client, command_client, initial_transforms,
                                             pre_mess_x, pre_mess_y,
                                             mess_yaw)
            if goal_reached:
                break

            reset_before_regrasp(command_client, initial_transforms)

            # Grasp the hose to get it UNSTUCK
            walk_to(robot_state_client, image_client, command_client, predictor,
                    partial(get_hose_and_regrasp_point, predictor, ideal_dist_to_obs=5))

            # Move the arm to get the hose unstuck
            for _ in range(3):
                align_with_hose(command_client, robot_state_client, image_client,
                                partial(get_hose_and_regrasp_point, predictor, ideal_dist_to_obs=40))

                # Center the obstacles in the frame
                center_obstacles(predictor, command_client, robot_state_client, image_client)

                rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
                depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
                predictions = predictor.predict(rgb_np)
                save_data(rgb_np, depth_np, predictions)

                _, obstacles_mask = get_obstacle_coms(predictions)
                if np.sum(obstacles_mask) == 0:
                    walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0, y=0, yaw=0)
                    continue

                obstacle_mask_with_valid_depth = np.logical_and(obstacles_mask, depth_np.squeeze(-1) > 0)
                nearest_obs_to_hand = np.min(depth_np[np.where(obstacle_mask_with_valid_depth)]) / 1000

                try:
                    hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
                except DetectionError:
                    walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0, y=0, yaw=0)
                    continue

                _, regrasp_px = detect_regrasp_point_from_hose(predictions, hose_points, ideal_dist_to_obs=70)
                regrasp_vec2 = np_to_vec2(regrasp_px)
                regrasp_x_in_cam, regrasp_y_in_cam, _ = pixel_to_camera_space(rgb_res, regrasp_px[0], regrasp_px[1],
                                                                              depth=1.0)
                regrasp_x, regrasp_y = pos_in_cam_to_pos_in_hand([regrasp_x_in_cam, regrasp_y_in_cam])

                # BEFORE we grasp, get the robot's position in image space
                transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
                body_in_hand = get_a_tform_b(transforms, HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
                hand_in_gpe = get_a_tform_b(transforms, GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME)
                hand_to_floor = hand_in_gpe.z
                body_in_cam = np.array([-body_in_hand.y, -body_in_hand.z])
                robot_px = np.array(camera_space_to_pixel(rgb_res, body_in_cam[0], body_in_cam[1], hand_to_floor))

                _, place_px = homotopy_planner.plan(rgb_np, predictions, predictor.colors, hose_points, regrasp_px,
                                                    robot_px)

                place_x_in_cam, place_y_in_cam, _ = pixel_to_camera_space(rgb_res, place_px[0], place_px[1], depth=1.0)
                place_x, place_y = pos_in_cam_to_pos_in_hand([place_x_in_cam, place_y_in_cam])

                # Compute the desired poses for the hand
                nearest_obs_height = hand_to_floor - nearest_obs_to_hand
                dplace_x = place_x - regrasp_x
                dplace_y = place_y - regrasp_y

                # Do the grasp
                success = do_grasp(command_client, manipulation_api_client, robot_state_client, rgb_res, regrasp_vec2)
                if success:
                    break
            else:
                walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0, y=0, yaw=0)
                continue

            hand_delta_in_body_frame(command_client, robot_state_client, dx=0, dy=0, dz=nearest_obs_height + 0.2,
                                     follow=False)
            hand_delta_in_body_frame(command_client, robot_state_client, dx=dplace_x, dy=dplace_y, dz=0)
            # Move down to the floor
            transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
            hand_in_gpe = get_a_tform_b(transforms, GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME)
            hand_delta_in_body_frame(command_client, robot_state_client, dx=0, dy=0, dz=-hand_in_gpe.z + 0.05)
            # Open the gripper
            open_gripper(command_client)
            blocking_arm_command(command_client, RobotCommandBuilder.arm_stow_command())

            # reset before trying again
            walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0, y=0, yaw=0)

        # some finishing walking commands to make it look pretty
        walk_to_pose_in_initial_frame(command_client, initial_transforms, x=post_mess_x, y=post_mess_y, yaw=mess_yaw,
                                      crawl=True)

        open_gripper(command_client)
        blocking_arm_command(command_client, RobotCommandBuilder.arm_stow_command())

        walk_to_pose_in_initial_frame(command_client, initial_transforms, x=pre_mess_x, y=pre_mess_y, yaw=mess_yaw,
                                      crawl=True)

        # Go home, you're done!
        walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0.0, y=0.0, yaw=0.0)

        vr.stop_in_thread()


if __name__ == '__main__':
    # Checks to see IP address is specified
    if not main(sys.argv[1:]):
        sys.exit(1)
