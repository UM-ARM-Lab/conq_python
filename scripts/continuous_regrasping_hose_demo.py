import argparse
import time
from functools import partial
from typing import Callable

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rerun as rr
from bosdyn import geometry
from bosdyn.api import basic_command_pb2, manipulation_api_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, \
    GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b, VISION_FRAME_NAME, ODOM_FRAME_NAME, \
    BODY_FRAME_NAME
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_command import block_for_trajectory_cmd
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import wrappers_pb2
from matplotlib import pyplot as plt

from arm_segmentation.predictor import Predictor, get_combined_mask
from arm_segmentation.viz import viz_predictions
from conq.cameras_utils import rot_2d, get_color_img, camera_space_to_pixel, pos_in_cam_to_pos_in_hand, get_depth_img
from conq.clients import Clients
from conq.conq_astar import ConqAStar, yaw_diff, offset_from_hose
from conq.exceptions import DetectionError, GraspingException, PlanningException
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command
from conq.manipulation import open_gripper, force_measure, \
    do_grasp, add_follow_with_body
from conq.perception import project_points_in_gpe
from conq.utils import setup
from regrasping_demo import homotopy_planner
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.center_object import center_object_step
from regrasping_demo.detect_regrasp_point import min_angle_to_x_axis, detect_regrasp_point_from_hose
from regrasping_demo.get_detections import GetRetryResult, np_to_vec2, \
    get_object_on_floor, detect_object_points, get_body_goal_se2_from_hose_points, \
    DEFAULT_IDEAL_DIST_TO_OBS, get_hose_and_regrasp_point

HOME = math_helpers.SE2Pose(0, 0, 0)


def look_at_scene(clients: Clients, x=0.56, y=0.1, z=0.55, pitch=0., yaw=0., dx=0., dy=0., dpitch=0.,
                  dyaw=0.):
    look_cmd = hand_pose_cmd(clients, x + dx, y + dy, z, 0, pitch + dpitch, yaw + dyaw, duration=0.5)
    blocking_arm_command(clients, look_cmd)


def get_point_f_retry(clients: Clients, get_point_f: Callable) -> GetRetryResult:
    # move through a series of hard-coded body/hand poses
    body_yaws = [
        0,
        0,
        0,
        0,
        0,
        0,
        np.deg2rad(90),
        np.deg2rad(180),
        np.deg2rad(270),
        0,
    ]
    hand_poses = [
        (0.6, 0, 0.4, 0, np.deg2rad(40), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(90), 0),
        (0.5, 0.25, 0.4, 0, np.deg2rad(60), np.deg2rad(75)),
        (0.5, -0.25, 0.4, 0, np.deg2rad(60), np.deg2rad(-75)),
        (0.5, 0.25, 0.4, 0, np.deg2rad(50), np.deg2rad(35)),
        (0.5, -0.25, 0.4, 0, np.deg2rad(50), np.deg2rad(-35)),
        (0.6, 0, 0.4, 0, np.deg2rad(40), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(40), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(40), 0),
        (0.6, 0, 0.4, 0, np.deg2rad(40), 0),
    ]

    snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot

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
            body_in_odom = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
            body_in_odom.angle += body_yaw
            body_command = RobotCommandBuilder.synchro_se2_trajectory_command(body_in_odom.to_proto(),
                                                                              VISION_FRAME_NAME)

            cmd = RobotCommandBuilder.build_synchro_command(arm_command, body_command)
            clients.command.robot_command(lease=None, command=cmd, end_time_secs=time.time() + 999)
            blocking_arm_command(clients, arm_command)

    raise DetectionError()


def drag_hand_to_goal(clients: Clients, predictor: Predictor):
    """ Move the robot to a pose relative to the body while dragging the hose, so the hand arrives at the goal pose """

    # First find the goal
    snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    viz_common_frames(snapshot)

    _get_goal = partial(get_object_on_floor, predictor, clients.image, 'mess mat')
    get_goal_res = get_point_f_retry(clients, _get_goal)

    hand_in_body = get_se2_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    offset = hand_in_body.x
    pixel_xy = get_goal_res.best_vec2.to_proto()

    res_image = get_goal_res.rgb_res
    offset_distance = wrappers_pb2.FloatValue(value=offset)

    locomotion_hint = spot_command_pb2.HINT_CRAWL
    walk_to = manipulation_api_pb2.WalkToObjectInImage(pixel_xy=pixel_xy,
                                                       transforms_snapshot_for_camera=res_image.shot.transforms_snapshot,
                                                       frame_name_image_sensor=res_image.shot.frame_name_image_sensor,
                                                       camera_model=res_image.source.pinhole,
                                                       offset_distance=offset_distance)
    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
    # cmd_response = clients.manipulation.manipulation_api_command(manipulation_api_request=walk_to_request)
    clients.manipulation.manipulation_api_command_async(manipulation_api_request=walk_to_request)

    while True:
        # TODO: could we do continuous detection here?
        time.sleep(0.25)
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)
        response = clients.manipulation.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)
        print('Current state: ', manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))
        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            break

    force_buffer = []

    print("DEBUGGIN!!! PUT ME BACK!!!")
    # Raise arm a bit
    raise_hand(clients, 0.3)

    # Ideally here we would plan a collision free path to the goal, but what should the model of the hose be?
    # We assume that the end not grasped by the robot is fixed in the world, but we don't know where that is.
    snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot

    # get the offset from hand to body
    hand_in_body = get_se2_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    offset = np.array([hand_in_body.x, hand_in_body.y])

    rr_tform("goal", goal_pose_in_odom.get_closest_se3_transform(0))  # TODO: account for goal yaw
    offset2 = goal_pose_in_odom.to_rot_matrix() @ offset

    # frames???
    goal_pose_in_odom_offset = math_helpers.SE2Pose(goal_pose_in_odom.x,
                                                    goal_pose_in_odom.y,
                                                    goal_pose_in_odom.angle)

    rr_tform("goal_offset", goal_pose_in_odom_offset.get_closest_se3_transform(0))  # TODO: account for goal yaw
    viz_common_frames(snapshot)

    walk_cmd_id = walk_to_pose_in_odom(clients, goal_pose_in_odom_offset, block=False,
                                       locomotion_hint=spot_command_pb2.HINT_CRAWL)

    # loop to check forces
    while True:
        feedback = clients.command.robot_command_feedback(walk_cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach goal.")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at dragging goal.")
            return True
        if force_measure(clients, force_buffer):
            time.sleep(1)  # makes the video look better in my opinion
            print("High force detected. Failed to reach goal.")
            return False
        time.sleep(0.25)


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
        return pick_res, angle

    # This is the point we want to rotate around
    best_px = hose_points[best_idx]

    # convert to camera frame and ignore the Z. Assumes the camera is pointed straight down.
    best_pt_in_cam = np.array(pixel_to_camera_space(pick_res.rgb_res, best_px[0], best_px[1], depth=1.0))[:2]
    best_pt_in_hand = pos_in_cam_to_pos_in_hand(best_pt_in_cam)

    rotate_around_point_in_hand_frame(clients, best_pt_in_hand, angle)
    return pick_res, angle


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


def retry_do_grasp(clients: Clients, get_point_f):
    for _ in range(5):
        pick_res = get_point_f_retry(clients, get_point_f)
        success = do_grasp(clients, pick_res.rgb_res, pick_res.best_vec2)
        if success:
            return
        else:
            # reset to retry the grasp
            look_at_scene(clients, z=0.4, pitch=np.deg2rad(85))


def center_obstacles(predictor, clients, motion_scale=0.0004):
    rng = np.random.RandomState(0)
    for _ in range(5):
        rgb_np, rgb_res = get_color_img(clients.image, 'hand_color_image')
        # depth_np, depth_res = get_depth_img(clients.image, 'hand_depth_in_hand_color_frame')
        predictions = predictor.predict(rgb_np)

        delta_px = center_object_step(rgb_np, predictions, rng)

        if delta_px is None:
            print("Centered successfully!")
            break

        # FIXME: generalize this math/transform. This assumes that +x in image (column) is -Y in body, etc.
        # FIXME: should be using camera intrinsics here so motion scale makes more sense
        dx_in_body, dy_in_body = np.array([-delta_px[1], -delta_px[0]]) * motion_scale
        hand_delta_in_body_frame(clients, dx_in_body, dy_in_body, dz=0, follow=False)


def pregrasp_walk_up(clients: Clients, get_point_f):
    walk_to_res = get_point_f_retry(clients, get_point_f)

    cam2odom = get_a_tform_b(walk_to_res.rgb_res.shot.transforms_snapshot,
                             walk_to_res.rgb_res.shot.frame_name_image_sensor, VISION_FRAME_NAME)
    hose_points_in_odom = project_points_in_gpe(walk_to_res.hose_points, walk_to_res.rgb_res, cam2odom)
    gaze_point_in_odom = hose_points_in_odom[walk_to_res.best_idx]

    snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    base_in_odom = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    start = (base_in_odom.x, base_in_odom.y, base_in_odom.angle)

    goal_pose_in_odom = get_body_goal_se2_from_hose_points(hose_points_in_odom, walk_to_res.best_idx, start)
    offset_goal_pose_in_odom = offset_from_hose(goal_pose_in_odom, dist=0.8)

    gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_point_in_odom[0],
                                                        gaze_point_in_odom[1],
                                                        0,
                                                        VISION_FRAME_NAME,
                                                        max_linear_vel=2, max_angular_vel=0.8, max_accel=10)
    # start the gaze async with the walk
    clients.command.robot_command(gaze_command)

    goal_pose_in_odom = math_helpers.SE2Pose(*offset_goal_pose_in_odom)
    walk_to_pose_in_odom(clients, goal_pose_in_odom, block=True, locomotion_hint=spot_command_pb2.HINT_AUTO)


def plan_se2_path_to_regrasp_point(clients: Clients, best_idx, hose_pixels, predictions, rgb_res):
    cam2odom = get_a_tform_b(rgb_res.shot.transforms_snapshot, rgb_res.shot.frame_name_image_sensor,
                             VISION_FRAME_NAME)
    snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    battery_px_points = detect_object_points(predictions, "battery")
    gpe2odom = get_a_tform_b(snapshot, GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME)
    odom2gpe = gpe2odom.inverse()
    cam2gpe = cam2odom * odom2gpe
    hose_points_in_gpe = project_points_in_gpe(hose_pixels, rgb_res, cam2gpe)
    battery_points_in_gpe = project_points_in_gpe(battery_px_points, rgb_res, cam2gpe)
    obstacle_points_in_gpe = np.concatenate([hose_points_in_gpe, battery_points_in_gpe], axis=0)
    rr.log_points('gpe/hose_points', hose_points_in_gpe)
    rr.log_transform3d(f'gpe/gpe', rr.Translation3D([0, 0, 0]))
    a = ConqAStar()
    start = (0, 0, 0)
    obstacle_point_radius = 0.08
    for ob in obstacle_points_in_gpe:
        a.add_obstacle(ob[0], ob[1], obstacle_point_radius)
    d_to_goal = 0.5
    # goal is in GPE frame
    goal_x, goal_y, goal_angle = get_body_goal_se2_from_hose_points(hose_points_in_gpe, best_idx, start)
    while d_to_goal < 1.0:
        regrasp_goal = offset_from_hose((goal_x, goal_y, goal_angle), d_to_goal)
        if not a.in_collision(regrasp_goal):
            break
        d_to_goal += 0.05
        goal_x += np.random.uniform(-0.05, 0.05)
        goal_y += np.random.uniform(-0.05, 0.05)
        goal_angle += np.random.uniform(-np.deg2rad(1), np.deg2rad(1))
    path = a.astar(start=start, goal=regrasp_goal)
    if path is None:
        raise PlanningException("A star failed!")
    a.viz(start, regrasp_goal, list(path))
    return hose_points_in_gpe, path, snapshot


def follow_se2_path(clients: Clients, best_idx, hose_points_in_gpe, path, snapshot):
    gpe2odom_se2 = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    for point in path[1:]:  # skip the first point, which is the start
        point_in_gpe = math_helpers.SE2Pose(x=point[0], y=point[1], angle=point[2])
        point_in_odom = gpe2odom_se2 * point_in_gpe
        se2_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=point_in_odom.to_proto(),
                                                                     frame_name=VISION_FRAME_NAME,
                                                                     locomotion_hint=spot_command_pb2.HINT_CRAWL)
        se2_synchro_commnd = RobotCommandBuilder.build_synchro_command(se2_cmd)
        se2_cmd_id = clients.command.robot_command(lease=None, command=se2_synchro_commnd,
                                                   end_time_secs=time.time() + 999)
        block_until_near_se2(clients, se2_cmd_id, point_in_odom, xy_tol=0.1, yaw_tol=np.deg2rad(10))
    # be more precise at the end
    block_until_near_se2(clients, se2_cmd_id, point_in_odom, xy_tol=0.05, yaw_tol=np.deg2rad(5))

    # Gaze at the regrasp point walk walking
    gaze_point_in_gpe = hose_points_in_gpe[best_idx]
    gaze_point_in_odom = gpe2odom_se2 * math_helpers.SE2Pose(x=gaze_point_in_gpe[0], y=gaze_point_in_gpe[1],
                                                             angle=0)
    gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_point_in_odom.x,
                                                        gaze_point_in_odom.y,
                                                        0,
                                                        VISION_FRAME_NAME,
                                                        max_linear_vel=0.3, max_angular_vel=0.8, max_accel=2)
    blocking_arm_command(clients, gaze_command)


def block_until_near_se2(clients: Clients, cmd_id, point_in_odom: math_helpers.SE2Pose, xy_tol=0.1,
                         yaw_tol=np.deg2rad(5)):
    while True:
        feedback_resp = clients.command.robot_command_feedback(cmd_id)

        current_trajectory_state = feedback_resp.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status

        if current_trajectory_state == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL:
            return True

        snapshot = clients.state.get_robot_state().kinematic_state.transforms_snapshot
        # check if the robot is near the goal
        body_in_odom = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        dx = abs(body_in_odom.x - point_in_odom.x)
        dy = abs(body_in_odom.y - point_in_odom.y)
        dyaw = yaw_diff(body_in_odom.angle, point_in_odom.angle)

        if dx < xy_tol and dy < xy_tol and dyaw < yaw_tol:
            return True

        time.sleep(0.1)


def go_to_goal(predictor, clients, initial_transforms):
    while True:
        # # Grasp the hose to DRAG
        # _get_hose_head = partial(get_hose_and_head_point, predictor, clients.image)
        #
        # # pregrasp_walk_up(clients, _get_hose_head)
        # # align_with_hose(clients, _get_hose_head)
        # retry_do_grasp(clients, _get_hose_head)

        goal_reached = drag_hand_to_goal(clients, predictor)
        if goal_reached:
            return True

        # setup to look for the hose, which we just dropped
        open_gripper(clients)
        look_at_scene(clients, z=0.3, pitch=np.deg2rad(85))

        # TODO: Look for the hose and get the full estimate in base frame
        rotate_around_point_in_hand_frame(clients, [0, 0], np.deg2rad(-60))
        look_cmd = hand_pose_cmd(clients, 0.6, 0., 0.45, 0, np.deg2rad(45), np.deg2rad(-60), duration=0.5)
        blocking_arm_command(clients, look_cmd)

        # Detect the regrasp point
        # this sort of duplicates some code from get_hose_and_regrasp_point()
        rgb_np, rgb_res = get_color_img(clients.image, 'hand_color_image')
        predictions = predictor.predict(rgb_np)
        hose_pixels = single_frame_planar_cdcpd(rgb_np, predictions)

        # from regrasping_demo.get_detections import save_all_rgb
        # save_all_rgb(image_client)
        # TODO: what if we can't find it?
        best_idx, best_px = detect_regrasp_point_from_hose(predictions, hose_pixels,
                                                           ideal_dist_to_obs=DEFAULT_IDEAL_DIST_TO_OBS)

        # visualize
        fig, ax = plt.subplots()
        ax.imshow(rgb_np, zorder=0)
        viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
        ax.scatter(best_px[0], best_px[1], s=200, marker='*', c='orange', zorder=4)
        fig.show()

        try:
            hose_points_in_gpe, path, snapshot = plan_se2_path_to_regrasp_point(clients, best_idx, hose_pixels,
                                                                                predictions,
                                                                                rgb_res)
        except PlanningException:
            continue

        # execute the se2 command
        follow_se2_path(clients, best_idx, hose_points_in_gpe, path, snapshot)

        # Move the arm to get the hose unstuck
        for _ in range(3):
            # Center the obstacles in the frame
            try:
                center_obstacles(predictor, clients)
            except DetectionError:
                print("Failed to center obstacles, retrying")
                continue

            rgb_np, rgb_res = get_color_img(clients.image, 'hand_color_image')
            depth_np, depth_res = get_depth_img(clients.image, 'hand_depth_in_hand_color_frame')
            predictions = predictor.predict(rgb_np)

            obstacles_mask = get_combined_mask(predictions, "battery")
            if obstacles_mask is None:
                walk_to_pose_in_initial_frame(clients, initial_transforms, HOME)
                print("no obstacles found, restarting!")
                return

            if np.sum(obstacles_mask) == 0:
                walk_to_pose_in_initial_frame(clients, initial_transforms, HOME)
                continue

            obstacle_mask_with_valid_depth = np.logical_and(obstacles_mask, depth_np.squeeze(-1) > 0)
            nearest_obs_to_hand = np.min(depth_np[np.where(obstacle_mask_with_valid_depth)]) / 1000

            try:
                hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
            except DetectionError:
                walk_to_pose_in_initial_frame(clients, initial_transforms, HOME)
                continue

            _, regrasp_px = detect_regrasp_point_from_hose(predictions, hose_points, ideal_dist_to_obs=70)
            regrasp_vec2 = np_to_vec2(regrasp_px)
            regrasp_x_in_cam, regrasp_y_in_cam, _ = pixel_to_camera_space(rgb_res, regrasp_px[0], regrasp_px[1],
                                                                          depth=1.0)
            regrasp_x, regrasp_y = pos_in_cam_to_pos_in_hand([regrasp_x_in_cam, regrasp_y_in_cam])

            # BEFORE we grasp, get the robot's position in image space
            transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
            body_in_hand = get_a_tform_b(transforms, HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
            hand_in_gpe = get_a_tform_b(transforms, GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME)
            hand_to_floor = hand_in_gpe.z
            body_in_cam = np.array([-body_in_hand.y, -body_in_hand.z])
            robot_px = np.array(camera_space_to_pixel(rgb_res, body_in_cam[0], body_in_cam[1], hand_to_floor))

            _, place_px = homotopy_planner.plan(rgb_np, predictions, hose_points, regrasp_px, robot_px)

            place_x_in_cam, place_y_in_cam, _ = pixel_to_camera_space(rgb_res, place_px[0], place_px[1], depth=1.0)
            place_x, place_y = pos_in_cam_to_pos_in_hand([place_x_in_cam, place_y_in_cam])

            # Compute the desired poses for the hand
            nearest_obs_height = hand_to_floor - nearest_obs_to_hand
            dplace_x = place_x - regrasp_x
            dplace_y = place_y - regrasp_y

            # Do the grasp
            success = do_grasp(clients, rgb_res, regrasp_vec2)
            if success:
                break

            # reset view
            look_at_scene(clients, z=0.4, pitch=np.deg2rad(85))
            align_with_hose(clients, partial(get_hose_and_regrasp_point, predictor, clients.image))
        else:
            # Give up and reset
            walk_to_pose_in_initial_frame(clients, initial_transforms, HOME)
            continue

        hand_delta_in_body_frame(clients, dx=0, dy=0, dz=nearest_obs_height + 0.2, follow=False)
        hand_delta_in_body_frame(clients, dx=dplace_x, dy=dplace_y, dz=0)
        # Move down to the floor
        transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
        hand_in_gpe = get_a_tform_b(transforms, GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME)
        hand_delta_in_body_frame(clients, dx=0, dy=0, dz=-hand_in_gpe.z + 0.06)
        # Open the gripper
        open_gripper(clients)
        blocking_arm_command(clients, RobotCommandBuilder.arm_stow_command())

        # reset before trying again
        walk_to_pose_in_initial_frame(clients, initial_transforms, HOME)

    return False


def viz_common_frames(snapshot):
    body_in_odom = get_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    gpe_in_odom = get_a_tform_b(snapshot, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    hand_in_odom = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    rr_tform('body', body_in_odom)
    rr_tform('gpe', gpe_in_odom)
    rr_tform('hand', hand_in_odom)


def rr_tform(child_frame: str, tform: math_helpers.SE3Pose):
    translation = np.array([tform.position.x, tform.position.y, tform.position.z])
    rot_mat = tform.rotation.to_matrix()
    rr.log_transform3d(f'frames/{child_frame}', rr.TranslationAndMat3(translation, rot_mat))


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    rr.init("continuous_regrasping_hose_demo")
    rr.connect()
    rr.log_transform3d(f'frames/odom', rr.Translation3D([0, 0, 0]))

    predictor = Predictor('models/hose_regrasping.pth')

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('continuous_regrasping_hose_demo')
    robot = sdk.create_robot('192.168.80.3')
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

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        setup(robot)
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                          image=image_client, raycast=rc_client, command=command_client, robot=robot)

        # open the hand, so we can see more with the depth sensor
        open_gripper(clients)

        while True:
            initial_transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

            try:
                go_to_goal(predictor, clients, initial_transforms)
            except (GraspingException, DetectionError, PlanningException):
                print("Failed to grasp, restarting.")

            open_gripper(clients)
            blocking_arm_command(clients, RobotCommandBuilder.arm_stow_command())

            # Go home, you're done!
            walk_to_pose_in_initial_frame(clients, initial_transforms, math_helpers.SE2Pose(0, 0, 0))

            time.sleep(5)  # TODO: make this much larger for the real demo


if __name__ == '__main__':
    main()
