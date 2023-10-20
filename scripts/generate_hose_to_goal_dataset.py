#!/usr/bin/env python3

import argparse
import time
from functools import partial
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rerun as rr
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_state import RobotStateClient

from conq.cameras_utils import pos_in_cam_to_pos_in_hand
from conq.clients import Clients
from conq.data_recorder import ConqDataRecorder
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command, grasp_point_in_image
from conq.manipulation import open_gripper
from conq.utils import setup_and_stand
from regrasping_demo.detect_regrasp_point import min_angle_to_x_axis
from regrasping_demo.get_detections import get_hose_head_grasp_point
from regrasping_demo.rotate_about import rotate_around_point_in_hand_frame


def look_at_scene(clients: Clients, x=0.56, y=0.1, z=0.55, pitch=0., yaw=0., dx=0., dy=0., dpitch=0.,
                  dyaw=0.):
    look_cmd = hand_pose_cmd(clients, x + dx, y + dy, z, 0, pitch + dpitch, yaw + dyaw, duration=0.5)
    blocking_arm_command(clients, look_cmd)


def align_with_hose(clients: Clients, get_point_f):
    pick_res = get_point_f()
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


def retry_grasp_hose(clients: Clients, get_point_f):
    for _ in range(5):
        grasp_res = get_point_f()

        # first just try the automatic grasp
        success = grasp_point_in_image(clients, grasp_res.rgb_res, grasp_res.best_vec2)
        if success:
            return

        # If that fails try aligning with the hose
        look_at_scene(clients, z=0.4, pitch=np.deg2rad(85))
        align_with_hose(clients, get_point_f)
        grasp_res = get_point_f()
        success = grasp_point_in_image(clients, grasp_res.rgb_res, grasp_res.best_vec2)
        if success:
            return


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    rr.init("generate_dataset")
    rr.connect()

    from arm_segmentation.predictor import Predictor
    predictor = Predictor('models/hose_regrasping.pth')

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('generate_dataset')
    # robot = sdk.create_robot('192.168.80.3')
    # robot = sdk.create_robot('10.0.0.3')
    robot = sdk.create_robot('10.10.10.135')
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

        for mode, num_episodes in [('train', 2), ('val', 1)]:
            for episode_idx in range(num_episodes):
                print(f"Starting {episode_idx}")
                recorder.start_episode(mode)
                clients.recorder.add_instruction("grasp hose")

                retry_grasp_hose(clients, partial(get_hose_head_grasp_point, predictor, image_client))

                recorder.next_episode()
        recorder.stop()


if __name__ == '__main__':
    main()
