"""
Spot will look for a volleyball. The ball should be placed on the ground about 1.5m in front of of the robot.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
import cv2
import numpy as np
import rerun as rr
from PIL import Image
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from roboflow import Roboflow

from arm_segmentation.predictor import Predictor
from conq.cameras_utils import get_color_img, get_depth_img
from conq.fan import try_reduce_fan
from conq.hand_motion import randomized_look, hand_pose_cmd
from conq.manipulation import open_gripper, blocking_arm_command
from conq.utils import setup_and_stand
from regrasping_demo.detect_regrasp_point import get_masks

rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
project = rf.project("volleyball-rjf9c")


def save_and_upload(rgb_np):
    now = int(time.time())
    rgb_filename = f"volleyball_data/{now}_rgb.png"
    Image.fromarray(rgb_np).save(rgb_filename)
    project.upload(rgb_filename)


def gather_data(power_client, command_client, robot_state_client, image_client):
    img_counter = 0

    def _save_data():
        nonlocal img_counter
        rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
        save_and_upload(rgb_np)

        img_counter += 1

        time.sleep(1)

        return img_counter > 10

    try_reduce_fan(power_client)
    randomized_look(command_client, robot_state_client, _save_data, 0.75, 0, 0.35, np.deg2rad(35), 0)


def look_at_ball(predictor, power_client, command_client, robot_state_client, image_client):
    home_pitch = np.deg2rad(35)
    home_x = 0.75
    home_y = 0
    home_z = 0.45
    look_cmd = hand_pose_cmd(robot_state_client, home_x, home_y, home_z, 0, home_pitch, 0, duration=1)
    blocking_arm_command(command_client, look_cmd)

    i = 0
    while True:
        try_reduce_fan(power_client)

        rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
        depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
        predictions = predictor.predict(rgb_np, min_score_threshold=0.6)

        if i % 100 == 0:
            save_and_upload(rgb_np)

        visualize_predictions(predictions, predictor, rgb_np, sleep_ms=1)

        centroid_in_body = get_centroid_in_body(predictions, depth_np, rgb_res)
        if centroid_in_body is None:
            print("No ball found!")
            continue

        gaze_command = RobotCommandBuilder.arm_gaze_command(*centroid_in_body, BODY_FRAME_NAME,
                                                            max_linear_vel=1,
                                                            max_angular_vel=1,
                                                            max_accel=0.1)
        # Non-blocking
        command_client.robot_command(lease=None, command=gaze_command, end_time_secs=5)

        i += 1


def get_centroid_in_body(predictions, depth_np, rgb_res):
    masks = get_masks(predictions, "volleyball")
    if len(masks) == 0:
        print("No ball found!")
        return None

    mask = masks[0]
    ys, xs = np.where(mask)
    depths = depth_np[ys, xs] / 1000
    valid_depths = depths[np.nonzero(depths)[0]]

    if len(valid_depths) == 0:
        print("no depth data!")
        return None

    avg_depth = np.mean(valid_depths)
    centroid = np.stack([xs, ys], -1).mean(0)
    centroid_in_cam = np.array(pixel_to_camera_space(rgb_res, centroid[0], centroid[1], avg_depth))
    cam_to_body = get_a_tform_b(rgb_res.shot.transforms_snapshot, BODY_FRAME_NAME, rgb_res.shot.frame_name_image_sensor)
    centroid_in_body = cam_to_body * math_helpers.Vec3(*centroid_in_cam)
    return centroid_in_body


def visualize_predictions(predictions, predictor, rgb_np, sleep_ms: int = 1):
    viz_img = rgb_np.copy()
    for pred in predictions:
        mask = pred['mask']
        color = (np.array(predictor.colors[pred['class']]) * 255).astype(np.uint8)
        masked_img = np.where(mask[..., None] > 0.5, color, viz_img).astype(np.uint8)
        viz_img = cv2.addWeighted(viz_img, 0.2, masked_img, 0.8, 0)
    cv2.imshow('hand_color_image', cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(sleep_ms)


def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    args = parser.parse_args(argv)

    bosdyn.client.util.setup_logging(args.verbose)

    rr.init('volleyball')
    rr.connect()

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('DemoClient')
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    Path(f"volleyball_data").mkdir(exist_ok=True, parents=True)

    assert robot.has_arm(), "Robot requires an arm to run this example."

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, such as the" \
                                    " estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    power_client = robot.ensure_client(PowerClient.default_service_name)

    lease_client.take()

    model_path = Path("models/volleyball.pth")
    predictor = Predictor(model_path)

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        command_client = setup_and_stand(robot)
        open_gripper(command_client)

        look_at_ball(predictor, power_client, command_client, robot_state_client, image_client)

        gather_data(power_client, command_client, robot_state_client, image_client)


if __name__ == '__main__':
    # Checks to see IP address is specified
    if not main(sys.argv[1:]):
        sys.exit(1)
