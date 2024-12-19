import argparse
import os
import sys
import time
from pathlib import Path

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder,
                                         RobotCommandClient,
                                         block_until_arm_arrives,
                                         blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import wrappers_pb2
from PIL import Image

from conq.cameras_utils import DEPTH_SOURCES, RGB_SOURCES, get_color_img
from conq.clients import Clients
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command


def stand_at_place(config):
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('StandAtPlace')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        
        robot.logger.info('Robot powered on.')
        robot.logger.info('Commanding robot to stand...')

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        camera_sources = [
            'back_fisheye_image',
            'frontleft_fisheye_image',
            'frontright_fisheye_image',
            'hand_color_image',
            'left_fisheye_image',
            'right_fisheye_image'
        ]

        output_folder = "/Users/saketpradhan/Desktop/ARMLAB/conq_python/scripts/"

        if not os.path.exists(output_folder):
            print('Error: output folder does not exist: ' + output_folder)
            return

        for camera_source in camera_sources:
            counter = 0        
            image_responses = image_client.get_image_from_sources([camera_source])
            rgb_np, _ = get_color_img(image_client, camera_source)

            while True:
                image_saved_path = os.path.join(output_folder, image_responses[0].source.name + '_{:0>4d}'.format(counter) + '.jpg')
                counter += 1

                if not os.path.exists(image_saved_path): break

            cv2.imwrite(image_saved_path, rgb_np)
            print('Wrote: ' + image_saved_path)
            time.sleep(0.7)

        robot.logger.info('Finished.')
        robot.logger.info('Sitting down and turning off.')


def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--folder', help='Path to write images to', default='')
    options = parser.parse_args(argv)

    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    try:
        stand_at_place(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)