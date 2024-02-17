# Copyright (c) 2024 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""
Tutorial to walk the robot to an object by giving a text prompt, usually in preparation for manipulation.

Example usage (to got to a ball in the scene):
- python3 scripts/speech_prompt_walk_to_object.py 192.168.80.3 --image-source hand_color_image 
"""

import os
import sys
import time
import argparse
import subprocess

import cv2
import numpy as np
from google.protobuf import wrappers_pb2

from pathlib import Path
from PIL import Image

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

from conq.clients import Clients
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command
from conq.cameras_utils import get_color_img, RGB_SOURCES, DEPTH_SOURCES


g_image_click, g_image_display = None, None


def walk_to_object(config):
    """Get an image and command the robot to walk up to a selected object.
       We'll walk "up to" the object, not on top of it.  The idea is that you
       want to interact or manipulate the object."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('WalkToObjectClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.

        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.

        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        status = open_gripper(Clients) #FIXME #BUG: Open the gripper before capturing image

        # Take a picture with a camera
        robot.logger.info('Getting an image from: %s', config.image_source)
        image_responses = image_client.get_image_from_sources([config.image_source])

        if len(image_responses) != 1:
            print(f'Got invalid number of images: {len(image_responses)}')
            print(image_responses)
            assert False

        image = image_responses[0]

        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16: dtype = np.uint16
        else: dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else: img = cv2.imdecode(img, -1)

        #FIXME: No need to save image locally, optimize this code
        image_pil = Image.fromarray(img)
        image_pil.save('image_capture.jpg') 

        global g_image_click, g_image_display
        
        #FIXME: Precautionary measure, to be dealt with later
        time.sleep(1)
        # text_prompt = "ball" #FIXME: Forgot to remove this in the last test, TBD later

        print('Done till process 1')

        text_prompt_langsam = str(run_chatgpt_integration())

        print('Done till process 2')
        
        centroid_x, centroid_y = run_test_script(text_prompt_langsam)

        if centroid_x is not None and centroid_y is not None: print("Centroid coordinates in main:", centroid_x, centroid_y)
        else: print("Failed to obtain centroid coordinates.")
        g_image_click = [centroid_x, centroid_y]

        robot.logger.info('Walking to object at image location (%s, %s)', g_image_click[0],
                          g_image_click[1])

        walk_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # Optionally populate the offset distance parameter.
        if config.distance is None: offset_distance = None
        else: offset_distance = wrappers_pb2.FloatValue(value=config.distance)

        # Build the proto
        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole, offset_distance=offset_distance)

        # Ask the robot to pick up the object
        walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
            walk_to_object_in_image=walk_to)

        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=walk_to_request)

        # Get feedback from the robot
        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                  manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                break

        robot.logger.info('Finished.')
        robot.logger.info('Sitting down and turning off.')

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')


def run_chatgpt_integration():
    """Setup command to run ChatGPT integration code from within a conda environment"""

    script_directory = "/Users/saketpradhan/Desktop/chatgpt-integration/"
    command = [
        "conda", "run", "-n", "chatgpt-env", "python3 chatgpt_integration.py --speech-lang en-US"
    ]

    try: 
        os.chdir(script_directory) # Change directory to where chatgpt_integration.py is located

        print('went to the directory!!')

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print('got the result!!')

        output_lines = result.stdout.splitlines()
        text_prompt_line  = output_lines[-1]

        _, _, lang_sam_text_prompt = text_prompt_line.split(' ')
        return lang_sam_text_prompt

    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return "ball"


def run_test_script(text_prompt):
    """Setup command to run langsam from within a conda environment"""
    
    #FIXME: Remove this, set relative(remote?)(S3?)(Modal?) paths for everything
    #TODO: lang-sam setup and environment
    #HACK: Make this a microservice; host somewhere TBD

    script_directory = "/Users/saketpradhan/Desktop/lang-segment-anything/lang-segment-anything/" 
    command = [
        "conda", "run", "-n", "lang-sam", "python3", "test_script.py", "--conq-image-source", 
        "/Users/saketpradhan/Desktop/ARMLAB/conq_python/scripts/image_capture.jpg", "--text-prompt", text_prompt
    ] 

    try:
        os.chdir(script_directory)  # Change directory to where test_script.py is located
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.splitlines()
        centroid_line = output_lines[-2]

        _, _, cx_str, cy_str = centroid_line.split(' ')
        centroid_x = int(cx_str)
        centroid_y = int(cy_str)

        return centroid_x, centroid_y
    
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None, None
    

#BUG #FIXME: Open the gripper before capturing image
def open_gripper(clients: Clients):
    """Open the arm gripper so that the image view is not occluded by the claw"""

    try:
        clients.command.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(1) 
        return True
    except: return False


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        """
        Draw some lines on the image.
        print('mouse', x, y)
        """
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to walk up to something'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


def arg_float(x):
    try: x = float(x)
    except ValueError: raise argparse.ArgumentTypeError(f'{repr(x)} not a number')
    return x


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='hand_color_image')
    parser.add_argument('-d', '--distance', help='Distance from object to walk to (meters).',
                        default=None, type=arg_float)
    # parser.add_argument('-p', '--text-prompt', help='Action instruction to be performed by Spot.', 
    #                     default='ball', required=True)
    options = parser.parse_args(argv)

    try:
        walk_to_object(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
