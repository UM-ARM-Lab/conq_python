"""
Add in $ROOT/scripts/
Create $ROOT/data/RGB directory

To run:
- python3 scripts/collect_ig.py 192.168.80.3 --to-depth --gaze-x 0.75 --gaze-y 0.0 --gaze-z 0.4 --gaze-roll 0.0 --gaze-yaw 0.0

default --gaze-pose: 
0.75, 0.0, 0.4, np.pi/2, 0.0, 0
"""

import argparse
import sys
import cv2
import time
import numpy as np

import bosdyn.client.estop
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
from bosdyn.api import estop_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command

from pathlib import Path
from PIL import Image

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from conq.cameras_utils import get_color_img, RGB_SOURCES, DEPTH_SOURCES
from conq.clients import Clients


# FIXME: To be ignored during push (Stanley's code to be used only for image collection)
def move_to(clients: Clients, pose, duration = 1, follow=False):
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
        
        blocking_arm_command(clients, arm_cmd)
        print("Arm command done")
        return True
    except Exception as e:
        print(e)
        return False
    

def open_gripper(clients: Clients):
    try:
        clients.command.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(1)  # FIXME: how to block on a gripper command?
        return True
    except: return False


def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)


def main(argv):
    # Parse args
    parser = argparse.ArgumentParser(description="Collect images of Garden tools from Conq's perspective")
    bosdyn.client.util.add_base_arguments(parser)
    
    parser.add_argument('--to-depth',
                        help='Convert to the depth frame. Default is convert to visual.',
                        action='store_true')
    
    parser.add_argument('--camera', help='Camera to acquire image from.', default='hand',\
                        choices=['frontleft', 'frontright', 'left', 'right', 'back','hand',
                        ])
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')

    parser.add_argument('--gaze-x', required=False, type=float, default=0.75, help='X position for gaze pose')
    parser.add_argument('--gaze-y', required=False, type=float, default=0.0, help='Y position for gaze pose')
    parser.add_argument('--gaze-z', required=False, type=float, default=0.4, help='Z position for gaze pose')
    parser.add_argument('--gaze-pitch', required=False, default=np.pi/2, help='Pitch for gaze pose')
    parser.add_argument('--gaze-roll', required=False, type=float, default=0.0, help='Roll for gaze pose')
    parser.add_argument('--gaze-yaw', required=False, type=float, default=0.0, help='Yaw for gaze pose')

    # parser.add_argument("--gaze-pose", required=True, help="defines where the Conq's arm camera looks at", action='store_true')

    options = parser.parse_args(argv)

    if options.to_depth: sources = [options.camera + '_depth', options.camera +'_color_image']
    else: sources = [options.camera + 'frontleft_depth_in_visual_frame', options.camera + '_fisheye_image']

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('acquire_point_cloud')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped and that an external application has registered and holds an estop endpoint.
    verify_estop(robot)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_client.take()


    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                            image=image_client, raycast=rc_client, command=command_client, robot=robot, graphnav = None)

        
        robot.logger.info('Commanding robot to stand...')
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')


        # Deploy the arm
        robot_cmd = RobotCommandBuilder.arm_ready_command()
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)


        # Task 1: Look at scene
        gaze_pose = (options.gaze_x, options.gaze_y, options.gaze_z, options.gaze_pitch, options.gaze_roll, options.gaze_yaw)
        print(gaze_pose)
        
        status = move_to(clients, gaze_pose, duration = 2, follow=False)

        status = open_gripper(clients)
        time.sleep(1)


        # Live RGB
        start_time = time.time()
        c = 1
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        filenames = []
        count=0

        while True:
            count+=1
            rgb_np, _ = get_color_img(image_client, sources[1]) #(0 -> depth, 1 -> RGB)
            rgb_np = np.array(rgb_np, dtype=np.uint8)

            img_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR) 
            cv2.imshow("RGB", img_bgr)

            store = input("Do you want to store this? \n Y/N")
            if store == "Y":
                filename = Path(f"data/RGB/{count}.jpg")
                Image.fromarray(rgb_np).save(filename)
                filenames.append(filename)
            else: continue
            if count>=10: break
        
        
if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)