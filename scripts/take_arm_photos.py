import bosdyn.client.estop
import bosdyn.client.lease


from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (RobotCommandClient, block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient
from bosdyn.client.image import build_image_request
from bosdyn.client.lease import LeaseClient
from bosdyn.api import image_pb2

from conq.manipulation_lib.Manipulation import grasped_bool, open_gripper, close_gripper, move_gripper, move_gripper
from conq.manipulation_lib.Perception3D import VisualPoseAcquirer, PointCloud, Vision
from conq.perception_lib.fast_owlsam import FastOwlsam
from conq.perception_lib.fast_grounded_sam import FastGroundedSAM
from conq.clients import Clients


import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, depth_image_to_pointcloud, _depth_image_data_to_numpy
from conq.cameras_utils import get_color_img, get_depth_img, pos_in_cam_to_pos_in_hand, image_to_opencv, RGB_SOURCES, DEPTH_SOURCES

import time
import numpy as np
import cv2

def spot_hand_click(object_str, viewpoint_str):

    src = "hand_color_image"
    location = "/home/adibalaji/Desktop/agrobots/conq_python/data/dataset_imgs/"

    rgb_request = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
    rgb_response= image_client.get_image([rgb_request])[0]
    rgb_np = image_to_opencv(rgb_response, auto_rotate=True)
    image = np.array(rgb_np,dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(location + src + f"_{object_str}_{viewpoint_str}.jpg", image)


if __name__ == "__main__":

    sdk = bosdyn.client.create_standard_sdk('SemanticGrasperTest')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot) 
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    lease_client.take()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client, image=image_client, raycast=rc_client, command=command_client, robot=robot)
    
    picture_poses = {
        # "top" : (0.85,0.0, -0.010, 0.7071, 0.0, 0.7071, 0.0),
        # "top_right" : (0.85,-0.10, -0.010, 0.683, 0.183, 0.683, 0.183),
        # "top_left" : (0.85,0.10, -0.010, 0.683, -0.183, 0.683, -0.183),
        # "top_front" : (0.95,0.0, -0.010, -0.500, 0.0, -0.866, 0.0),
        # "top_back" : (0.75,0.0, -0.010, 0.866, 0.0, 0.500, 0.0),
        "front" : (0.85, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0),
        "front_left" : (0.85, 0.10, 0.30, 0.966, 0.0, 0.0, -0.259),
        "front_right" : (0.85, -0.10, 0.30, 0.966, 0.0, 0.0, 0.259),
        "front_back" : (0.75, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0),
        "front_left_back" : (0.75, 0.10, 0.30, 0.966, 0.0, 0.0, -0.259),
        "front_left_right" : (0.75, -0.10, 0.30, 0.966, 0.0, 0.0, 0.259),
    }

    object = "drill"

    default_pose = (0.50,0.0, 0.45, 1, 0, 0, 0)
    status = open_gripper(clients)
    status = move_gripper(clients, pose=default_pose, blocking=False, duration=1)
    time.sleep(2)

    for viewpoint, pose in picture_poses.items():

        status = move_gripper(clients, pose=pose, blocking=True, duration=0.75)
        print(f"Taking pic from {viewpoint}..")
        time.sleep(0.5)
        spot_hand_click(object_str=object, viewpoint_str=viewpoint)

    status = move_gripper(clients, pose=default_pose, blocking=False, duration=1)
    status = close_gripper(clients)
    time.sleep(2)
