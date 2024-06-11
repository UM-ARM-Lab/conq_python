# export PYTHONPATH="${PYTHONPATH}:/home/conq/Spot590/conq_python/src"
# export BOSDYN_CLIENT_USERNAME=user && export BOSDYN_CLIENT_PASSWORD=4f9y0eftzh76

# PYTHON 
import numpy as np
import sys
import cv2
import time
import pdb

# BOSDYN: Clients
import bosdyn.client.estop
import bosdyn.client.lease
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.lease import LeaseClient

#BOSDYN: Helpers
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, HAND_FRAME_NAME,GRAV_ALIGNED_BODY_FRAME_NAME

# CONQ: Clients
from conq.clients import Clients

# CONQ: Manipulation modules
from conq.manipulation_lib.Manipulation import move_gripper, open_gripper, close_gripper
from conq.manipulation_lib.Perception3D import VisualPoseAcquirer, PointCloud, Vision
from conq.manipulation_lib.Grasp import get_grasp_candidates, get_best_grasp_pose

# CONQ: Perception modules
# from conq.perception_lib.get_mask import lang_sam
# CONQ: Utils
from conq.manipulation_lib.utils import verify_estop, get_segmask_manual, get_segmask, rotate_quaternion, unstow_arm, stow_arm, stand

from bosdyn.client.image import ImageClient
import open3d as o3d

PCD_PATH = "src/conq/manipulation_lib/gpd/data/PCD/"
NPY_PATH = "src/conq/manipulation_lib/gpd/data/NPY/"
RGB_PATH = "src/conq/manipulation_lib/gpd/data/RGB/"
DEPTH_PATH = "src/conq/manipulation_lib/gpd/data/DEPTH/"
MASK_PATH = "src/conq/manipulation_lib/gpd/data/MASK/"

# Usage:
if __name__ == "__main__":
    sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('acquire_point_cloud')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop(robot)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # lease_client.take()

    # Create instances of vision and point clouds which allows for interfacing with images and point clouds
    vision = Vision(image_client, sources)
    pointcloud = PointCloud(vision)

    # image_sources = image_client.list_image_sources()
    # print("All image sources: ", image_sources)
    while True:
        try:
            lease_client.acquire()
            robot.power_on()
            stand(robot, command_client)
        except bosdyn.client.lease.ResourceAlreadyClaimedError:
            print("Lease currently in use...")

