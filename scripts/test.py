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

    lease_client.take()

    # Create instances of vision and point clouds which allows for interfacing with images and point clouds
    vision = Vision(image_client, sources)
    pointcloud = PointCloud(vision)

    # image_sources = image_client.list_image_sources()
    # print("All image sources: ", image_sources)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        clients = Clients(  lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                            image=image_client, raycast=rc_client, command=command_client, robot=robot, graphnav=None)
        
        robot.logger.info('Commanding robot to stand...')
        # blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')
        stand(robot, command_client)

        # Deploy the arm
        unstow_arm(robot, command_client)

        # Task 1: Look at scene
        gaze_pose = (0.75,0.0,0.3, 0.7071,0.,0.7071,0)

        # gaze_pose = (0.75,0.0,0.3, 1,0.,0.,0.)
        status = move_gripper(clients, gaze_pose, blocking = True, duration = 0.1)
        
        time.sleep(2)
        
        status = open_gripper(clients)

        # These are automatically saved in the gpd folder which the docker contain has access to
        rgb = vision.get_latest_RGB(path = RGB_PATH,save = True)
        depth = vision.get_latest_Depth(path = DEPTH_PATH, save = True)
        xyz = pointcloud.get_raw_point_cloud() # RAW point cloud (N,3)

        # Segment
        seg_mask = get_segmask_manual(RGB_PATH+"live.jpg", save_path = MASK_PATH)
        print(seg_mask)
        # Segment pointcloud
        # Gets a point cloud that corresponds to ONLY the pixels that exist in the segmentation mask
        pointcloud.segment_xyz(seg_mask) #.squeeze().numpy())


        pointcloud.save_pcd(path = PCD_PATH) # pcd = point cloud file
        
        # Call Grasp detection Module
        grasp_pose = get_best_grasp_pose() # looks at live.pcd and finds the x y z quat position
        modified_pose = list(grasp_pose)
        new_grasp_pose = tuple(modified_pose) # converts the best grasp pose into something usable

        # rotated_pose = rotate_quaternion(new_grasp_pose,-90,axis=(0,1,0))
        status = move_gripper(clients, new_grasp_pose, blocking = True, duration = 1)
        status = close_gripper(clients)

        # gaze_pose = (0.75,0.0,0.3, 1,0.,0.,0.)
        status = move_gripper(clients, gaze_pose, blocking = True, duration = 0.1)

        time.sleep(3)

        # Stow the arm
        stow_arm(robot, command_client)
        




        
