# export PYTHONPATH="${PYTHONPATH}:/home/conq/Spot590/conq_python/src"
# export BOSDYN_CLIENT_USERNAME=user && export BOSDYN_CLIENT_PASSWORD=4f9y0eftzh76

"""
Test script for grasp execution
Written by Jemuel Stanley Premkumar (jemprem@umich.edu)
"""

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
# CONQ: Utils
from conq.manipulation_lib.utils import verify_estop, get_segmask_manual, get_segmask, rotate_quaternion

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

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                            image=image_client, raycast=rc_client, command=command_client, robot=robot, graphnav=None)

        
        robot.logger.info('Commanding robot to stand...')
        # blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # Deploy the arm
        robot_cmd = RobotCommandBuilder.arm_ready_command()
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)
    
        # Task 1: Look at scene
        # gaze_pose = (0.75,0.0,0.3, 0.7071,0.,0.7071,0)
        gaze_pose = (0.75,0.0,0.3, 1,0.,0.,0.)
        status = move_gripper(clients, gaze_pose, blocking = True, duration = 0.1)
        # Task 1.1: Open gripper
        status = open_gripper(clients)
        time.sleep(0.2)

        vision = Vision(image_client, sources)
        pointcloud = PointCloud(vision)
        
        while True:
            try:
                rgb = vision.get_latest_RGB(path = RGB_PATH,save = True)
                #print("Shape of RGB Image: ", image)
                depth = vision.get_latest_Depth(path = DEPTH_PATH, save = True)
                # depth_img = vision.align_depth_with_RGB(path = DEPTH_PATH, save = True)
                
                xyz = pointcloud.get_raw_point_cloud() # RAW point cloud (N,3)
                print("Shape of xyz: ", np.shape(xyz))
                # Get segmentation mask from Lang-SAM
                # Random segmentation mask
                seg_mask = get_segmask_manual(RGB_PATH+"live.jpg", save_path = MASK_PATH)

                # Segment pointcloud
                pointcloud.segment_xyz(seg_mask)

                pointcloud.save_pcd(path = PCD_PATH)
                pointcloud.save_npy(path = NPY_PATH)
                
                # Call Grasp detection Module
                grasp_pose = get_best_grasp_pose()
                modified_pose = list(grasp_pose)
                # modified_pose[0]-=0.30
                new_grasp_pose = tuple(modified_pose)

                rotated_pose = rotate_quaternion(new_grasp_pose,-90,axis=(0,1,0))
                #rotated_pose = rotate_quaternion(rotated_pose,-90,axis=(0,0,1))
                # Execute grasp
                
                status = move_gripper(clients, rotated_pose, blocking = True, duration = 1)
                status = close_gripper(clients)
                pdb.set_trace()
                break # FIXME: Remove later
            except KeyboardInterrupt:   
                break
            
            except Exception as e:
                print(e)
                break
                
        # print("Stopping threads.")
        # pose_acquirer.stop()
        # controller.stop()
        # pose_acquirer.join()
        #controller.join()

        # Stow the arm
        # Build the stow command using RobotCommandBuilder
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = command_client.robot_command(stow)

        robot.logger.info('Stow command issued.')
        block_until_arm_arrives(command_client, stow_command_id, 3.0)
    

"""
Grasp #2: Score = 798.020996
	Position: [0.773, -0.069, -0.468]
	Orientation (Euler angles): [0.518, -1.815, -2.474] (Roll, Pitch, Yaw)
	Orientation (Quaternion): [w: 0.004, x: 0.771, y: -0.101, z: -0.628]

"""

# Two-spray thingy
"""
Grasp #1: Score = 635.587830
	Position: [0.846, -0.122, -0.454]
	Orientation (Euler angles): [1.835, -2.493, -2.197] (Roll, Pitch, Yaw)
	Orientation (Quaternion): [w: 0.582, x: -0.628, y: 0.037, z: 0.515]
    0.047, -0.470, -0.418, 0.776
"""

# with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
#         # Now, we are ready to power on the robot. This call will block until the power
#         # is on. Commands would fail if this did not happen. We can also check that the robot is
#         # powered at any point.
#         robot.logger.info('Powering on robot... This may take a several seconds.')
#         robot.power_on(timeout_sec=20)
#         assert robot.is_powered_on(), 'Robot power on failed.'
#         robot.logger.info('Robot powered on.')

#         clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
#                             image=image_client, raycast=rc_client, command=command_client, robot=robot, graphnav=None)

        
#         robot.logger.info('Commanding robot to stand...')
#         blocking_stand(command_client, timeout_sec=10)
#         robot.logger.info('Robot standing.')

#         # Deploy the arm
#         robot_cmd = RobotCommandBuilder.arm_ready_command()
#         cmd_id = command_client.robot_command(robot_cmd)
#         block_until_arm_arrives(command_client, cmd_id)
    
#         # Task 1: Look at scene
#         gaze_pose = (0.75,0.0,0.4, 1.,0.,0.,0)
#         status = move_gripper(clients, gaze_pose, blocking = True, duration = 0.1)
#         # Task 1.1: Open gripper
#         status = open_gripper(clients)
#         time.sleep(1)

#         # Task 2: Go to grasp pose hover
#         # Note: Need 90 degree rotation about z-axis to perform actual grasp (weird)
#         # Note: Probably needs alignment with RGB
#         grasp_pose = (0.846, -0.122, 0.0, 0.047, -0.470, -0.418, 0.776)
#         status = move_gripper(clients, grasp_pose, blocking = True, duration = 0.1)

#         # Task 2: Go to grasp pose hover
#         grasp_pose = (0.846, -0.122, -0.3, 0.047, -0.470, -0.418, 0.776)
#         status = move_gripper(clients, grasp_pose, blocking = True, duration = 0.1)

#         # Task 2: Go to grasp pose
#         grasp_pose = (0.846, -0.122, -0.454, 0.047, -0.470, -0.418, 0.776)
#         status = move_gripper(clients, grasp_pose, blocking = True, duration = 0.1)
#         # TASk 2.1: Close gripper
#         status = close_gripper(clients)
#         time.sleep(1)

#         # Task 3: Go to gaze pose
#         gaze_pose = (0.846, -0.122, 0.4, 0.047, -0.470, -0.418, 0.776)
#         status = move_gripper(clients, gaze_pose, blocking = True, duration = 0.1)
#         time.sleep(1)

#         # Task 4: Go to grasp pose
#         grasp_pose = (0.846, -0.122, -0.454, 0.047, -0.470, -0.418, 0.776)
#         status = move_gripper(clients, grasp_pose, blocking = True, duration = 0.1)
#         # TASk 4.1: Close gripper
#         status = open_gripper(clients)
#         time.sleep(1)

#          # Task 5: Look at scene
#         gaze_pose = (0.75,0.0,0.4, 1.,0.,0.,0)
#         status = move_gripper(clients, gaze_pose, blocking = True, duration = 0.1)

#         # Stow the arm
#         # Build the stow command using RobotCommandBuilder
#         stow = RobotCommandBuilder.arm_stow_command()

#         # Issue the command via the RobotCommandClient
#         stow_command_id = command_client.robot_command(stow)

#         robot.logger.info('Stow command issued.')
#         block_until_arm_arrives(command_client, stow_command_id, 3.0)