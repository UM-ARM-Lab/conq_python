import bosdyn.api
import numpy as np
import requests
import base64
from openai import OpenAI
import json
import os
import base64
import math

from dotenv import load_dotenv

from conq.navigation.graph_nav.waypoint_photographer import WaypointPhotographer

import argparse
import sys
import open3d as o3d
import cv2
import time
import numpy as np
import pdb
import torch

from google.protobuf import any_pb2, wrappers_pb2

from conq.manipulation_lib.Manipulation import grasped_bool, open_gripper, close_gripper, move_gripper, move_gripper
from conq.manipulation_lib.Perception3D import VisualPoseAcquirer, PointCloud, Vision

import bosdyn.client.estop
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME,GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, BODY_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import trajectory_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus

from conq.manipulation_lib.Manipulation import grasped_bool, open_gripper, close_gripper, move_gripper, move_gripper
from conq.manipulation_lib.Perception3D import VisualPoseAcquirer, PointCloud, Vision

import bosdyn.client.estop
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client import math_helpers as client_math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.util import seconds_to_duration

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, depth_image_to_pointcloud, _depth_image_data_to_numpy
from conq.cameras_utils import get_color_img, get_depth_img, pos_in_cam_to_pos_in_hand, image_to_opencv, RGB_SOURCES, DEPTH_SOURCES
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME

from conq.manipulation import grasp_point_in_image
from conq.clients import Clients
from conq.manipulation_lib.utils import stow_arm
import matplotlib.pyplot as plt
from conq.owlsam import OwlSam
from conq.perception_lib.grounded_sam_inference import GroundedSAM
from conq.grounding_dino import GroundingDino
from conq.cameras_utils import image_to_opencv

from conq.manipulation import grasp_point_in_image
from conq.manipulation_lib.Grasp import get_grasp_candidates, get_best_grasp_pose
from conq.manipulation_lib.utils import rotate_quaternion
from conq.clients import Clients
from conq.manipulation_lib.utils import stow_arm
from conq.perception_lib.grounded_sam_inference import GroundedSAM
from conq.cameras_utils import image_to_opencv

PCD_PATH = "/Users/adibalaji/Desktop/agrobots/conq_python/src/conq/manipulation_lib/gpd/data/PCD/"
NPY_PATH = "/Users/adibalaji/Desktop/agrobots/conq_python/src/conq/manipulation_lib/gpd/data/NPY/"
RGB_PATH = "/Users/adibalaji/Desktop/agrobots/conq_python/src/conq/manipulation_lib/gpd/data/RGB/"
DEPTH_PATH = "/Users/adibalaji/Desktop/agrobots/conq_python/src/conq/manipulation_lib/gpd/data/DEPTH/"
MASK_PATH = "/Users/adibalaji/Desktop/agrobots/conq_python/src/conq/manipulation_lib/gpd/data/MASK/"

ORIENTATION_MAP = {
    'back_fisheye_image':               (-1.00,0.0,0.0, 0.7071,0.,0.7071,0),
    'frontleft_fisheye_image':          (0.75,0.0,0.0, 0.7071,0.,0.7071,0),
    'frontright_fisheye_image':         (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'left_fisheye_image':               (0,0.65,0.1, 0.924,0.0,0.0,0.383),
    'right_fisheye_image':              (0,-0.65,0.1, 0.7071,0.,0.7071,0),
    'straight_up':                      (0.5,0,0.85, 1.0,0,0,0),
    'put_down':                         (0.75,0,-0.30, 0.7071,0.,0.7071,0),
    'hand_search':                      (0.55,0.0,0.65, 0.819,0.0,0.574,0.0),
    'hand_search_forward':              (0.8,0.0,0.15, 0.819,0.0,0.574,0.0),
    'find_grasp_front':                 (0.75,0.0,0.25, 0.7071,0.,0.7071,0)
}

# Mapping from visual to depth data
VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE = {
    # 'frontleft_fisheye_image': 'frontleft_depth_in_visual_frame',
    # 'frontright_fisheye_image': 'frontright_depth_in_visual_frame'
    # 'left_fisheye_image' : 'left_depth_in_visual_frame',
    # 'back_fisheye_image' : 'back_depth_in_visual_frame',
    # 'right_fisheye_image': 'right_depth_in_visual_frame'
    "hand_color_image" : 'hand_depth'
}

ROTATION_ANGLES = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'hand_color_image' : 0
}

class SemanticGrasper:

    def __init__(self, robot):

        load_dotenv('.env.local')

        self.robot = robot
        self.MY_API_KEY = os.getenv('GPT_KEY')
        self.ORG_KEY = os.getenv('ORG_KEY')
        self.client = OpenAI(organization=self.ORG_KEY, api_key=self.MY_API_KEY)

        self.images_loc = os.getenv('MEMORY_IMAGE_PATH')

        self.lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self.manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.rc_client = robot.ensure_client(RayCastClient.default_service_name)
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        self.clients = Clients(lease=self.lease_client, state=self.robot_state_client, manipulation=self.manipulation_api_client, image=self.image_client, raycast=self.rc_client, command=self.command_client, robot=self.robot)

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    #verify estop
    def verify_estop(self):
        """Verify the robot is not estopped"""

        client = self.robot.ensure_client(EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                            ' estop SDK example, to configure E-Stop.'
            self.robot.logger.error(error_message)
            raise Exception(error_message)
        
    def put_down(self):
        clients = Clients(lease=self.lease_client, state=self.robot_state_client, manipulation=self.manipulation_api_client,
                image=self.image_client, raycast=self.rc_client, command=self.command_client, robot=self.robot, graphnav=None)
        status = move_gripper(clients=clients, pose=ORIENTATION_MAP['put_down'], blocking=True, duration = 2)
        time.sleep(0.5)
        status = open_gripper(clients)
        status = move_gripper(clients=clients, pose=ORIENTATION_MAP['frontleft_fisheye_image'], blocking=True, duration = 0.3)


    def save_image_to_local(self, image, source_name):
        directory = os.getenv('MEMORY_IMAGE_PATH')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{source_name}.jpg")
        cv2.imwrite(file_path, image)
        print(f"Image saved to {file_path}")
        return file_path

    def relative_move(self, dx, dy, dyaw, frame_name, stairs=False):
        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # Build the transform for where we want the robot to be relative to where the body currently is.
        body_tform_goal = client_math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
        # We do not want to command this goal in body frame because the body will move, thus shifting
        # our goal. Instead, we transform this offset to get the goal position in the output frame
        # (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # Command the robot to go to the goal point in the specified frame. The command will stop at the
        # new position.
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
            frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
        end_time = 10.0
        cmd_id = self.command_client.robot_command(lease=None, command=robot_cmd,
                                                    end_time_secs=time.time() + end_time)
        # Wait until the robot has reached the goal.
        while True:
            feedback = self.command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print('Failed to reach the goal')
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print('Arrived at the goal.')
                return True
            time.sleep(0.25)

        return True

    def walk_to_pixel(self, rgb_response, walk_x, walk_y):
            walk_vec = geometry_pb2.Vec2(x= walk_x, y=walk_y)
            offset_distance = wrappers_pb2.FloatValue(value=0.75)

            # Build the proto
            walk_to = manipulation_api_pb2.WalkToObjectInImage(
                pixel_xy=walk_vec, transforms_snapshot_for_camera=rgb_response.shot.transforms_snapshot,
                frame_name_image_sensor=rgb_response.shot.frame_name_image_sensor,
                camera_model=rgb_response.source.pinhole, offset_distance=offset_distance)

            # Ask the robot to pick up the object
            walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
                walk_to_object_in_image=walk_to)

            # Send the request
            cmd_response = self.manipulation_api_client.manipulation_api_command(
                manipulation_api_request=walk_to_request)

            # Get feedback from the robot
            while True:
                time.sleep(0.25)
                feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=cmd_response.manipulation_cmd_id)

                # Send the request
                response = self.manipulation_api_client.manipulation_api_feedback_command(
                    manipulation_api_feedback_request=feedback_request)

                print('Current state: ',
                    manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

                if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                    break    

    def search_object_with_gripper(self,text):

        self.robot.logger.info('Powering on robot... This may take a several seconds.')
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'
        self.robot.logger.info('Robot powered on.')

        self.robot.logger.info('Commanding robot to stand...')
        blocking_stand(self.command_client, timeout_sec=10)
        self.robot.logger.info('Robot standing.')
        open_gripper(self.clients)


        robot_state = self.robot_state_client.get_robot_state()

        detector = GroundingDino()
        best_score = 0
        best_angle_index = 0

        gaze_pose = ORIENTATION_MAP['hand_search']
        status = move_gripper(self.clients, gaze_pose, blocking=True, duration=0.5)
        time.sleep(0.5) 


        for i in range(12):

            #upper view
            source_name = "hand_color_image"
            
            rgb_request = build_image_request(source_name, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
            rgb_response= self.image_client.get_image([rgb_request])[0]
            rgb_np = image_to_opencv(rgb_response, auto_rotate=True)
            image = np.array(rgb_np,dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_path = self.save_image_to_local(image, f'hand_search_{i}')
            
            boxes,scores = detector.predict_box_score_with_text(image_path, text)
            if boxes.any():
                for box,score in zip(boxes,scores):
                    if score>best_score:
                        best_score = score
                        best_angle_index = i

            #lower forwards view
            gaze_pose = ORIENTATION_MAP['hand_search_forward']
            status = move_gripper(self.clients, gaze_pose, blocking=True, duration=0.25)
            time.sleep(0.25) 

            rgb_request = build_image_request(source_name, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
            rgb_response= self.image_client.get_image([rgb_request])[0]
            rgb_np = image_to_opencv(rgb_response, auto_rotate=True)
            image = np.array(rgb_np,dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_path = self.save_image_to_local(image, f'hand_search_forward_{i}')
            
            boxes,scores = detector.predict_box_score_with_text(image_path, text)
            if boxes.any():
                for box,score in zip(boxes,scores):
                    if score>best_score:
                        best_score = score
                        best_angle_index = i


            #ROTATE BODY
            gaze_pose = ORIENTATION_MAP['hand_search']
            status = move_gripper(self.clients, gaze_pose, blocking=False, duration=0.5)
            self.relative_move(0, 0,  math.radians(30), ODOM_FRAME_NAME)
            time.sleep(0.25)
        
        #go back to the best view point
        back_angel = [0,30,60,90,120,150,180,-150,-120,-90,-60,-30]
        self.relative_move(0, 0,  math.radians(back_angel[best_angle_index]), ODOM_FRAME_NAME)
        move_gripper(self.clients, ORIENTATION_MAP["hand_search"], blocking=False, duration=0.5)
        time.sleep(0.5)

        #go to the detected object in view
        source_name = "hand_color_image"
        rgb_request = build_image_request(source_name, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
        rgb_response= self.image_client.get_image([rgb_request])[0]
        rgb_np = image_to_opencv(rgb_response, auto_rotate=True)
        image = np.array(rgb_np,dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = self.save_image_to_local(image, f'hand_walk_found_object')
        boxes,scores = detector.predict_box_score_with_text(image_path, text)
        max_score_for_walk = 0
        best_centroid_for_walk = None
        if boxes.any():
            for box,score in zip(boxes,scores):

                if score>max_score_for_walk:
                    max_score_for_walk = score
                    best_centroid_for_walk =  detector.compute_box_centroid(box)
            
            walk_x, walk_y = best_centroid_for_walk
            self.walk_to_pixel(rgb_response=rgb_response, walk_x=walk_x, walk_y=walk_y)
            
        else:
            print("Nothing in view!!!")

    def create_rotation_quat(self,degrees):
        radians = math.radians(degrees)
        return math_helpers.Quat.from_yaw(radians)

    #orient gripper along with object direction
    def orient_and_grasp(self, object_direction_name,object_name):
        assert self.robot.has_arm(), 'Robot requires an arm to run this example.'
        self.verify_estop()
        self.lease_client.take()
        gds = GroundedSAM()

        sources = ["hand_depth_in_hand_color_frame", "hand_color_image"]
        vision = Vision(self.image_client, sources)
        pointcloud = PointCloud(vision)

        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=False):
            
            clients = Clients(lease=self.lease_client, state=self.robot_state_client, manipulation=self.manipulation_api_client,
                image=self.image_client, raycast=self.rc_client, command=self.command_client, robot=self.robot, graphnav=None)
            
            self.robot.logger.info('Commanding robot to stand...')
            blocking_stand(self.command_client, timeout_sec=10)
            self.robot.logger.info('Robot standing.')
            
            # Deploy the arm
            robot_cmd = RobotCommandBuilder.arm_ready_command()
            cmd_id = self.command_client.robot_command(robot_cmd)
            block_until_arm_arrives(self.command_client, cmd_id)

            grasp_result = False
            while grasp_result is False:

                #look at scene
                gaze_pose = ORIENTATION_MAP[object_direction_name]
                status = move_gripper(clients, gaze_pose, blocking=False, duration = 0.5)
                status = open_gripper(clients)
                time.sleep(3)

                image_responses = self.image_client.get_image_from_sources(sources)
                rgb = vision.get_latest_RGB(path=self.images_loc, save=True, file_name='live_hand')


                # ------------------------------------- USING BOSDYN STOCK PIXEL GRASP -----------------------------------------------------------------------------------
                pred_mask = gds.predict_segmentation(image_path=self.images_loc+'live_hand.jpg', text = object_name)
                pred_centroid = gds.compute_mask_centroid(pred_mask)


                pix_x, pix_y = (pred_centroid[0],pred_centroid[1]) # Get from object detector
                pick_vec = geometry_pb2.Vec2(x=pix_x, y=pix_y)

                try:
                    grasp_result = grasp_point_in_image(clients,image_responses[0],pick_vec)
                except Exception as e:
                    print("Whoops")
                    close_gripper(clients=clients)
                    stow_arm(self.robot, self.command_client)
                # --------------------------------------------------------------------------------------------------------------------------------------------

                # ------------------------------------- USING GPD --------------------------------------------------------------------------------------------

                # depth = vision.get_latest_Depth(path = DEPTH_PATH, save = True)
                # xyz = pointcloud.get_raw_point_cloud()
                # seg_mask = gds.predict_segmentation(image_path=self.images_loc+'live_hand.jpg', text = object_name).squeeze()
                # print(f'Using mask found of shape {seg_mask.shape}')
                # pointcloud.segment_xyz(seg_mask.squeeze())

                # pointcloud.save_pcd(path = PCD_PATH)
                # pointcloud.save_npy(path = NPY_PATH)
                
                # # Call Grasp detection Module
                # grasp_pose = get_best_grasp_pose()
                
                # status = move_gripper(clients, rotate_quaternion(grasp_pose), blocking = True, duration = 1)
                # status = close_gripper(clients)

                # --------------------------------------------------------------------------------------------------------------------------------------------
            
            status = move_gripper(clients, ORIENTATION_MAP[object_direction_name], blocking=True, duration=1)
            time.sleep(1)
            status = move_gripper(clients, ORIENTATION_MAP['straight_up'], blocking=True, duration=1)
            time.sleep(1)
            
        return True

# sdk = bosdyn.client.create_standard_sdk('VoicePromptNav')
# robot = sdk.create_robot('192.168.80.3')
# bosdyn.client.util.authenticate(robot) 

# lease_client = robot.ensure_client(LeaseClient.default_service_name)

# lease_client.take()

# sg = SemanticGrasper(robot)

# sg.search_object_with_gripper("red clippers")

# sg.orient_and_grasp('find_grasp_front', 'red clippers')

# sg.put_down()