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
    'hand_search':                       (0.55,0.0,0.55, 0.843,0.0,0.537,0.0),
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

        self.waypoint_photographer = WaypointPhotographer(self.robot)
        self.waypoint_photographer._img_sources = ['right_fisheye_image', 'left_fisheye_image', 'back_fisheye_image', 'frontleft_fisheye_image', 'frontright_fisheye_image']


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
        

    def take_photos(self):
        self.waypoint_photographer._take_photos_at_waypoint(waypoint_str='grasp_waypoint')

    def find_object_in_photos(self, object):
        image_paths = os.listdir(self.images_loc)
        images = [self._encode_image(self.images_loc + path) for path in image_paths]

        for idx, image in enumerate(images):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.MY_API_KEY}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": 
                     """
                     You are an expert tool/object identifier robot. User will give you an image and ask if an tool/object is present in the image. You will ONLY respond 'yes' or 'no' all lowercase, and you must respond one or the other.
                     Example:
                     If you see an image of a table with a drill, potting soil and rake, with the prompt 'Do you see shovel?', you will output: no
                     If you see an image of a table with a remote, teddy bear and pruners, with the prompt 'Do you see pruners?', you will output: yes
                     """},
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": f"Do you see {object}?"
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            os.remove(self.images_loc + image_paths[idx])

            found_obj = True if response.json()['choices'][0]['message']['content'] == 'yes' else False

            if found_obj:
                found_obj_img_path = image_paths[idx]
                found_camera_frame = found_obj_img_path.split('_')[0:3] #
                print(f'Frame of found obj: {found_camera_frame}')
                return f'{found_camera_frame[0]}_{found_camera_frame[1]}_{found_camera_frame[2]}'
            
        
        #fail to find object
        return 'back_fisheye_image'

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

    def capture_image(self):
        sources = self.image_client.list_image_sources()
        source_list = []
        for source in sources:
            if source.image_type == ImageSource.IMAGE_TYPE_VISUAL:
                # only append if sensor has corresponding depth sensor
                if source.name in VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE:
                    source_list.append(source.name)
                    #source_list.append(VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE[source.name])
        image_list = []
        for i in range(len(source_list)):
            source_name = source_list[i]
            img_req = None
            if 'depth' not in source_name:
                img_req = build_image_request(source_name, quality_percent=100,
                                        image_format=image_pb2.Image.FORMAT_RAW,
                                        pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8)

            image_response = self.image_client.get_image([img_req])
            image_list.append(image_response[0])
        
        return image_list
    def save_image_to_local(self, image, source_name):
        directory = os.getenv('MEMORY_IMAGE_PATH')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{source_name}.jpg")
        cv2.imwrite(file_path, image)
        print(f"Image saved to {file_path}")
        return file_path
    
    # def set_mobility_params(self):
    #     """Set robot mobility params to disable obstacle avoidance."""
    #     obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
    #                                                 disable_vision_foot_obstacle_avoidance=True,
    #                                                 disable_vision_foot_constraint_avoidance=True,
    #                                                 obstacle_avoidance_padding=.001)
    #     body_control = self.set_default_body_control()
    #     if self._limit_speed:
    #         speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
    #             linear=Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
    #         if not self._avoid_obstacles:
    #             mobility_params = spot_command_pb2.MobilityParams(
    #                 obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control,
    #                 locomotion_hint=spot_command_pb2.HINT_AUTO)
    #         else:
    #             mobility_params = spot_command_pb2.MobilityParams(
    #                 vel_limit=speed_limit, body_control=body_control,
    #                 locomotion_hint=spot_command_pb2.HINT_AUTO)
    #     elif not self._avoid_obstacles:
    #         mobility_params = spot_command_pb2.MobilityParams(
    #             obstacle_params=obstacles, body_control=body_control,
    #             locomotion_hint=spot_command_pb2.HINT_AUTO)
    #     else:
    #         #When set to none, RobotCommandBuilder populates with good default values
    #         mobility_params = None
    #     return mobility_params

    def go_to_object(self,text):
        image_list = self.capture_image()
        detector = GroundingDino()
        best_score = 0
        best_centroid = None
        best_visual_response  = None
        for i in range(len(image_list)):
            visual_response = image_list[i]
            cv_image = image_to_opencv(visual_response)
            image_path = self.save_image_to_local(cv_image, source_name)
            boxes,scores = detector.predict_box_score_with_text(image_path, text)
            if boxes.any():
                for box,score in zip(boxes,scores):
                    if score>best_score:
                        best_score = score
                        best_centroid = detector.compute_box_centroid(box)
                        best_visual_response = visual_response
            else:
                continue
        
        if best_score!=0:
            walk_x,walk_y = best_centroid
            walk_vec = geometry_pb2.Vec2(x= walk_x, y=walk_y)
            offset_distance = wrappers_pb2.FloatValue(value=0.5)
            # Build the proto
            walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec, transforms_snapshot_for_camera=best_visual_response.shot.transforms_snapshot,
            frame_name_image_sensor=best_visual_response.shot.frame_name_image_sensor,
            camera_model=best_visual_response.source.pinhole, offset_distance=offset_distance)

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

        else:
            print("No object in sight!!!")

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
            time.sleep(1)

        return True


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
        flat_body_T_pose = [
            math_helpers.SE3Pose(x=0.0, y=0.0, z=0.0, rot=self.create_rotation_quat(0)),
            math_helpers.SE3Pose(x=0.0, y=0.0, z=0.0, rot=self.create_rotation_quat(60)),
            math_helpers.SE3Pose(x=0.0, y=0.0, z=0.0, rot=self.create_rotation_quat(120)),
            math_helpers.SE3Pose(x=0.0, y=0.0, z=0.0, rot=self.create_rotation_quat(180)),
            math_helpers.SE3Pose(x=0.0, y=0.0, z=0.0, rot=self.create_rotation_quat(240)),
            math_helpers.SE3Pose(x=0.0, y=0.0, z=0.0, rot=self.create_rotation_quat(300))
        ]

        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)


        # detector = GroundingDino()
        best_score = 0
        best_box = None

        for i in range(6):

            source_name = "hand_color_image"
            gaze_pose = ORIENTATION_MAP['hand_search']
            status = move_gripper(self.clients, gaze_pose, blocking=True, duration=0.5)
            time.sleep(0.5) 
            rgb_request = build_image_request(source_name, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
            rgb_response= self.image_client.get_image([rgb_request])[0]
            rgb_np = image_to_opencv(rgb_response, auto_rotate=True)
            image = np.array(rgb_np,dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_path = self.save_image_to_local(image, f'hand_search_{i}')
            
            # boxes,scores = detector.predict_box_score_with_text(image_path, text)
            # if boxes.any():
            #     for box,score in zip(boxes,scores):
            #         if score>best_score:
            #             best_score = score
            #             best_box = box

            #ROTATE BODY
            
            self.relative_move(0, 0,  math.radians(60), ODOM_FRAME_NAME)
            


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


                # ------------------------------------- USING PIXEL GRASP -----------------------------------------------------------------------------------
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

sdk = bosdyn.client.create_standard_sdk('VoicePromptNav')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot) 

lease_client = robot.ensure_client(LeaseClient.default_service_name)

lease_client.take()

sg = SemanticGrasper(robot)

sg.search_object_with_gripper("hose nozzle")

# sg.orient_and_grasp(object_direction_name='frontleft_fisheye_image', object_name='hose nozzle')
# sg.put_down()