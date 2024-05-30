import numpy as np
import requests
import base64
from openai import OpenAI
import json
import os
import base64

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
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, depth_image_to_pointcloud, _depth_image_data_to_numpy
from conq.cameras_utils import get_color_img, get_depth_img, pos_in_cam_to_pos_in_hand, image_to_opencv, RGB_SOURCES, DEPTH_SOURCES
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME

from conq.manipulation import grasp_point_in_image
from conq.clients import Clients

import matplotlib.pyplot as plt
from owlsam import OwlSam

ORIENTATION_MAP = {
    'back_fisheye_image':               (-1.00,0.0,0.1, 0.7071,0.,0.7071,0),
    'frontleft_fisheye_image':          (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'frontright_fisheye_image':         (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'frontleft_depth_in_visual_frame':  (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'frontright_depth_in_visual_frame': (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'hand_depth_in_hand_color_frame':   (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'hand_depth':                       (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'hand_color_image':                 (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'left_fisheye_image':               (0,0.75,0.1, 0.7071,0.,0.7071,0),
    'right_fisheye_image':              (0,-0.75,0.1, 0.7071,0.,0.7071,0)
}

class SemanticGrapser:

    def __init__(self, robot):

        load_dotenv('/Users/adibalaji/Desktop/agrobots/conq_python/src/.env.local')

        self.robot = robot
        self.MY_API_KEY = os.getenv('GPT_KEY')
        self.ORG_KEY = os.getenv('ORG_KEY')
        self.client = OpenAI(organization=self.ORG_KEY, api_key=self.MY_API_KEY)

        self.waypoint_photographer = WaypointPhotographer(self.robot)

        self.images_loc = os.getenv('MEMORY_IMAGE_PATH')

        self.lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self.manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.rc_client = robot.ensure_client(RayCastClient.default_service_name)
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    def take_photos(self):
        self.waypoint_photographer._take_photos_at_waypoint(waypoint_str='grasp_waypoint')

    def find_object_in_photos(self, object, img_loc):
        image_paths = os.listdir(img_loc)
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

            found_obj = True if response.json()['choices'][0]['message']['content'] == 'yes' else False

            if found_obj:
                found_obj_img_path = image_paths[idx]
                found_camera_frame = found_obj_img_path.split('_')[0:3]
                return found_camera_frame
            
        
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

    #orient gripper along with object direction
    def orient_and_grasp(self,name):
        assert self.robot.has_arm(), 'Robot requires an arm to run this example.'
        self.verify_estop()
        self.lease_client.take()
        owlsam = OwlSam()

        sources = ["hand_color_image"]
        vision = Vision(self.image_client, sources)

        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=False):
            assert self.robot.is_powered_on(), 'Robot power on failed.'
            
            clients = Clients(lease=self.lease_client, state=self.robot_state_client, manipulation=self.manipulation_api_client,
                image=self.image_client, raycast=self.rc_client, command=self.command_client, robot=self.robot, graphnav=None)
            
            # Deploy the arm
            robot_cmd = RobotCommandBuilder.arm_ready_command()
            cmd_id = self.command_client.robot_command(robot_cmd)
            block_until_arm_arrives(self.command_client, cmd_id)

            #look at scene
            gaze_pose = ORIENTATION_MAP[name]
            status = move_gripper(clients, gaze_pose, blocking=False, duration = 0.5)
            status = open_gripper(clients)
            time.sleep(1)

            image_responses = self.image_client.get_image_from_sources(sources)

            rgb = vision.get_latest_RGB()

            boxes, centroid = owlsam.predict_boxes(rgb, [["tool"]])
            print(f'predicted boxes shape: {boxes.shape}')
            pix_x, pix_y = (centroid[0],centroid[1]) # Get from object detector
                
            pick_vec = geometry_pb2.Vec2(x=pix_x, y=pix_y)

            grasp_point_in_image(clients,image_responses[0],pick_vec)

            status = move_gripper(clients, ORIENTATION_MAP['frontleft_fisheye_image'], blocking=False, duration=0.5)
            time.sleep(1)


            

           