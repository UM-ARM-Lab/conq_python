from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client.image import build_image_request
import bosdyn.client
import os
import cv2

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

from src.conq.navigation.graph_nav.waypoint_photographer import WaypointPhotographer

import argparse
import sys
import open3d as o3d
import cv2
import time
import numpy as np
import pdb
import torch
from PIL import Image

from google.protobuf import any_pb2, wrappers_pb2

from src.conq.manipulation_lib.Manipulation import grasped_bool, open_gripper, close_gripper, move_gripper, move_gripper
from src.conq.manipulation_lib.Perception3D import VisualPoseAcquirer, PointCloud, Vision

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

from src.conq.manipulation_lib.Manipulation import grasped_bool, open_gripper, close_gripper, move_gripper, move_gripper
from src.conq.manipulation_lib.Perception3D import VisualPoseAcquirer, PointCloud, Vision
from src.conq.perception_lib.fast_owlsam import FastOwlsam
from src.conq.perception_lib.fast_grounded_sam import FastGroundedSAM

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
from src.conq.cameras_utils import get_color_img, get_depth_img, pos_in_cam_to_pos_in_hand, image_to_opencv, RGB_SOURCES, DEPTH_SOURCES
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME

from src.conq.manipulation import grasp_point_in_image
from src.conq.clients import Clients
from src.conq.manipulation_lib.utils import stow_arm
import matplotlib.pyplot as plt
from src.conq.owlsam import OwlSam
from src.conq.perception_lib.grounded_sam_inference import GroundedSAM
from src.conq.grounding_dino import GroundingDino
from src.conq.cameras_utils import image_to_opencv

from src.conq.manipulation import grasp_point_in_image
from src.conq.manipulation_lib.Grasp import get_grasp_candidates, get_best_grasp_pose, compute_grass_z_range, transform_grasp_pose, transform_pose_body_to_hand, get_object_width_at_grasp, grasp_width_to_gripper_open_percent
from src.conq.manipulation_lib.utils import rotate_quaternion
from src.conq.clients import Clients
from src.conq.manipulation_lib.utils import stow_arm
from src.conq.perception_lib.grounded_sam_inference import GroundedSAM
from src.conq.cameras_utils import image_to_opencv

from PIL import Image
import numpy as np
from dotenv import load_dotenv
import time

# import rerun as rr

# from conq.conq_logging.logger import ConqLogger, get_blueprint

from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.lease import LeaseClient

from src.conq.clients import Clients
from src.conq.manipulation_lib.Perception3D import Vision
from src.conq.semantic_memory import SemanticMemory
from src.conq.semantic_grasper import SemanticGrasper
from src.conq.perception_lib.custom_yolo import YOLOFarm
from src.conq.navigation.graph_nav.waypoint_photographer import WaypointPhotographer
from src.conq.navigation.graph_nav.graph_nav_utils import GraphNav


load_dotenv('.env.local')

viewpoints = {
    "l":(0.50, 0.25, 0.0, 0.707, 0.0, 0.0, 0.707),
    "cl":(0.60, 0.15, 0.0, 0.924, 0.0, 0.0, 0.383),
    "c":(0.65, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "cr":(0.60, -0.15, 0.0, 0.924, 0.0, 0.0, -0.383),
    "r":(0.50, -0.25,0.0,  0.707, 0.0, 0.0, -0.707),
}

sdk = bosdyn.client.create_standard_sdk('FullSystemTestClient')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot) 

lease_client = robot.ensure_client(LeaseClient.default_service_name)

lease_client.take()

with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

    images_loc = os.getenv('MEMORY_IMAGE_PATH')
    sg = SemanticGrasper(robot)
    gn = GraphNav(robot)
    gn.navigate_to('waypoint_0')
    sm = SemanticMemory()
    wp = WaypointPhotographer(robot)
    yolofarm = YOLOFarm(os.getenv('YOLO_FARM'), device='cpu')
    target = input("Please enter the target object: ")

    ranking = sm.construct_semantic_search_ranking(target)

    for obj in ranking:
        obj_loc = sm.memory[obj]

        gn.navigate_to(obj_loc[0])
        open_gripper(sg.clients)
        move_gripper(sg.clients, pose=viewpoints[obj_loc[4]], duration=1)
        time.sleep(1)
        wp._take_hand_photo_at_waypoint('s', obj_loc[0])

        sources = ["hand_depth_in_hand_color_frame", "hand_color_image"]
        image_responses = sg.image_client.get_image_from_sources(sources)
        obj_wp = obj_loc[0]
        yolo_centroids = yolofarm.get_object_centroids(image_path=f'{images_loc}hand_color_image_s_{obj_wp}_.jpg', confidence_thresh=0.05)
        
        # if yolo_centroids is None:
        #     continue

        # else:
        #     for image_object in yolo_centroids:

        #         if image_object[0] == obj:
        #             sg.walk_to_pixel(image_responses[1], image_object[1], image_object[2])
        #             time.sleep(1)

        #             #inspect
        #             inspect_poses = [(0.65,0.0, 0.6, 0.683, -0.183, 0.683, -0.183),
        #                              (0.65,0.0, 0.6, 0.574, 0.0, 0.819, 0.0), 
        #                              (0.65,0.0, 0.6, 0.683, 0.183, 0.683, 0.183), 
        #                              (0.65,0.0, 0.6, 0.866, 0.0, 0.500, 0.0),
        #             ]
                    
        #             for p in inspect_poses:
        #                 move_gripper(sg.clients, pose=p, duration=1)
        #                 wp._take_hand_photo_at_waypoint('g', obj_loc[0], depth=False)
        #                 image_responses = sg.image_client.get_image_from_sources(sources)
        #                 time.sleep(1)
        #                 yolo_centroids = yolofarm.get_object_centroids(image_path=f'{images_loc}hand_color_image_g_{obj_wp}_.jpg', confidence_thresh=0.2)

        #                 for graspable_obj in yolo_centroids:
        #                     if graspable_obj[0] == target:
        #                         print(f"Found target {target}.")
        #                         pick_vec = geometry_pb2.Vec2(x=graspable_obj[1], y=graspable_obj[1])
        #                         grasp_point_in_image(sg.clients, image_res=image_responses[1], pick_vec=pick_vec)
        #                         time.sleep(3)
        #                         #hoist above body
        #                         move_gripper(sg.clients, pose=(0.5,0,0.85, 1.0,0,0,0), duration=2)
        #                         break



        stow_arm(robot, sg.command_client)

    gn.navigate_to(waypoint_number='waypoint_0')
    sg.put_down()
    print(f"Retrieved the {target}. All done!")

