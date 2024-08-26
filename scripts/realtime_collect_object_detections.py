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
from src.conq.navigation.graph_nav.graph_nav_utils import GraphNav

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
from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand

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
from src.conq.semantic_memory import SemanticMemory
from src.conq.perception_lib.custom_yolo import YOLOFarm

load_dotenv('.env.local')

sdk = bosdyn.client.create_standard_sdk('WaypointPhotographerClient')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot) 

lease_client = robot.ensure_client(LeaseClient.default_service_name)

lease_client.take()

with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

    yolo_farm = YOLOFarm(model_path=os.getenv('YOLO_FARM'), device = 'cuda')
    sm = SemanticMemory()
    gn = GraphNav(robot=robot)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = ["hand_color_image"]

    # hand low, looking straight out
    traj_pose1 = (0.50, 0.25, 0.20, 0.707, 0.0, 0.0, 0.707)
    traj_pose2 = (0.60, 0.15, 0.20, 0.924, 0.0, 0.0, 0.383)
    traj_pose3 = (0.65, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0)
    traj_pose4 = (0.60, -0.15, 0.20, 0.924, 0.0, 0.0, -0.383)
    traj_pose5 = (0.50, -0.25,0.20,  0.707, 0.0, 0.0, -0.707)

    # # hand high, looking straight out
    # traj_pose1 = (0.50, 0.25, 0.6, 0.707, 0.0, 0.0, 0.707)
    # traj_pose2 = (0.60, 0.15, 0.6, 0.924, 0.0, 0.0, 0.383)
    # traj_pose3 = (0.65, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0)
    # traj_pose4 = (0.60, -0.15, 0.6, 0.924, 0.0, 0.0, -0.383)
    # traj_pose5 = (0.50, -0.25,0.6,  0.707, 0.0, 0.0, -0.707)

    # # hand high, looking down
    # traj_pose1 = (0.55, 0.0, 0.6, 0.683, 0.183, 0.683, 0.183)
    # traj_pose2 = (0.55, 0.0, 0.6, 0.701, 0.092, 0.701, 0.092)
    # traj_pose3 = (0.55, 0.0, 0.6, 0.707, 0.0, 0.707, 0.0)
    # traj_pose4 = (0.55, 0.0, 0.6, 0.701, -0.092, 0.701, -0.092)
    # traj_pose5 = (0.55, 0.0,0.6,  0.683, -0.183, 0.683, -0.183)

    # # hand high, looking down angled out
    # traj_pose1 = (0.55, 0.10, 0.6, 0.859, 0.113, 0.496, 0.065)
    # traj_pose2 = (0.60, 0.05, 0.6, 0.859, 0.113, 0.496, 0.065)
    # traj_pose3 = (0.65, 0.0, 0.6, 0.819, 0.0, 0.574, 0.0)
    # traj_pose4 = (0.60, -0.05, 0.6, 0.859, -0.113, 0.496, -0.065)
    # traj_pose5 = (0.55, -0.10, 0.6, 0.859, -0.113, 0.496, -0.065)

    t_traj_pose1 = 0.0
    t_traj_pose2 = 3.0
    t_traj_pose3 = 6.0
    t_traj_pose4 = 9.0
    t_traj_pose5 = 12.0

    rot1 = math_helpers.Quat(traj_pose1[3], traj_pose1[4], traj_pose1[5], traj_pose1[6])
    hand_pose1 = math_helpers.SE3Pose(x=traj_pose1[0], y=traj_pose1[1], z=traj_pose1[2], rot=rot1)
    rot2 = math_helpers.Quat(traj_pose2[3], traj_pose2[4], traj_pose2[5], traj_pose2[6])
    hand_pose2 = math_helpers.SE3Pose(x=traj_pose2[0], y=traj_pose2[1], z=traj_pose2[2], rot=rot2)
    rot3 = math_helpers.Quat(traj_pose3[3], traj_pose3[4], traj_pose3[5], traj_pose3[6])
    hand_pose3 = math_helpers.SE3Pose(x=traj_pose3[0], y=traj_pose3[1], z=traj_pose3[2], rot=rot3)
    rot4 = math_helpers.Quat(traj_pose4[3], traj_pose4[4], traj_pose4[5], traj_pose4[6])
    hand_pose4 = math_helpers.SE3Pose(x=traj_pose4[0], y=traj_pose4[1], z=traj_pose4[2], rot=rot4)
    rot5 = math_helpers.Quat(traj_pose5[3], traj_pose5[4], traj_pose5[5], traj_pose5[6])
    hand_pose5 = math_helpers.SE3Pose(x=traj_pose5[0], y=traj_pose5[1], z=traj_pose5[2], rot=rot5)

    traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
        pose=hand_pose1.to_proto(), time_since_reference=seconds_to_duration(t_traj_pose1))
    traj_point2 = trajectory_pb2.SE3TrajectoryPoint(
        pose=hand_pose2.to_proto(), time_since_reference=seconds_to_duration(t_traj_pose2))
    traj_point3 = trajectory_pb2.SE3TrajectoryPoint(
        pose=hand_pose3.to_proto(), time_since_reference=seconds_to_duration(t_traj_pose3))
    traj_point4 = trajectory_pb2.SE3TrajectoryPoint(
        pose=hand_pose4.to_proto(), time_since_reference=seconds_to_duration(t_traj_pose4))
    traj_point5 = trajectory_pb2.SE3TrajectoryPoint(
        pose=hand_pose5.to_proto(), time_since_reference=seconds_to_duration(t_traj_pose5))        

        # Build the trajectory proto by combining the points.
    hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2, traj_point3, traj_point4, traj_point5])    

    for waypoint_num in range(0, len(gn._current_graph.waypoints)):
        # gn.navigate_to(waypoint_number=waypoint_num)

        #arm yolo

        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=hand_traj, root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)

        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command)

        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)

        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # keep the gripper open the whole time.
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            100, build_on_command=robot_command)

        # send the trajectory to the robot.
        cmd_id = command_client.robot_command(robot_command)

        arm_done = False
        while arm_done is False:

            feedback_resp = command_client.robot_command_feedback(cmd_id)

            image_responses = image_client.get_image_from_sources(sources)
            img_rgb = cv2.imdecode(np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8), -1)

            results = yolo_farm.run_inference_raw_image(image=img_rgb)
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = box
                conf = results[0].boxes.conf[i]
                cls = results[0].boxes.cls[i]

                if conf > 0.75:
                    print(f'{yolo_farm.model.names[int(cls)]}: {conf:.2f}')


                label = f'{yolo_farm.model.names[int(cls)]}: {conf:.2f}'
                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_rgb, label, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('hand cam', img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # cv2.imshow('hand cam', img_rgb)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


            if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                print('Traj complete.')
                arm_done = True

        time.sleep(1)



