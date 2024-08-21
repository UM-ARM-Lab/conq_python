import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
import time

from src.conq.navigation.graph_nav.waypoint_photographer import WaypointPhotographer
from src.conq.perception_lib.custom_yolo import YOLOFarm
from src.conq.navigation.graph_nav.graph_nav_utils import GraphNav
from src.conq.manipulation_lib.Manipulation import open_gripper, close_gripper, move_gripper
from src.conq.semantic_memory import SemanticMemory
from src.conq.semantic_grasper import SemanticGrasper

from dotenv import load_dotenv
import os

load_dotenv('.env.local')

# Setup and authenticate the robot.
sdk = bosdyn.client.create_standard_sdk('WaypointPhotographerClient')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot) 

lease_client = robot.ensure_client(LeaseClient.default_service_name)

lease_client.take()

with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
    wp = WaypointPhotographer(robot)

    # open_gripper(wp.clients)
    # move_gripper(wp.clients, (0.85, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0), duration=1)
    # wp._take_hand_photo_at_waypoint('c', 'waypoint_0')
    # time.sleep(1)

    sm = SemanticMemory()
    # sm.dream()
    ranking = sm.construct_semantic_search_ranking('water hose nozzle')
    time.sleep(10)
    print(ranking)

    obj_loc = sm.memory[ranking[0]]

    pose = (obj_loc[1], obj_loc[2], 0.4, 0.707, 0.707, 0.025, -0.025)
    open_gripper(wp.clients)
    move_gripper(wp.clients, pose, duration=5, with_body=True)
    time.sleep(5)
    