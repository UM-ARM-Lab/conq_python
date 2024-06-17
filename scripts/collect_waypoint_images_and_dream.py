# export PYTHONPATH="${PYTHONPATH}:/Users/adibalaji/Desktop/agrobots/conq_python/src"

import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
import time

from conq.navigation.graph_nav.waypoint_photographer import WaypointPhotographer
from conq.navigation.graph_nav.scene_labeler import SceneLabeler

# Setup and authenticate the robot.
sdk = bosdyn.client.create_standard_sdk('WaypointPhotographerClient')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot) 

lease_client = robot.ensure_client(LeaseClient.default_service_name)

lease_client.take()

with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
    wp = WaypointPhotographer(robot)
    wp.take_photos_of_full_map()

print(f'Photos recorded.')

print(f'Begin dreaming..')

sl = SceneLabeler()
object_dict = sl.extract_objects()
sl.save_dict_to_json(object_dict)
