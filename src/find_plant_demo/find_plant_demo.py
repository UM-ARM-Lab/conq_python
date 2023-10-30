# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" navigate to a desired location in the map, 
automatically detect a green object near its current location, and then pick up the object.

DEMO USAGE:
First, create a map using graph_nav_command line and save to a folder (in this case ~/Downloads/collabspace2_simple_graph)
get the ID of the waypoints you're interested in by the filenames in the folder
In my case: 
    Waypoint 0: south-cowrie-um.o6E6plYSbeaUK95peBw==
    Waypoint 45: unsaid-minnow-kMr5Fc65S2DYvMAwHol+rg==  (short code: um) (position [2.15, 3.39])
Also tune the HSV filter parameters using weed_detector/hsv_filter_tuning.py
And then update the values in weed_detector/base.py

python3 -m find_plant_demo --upload-filepath ~/Downloads/collabspace2_simple_graph 192.168.80.3
To upload the map
    5 [Enter]
To initialize at waypoint 0: 
    3 south-cowrie-um.o6E6plYSbeaUK95peBw== [Enter]
To navigate to Waypoint 45 and pickup the green object
    f um [Enter]

    
#TODO: Currently the robot will only look for green objects through its left camera, but it should look through all of them.
"""

# from graph_nav_command_line
import argparse
import logging
import math
import os
import sys
import time

import google.protobuf.timestamp_pb2
import graph_nav_util
import grpc

import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.api import geometry_pb2, power_pb2, robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

# from arm_grasp.py
import cv2
import numpy as np
import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

# from arm_simple:
import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

# for weed detection
from weed_detector.base import CameraWeedDetectorBase

g_image_click = None
g_image_display = None

# TODO: Make object-oriented version where there is a "ArmController" object
# and the FindPlantDemoInterface has an instance of that class


##################################################
######## Graph Nav ###############################
##################################################

# class GraphNavInterface(object):
class FindPlantDemoInterface(object):
    """Find Plant demo command line interface. (based on GraphNavInterface)"""

    # def __init__(self, robot,upload_path):
    def __init__(self, robot, options):
        self._options = options
        self._robot = robot

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Filepath for uploading a saved graph's and snapshots too.
        upload_path = options.upload_filepath
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        # clients for getting images and grasping
        self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
        # self._robot_state_client already made
        self._manipulation_api_client = self._robot.ensure_client(ManipulationApiClient.default_service_name)

        self._command_dictionary = {
            '1': self._get_localization_state,
            '2': self._set_initial_localization_fiducial,
            '3': self._set_initial_localization_waypoint,
            '4': self._list_graph_waypoint_and_edge_ids,
            '5': self._upload_graph_and_snapshots,
            '6': self._navigate_to,
            '7': self._navigate_route,
            '8': self._navigate_to_anchor,
            '9': self._clear_graph,
            'e': self._pickup_green_object,
            'f': self._navigate_and_pickup_green_object
        }

    def _navigate_and_pickup_green_object(self, *args):
        """ Navigate to waypoint ID given in args, and then pickup a green object at that location"""
        print("Navigating to where plant is...")
        self._navigate_to(*args, power_off_at_end=False)
        print("Looking for plant...")
        
        # Detect the plant and Pick up using autograsp
        try:
            self._pickup_green_object(method="green")

            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger = bosdyn.client.util.get_logger()
            logger.exception("Threw an exception in _pickup_green_object")
            return False


    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id)

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(self._upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(self._upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print("Uploaded {}".format(waypoint_snapshot.id))
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print("Uploaded {}".format(edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print("Upload complete! The robot is currently not localized to the map; please localize", \
                   "the robot using commands (2) or (3) before attempting a navigation command.")

    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments: [x, y], [x, y, yaw], [x, y, z, yaw],
        # [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from the yaw.
        # When yaw is not specified, an identity quaternion is used.

        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return

        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0, Quat())

        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state()
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return
            seed_T_goal.z = localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(w=float(args[0][3]), x=float(args[0][4]), y=float(args[0][5]),
                                   z=float(args[0][6]))

        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(), 1.0, command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_to(self, *args, power_off_at_end=True):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on and power_off_at_end:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_route(self, *args):
        """Navigate through a specific route of waypoints."""
        if len(args) < 1 or len(args[0]) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            print("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id)
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                print(
                    "List the graph's waypoints and edges to ensure pairs of waypoints has an edge."
                )
                break

        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                print("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0)
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_route_command_id)

            # Power off the robot if appropriate.
            if self._powered_on and not self._started_powered_on:
                # Sit the robot down + power off after the navigation command is complete.
                self.toggle_power(should_power_on=False)

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(RobotCommandBuilder.safe_power_off_command(),
                                                     end_time_secs=time.time())

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print("""
            Options:
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint)."""

                  """
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) Upload the graph and its snapshots (DO THIS FIRST).
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (q) Exit.
            (e) Find nearby green object and pick it up
            (f) Navigate to waypoint, find green object nearby and pick it up (combo of 5,e)
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                self._on_quit()
                break

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)

###########################################
############### from arm_grasp.py ##############
###########################################
    def _pickup_green_object(self, method="click"):
        config = self._options
        robot = self._robot

        """A simple example of using the Boston Dynamics API to command Spot's arm."""
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)
        manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

        # if not powered on, power on
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Tell the robot to stand up. 
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")


        # (from get_image.py) get RGB images from desired sources
        robot.logger.info('Getting an image from: ' + config.image_source)
        print('Getting an image from: ' + config.image_source)
        
        pixel_format = pixel_format_string_to_enum(config.pixel_format)
        image_request = []
        # TODO: Do this dynamically, based on args passed into parser
        # for source in config.image_source:
        #     req = image_pb2.ImageRequest(image_source_name=source,
        #                         quality_percent=75,
        #                         image_format=None,
        #                         pixel_format=pixel_format,
        #                         resize_ratio=None)
        #     print(source)
        #     image_request.append(req)
        # image_request = [
        #     image_pb2.ImageRequest(image_source_name=source,
        #                         quality_percent=75,
        #                         image_format=None,
        #                         pixel_format=pixel_format,
        #                         resize_ratio=None)
        #     for source in config.image_source
        # ]
        image_request = [image_pb2.ImageRequest(image_source_name='left_fisheye_image',
                                quality_percent=75,
                                image_format=None,
                                pixel_format=pixel_format,
                                resize_ratio=None)]
        
        image_responses = image_client.get_image(image_request)

        # the below just gets greyscale
        # image_responses = image_client.get_image_from_sources([config.image_source])

        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                num_bytes = 3
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                num_bytes = 4
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                num_bytes = 1
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                num_bytes = 2
        
        # img = np.fromstring(image.shot.image.data, dtype=dtype)
        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        # print(f"image string shape: {img.shape}, actual height: {image.shot.image.rows} actual width: {image.shot.image.cols}")
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            # img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            try:
                img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
            except:
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        print(f"final image shape: {img.shape}, image type: {type(img)}") #final image shape: (480, 640)
        # get pick_vec (vector to location of object to grasp) using different methods
        if method=="click":
            # Show the image to the user and wait for them to click on a pixel
            robot.logger.info('Click on an object to start grasping...')
            print("Getting object location from user click")
            image_title = 'Click to grasp'
            cv2.namedWindow(image_title)
            cv2.setMouseCallback(image_title, cv_mouse_callback)

            global g_image_click, g_image_display
            g_image_display = img
            cv2.imshow(image_title, g_image_display)
            while g_image_click is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    # Quit
                    print('"q" pressed, exiting.')
                    exit(0)

            robot.logger.info('Picking object at image location (' + str(g_image_click[0]) + ', ' +
                                str(g_image_click[1]) + ')')

            pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
        else:
            robot.logger.info("Getting object location from green detector")
            print("Getting object location from green detector")


            # Find vector to weed
            detector = CameraWeedDetectorBase(camera_name="left_fisheye_image")
            cent, bbox = detector.detect_weed(img)

            # Show window with what the robot sees
            image_title = 'Green Detector'
            cv2.namedWindow(image_title)
            # global g_image_display
            # g_image_display = img

            # draw bounding box
            cv2.rectangle(img, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (255,0,0), 4)
            # cv2.rectangle(g_image_display, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (255,0,0), 4)
            
            cv2.imshow(image_title, img)
            cv2.waitKey(4000) # show image for 4 sec
            cv2.destroyAllWindows()

            pick_vec = geometry_pb2.Vec2(x=cent.x, y=cent.y)


        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # #Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
        # add_grasp_constraint(config, grasp, robot_state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                    manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(0.25)

        robot.logger.info('Finished grasp.')
        time.sleep(2.0)

        #################################
        # Pick up object (from arm_simple)
                # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        x = 0.75
        y = 0
        z = 0.25
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        qw = 0.525322 #1
        qx = 0.8509035 #0
        qy = 0
        qz = 0
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                            ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

        # Send the request
        cmd_id = command_client.robot_command(arm_command)
        robot.logger.info('Moving arm to position 1.')
        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(command_client, cmd_id, 3.0)

        # drop the object 
        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        # Send the request
        cmd_id = command_client.robot_command(gripper_command)
        robot.logger.info('Dropping object (open)')
        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(command_client, cmd_id, 3.0)

        # Make the close gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
        # Send the request
        cmd_id = command_client.robot_command(gripper_command)
        robot.logger.info('Dropping object (close)')
        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(command_client, cmd_id, 3.0)

        # Stow the arm
        # Build the stow command using RobotCommandBuilder
        stow = RobotCommandBuilder.arm_stow_command()
        # Issue the command via the RobotCommandClient
        stow_command_id = command_client.robot_command(stow)
        robot.logger.info("Stow command issued.")
        block_until_arm_arrives(command_client, stow_command_id, 3.0)

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

# from get_image
def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()





###############################################
########## main (combined) ####################
###############################################

def main(argv):
    """Run the command-line interface."""
    # graph_nav_command_line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    bosdyn.client.util.add_base_arguments(parser)

    # from get_image (default needed to get RGB as opposed to greyscale)
    parser.add_argument(
        '--pixel-format', help='Requested pixel format of image. If supplied, will be used for all sources.',
        default='PIXEL_FORMAT_RGB_U8')
    
    # arm_grasp arguments
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='left_fisheye_image')#frontleft_fisheye_image')
    
    parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp',
                        help='Force the robot to use a horizontal grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument(
        '-r', '--force-45-angle-grasp',
        help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)',
        action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp',
                        help='Force the robot to use a squeeze grasp', action='store_true')
    
    options = parser.parse_args(argv)
    
    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1
    if num > 1:
        print("Error: cannot force more than one type of grasp.  Choose only one.")
        sys.exit(1)


    # Setup and authenticate the robot.
    # sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    sdk = bosdyn.client.create_standard_sdk('FindPlantDemoClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    graph_nav_command_line = FindPlantDemoInterface(robot, options) #.upload_filepath)

    # for arm_grasp
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "Robot requires an arm to run this example."
    verify_estop(robot)
    
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    # robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    # image_client = robot.ensure_client(ImageClient.default_service_name)
    # manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # options = config
    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                graph_nav_command_line.run()
                return True
            except Exception as exc:  # pylint: disable=broad-except
                print(exc)
                print("Graph nav command line client threw an error.")
                return False
    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )
        return False


if __name__ == '__main__':
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
