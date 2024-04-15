import argparse
import os
import sys
import time
from pathlib import Path
from threading import Timer

import bosdyn.client.channel
import bosdyn.client.util
import numpy as np
from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.recording import GraphNavRecordingServiceClient
from google.protobuf import wrappers_pb2 as wrappers

from clickmap_nav.graph_nav_interface import GraphNavInterface
from clickmap_nav.graph_nav_util import (
    find_unique_waypoint_id,
    find_waypoint_to_timestamp,
    sort_waypoints_chrono,
    update_waypoints_and_edges,
)
from clickmap_nav.view_map_highlighted import (
    BosdynVTKInterface,
    HighlightInteractorStyle,
    SpotMap,
    VTKEngine,
)
from conq.cameras_utils import ALL_SOURCES, RGB_SOURCES
from conq.clients import Clients
from conq.data_recorder import ConqDataRecorder


class ClickMapInterface(GraphNavInterface, HighlightInteractorStyle): 
    def __init__(self, robot, upload_path, clients: Clients=None, silhouette=None, silhouetteActor=None):
        GraphNavInterface.__init__(self,robot, upload_path)
        HighlightInteractorStyle.__init__(self, silhouette=silhouette, silhouetteActor=silhouetteActor)
        
        self.clients = clients
        self.clients.graphnav = self._graph_nav_client
        self.clients.state = self._robot_state_client

        # Instantiate the recording client
        self._recording_client = self._robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)

        # Adding a graphnav recording clients
        self.clients.recording = self._recording_client

        # Create a recording environment to get status if recording started
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(
            waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment())
        
        # Create map processing client
        self._map_processing_client = robot.ensure_client(
            MapProcessingServiceClient.default_service_name)
        
        # Set clients
        self.clients.map_processing = self._map_processing_client

        now = int(time.time())
        root = Path(f"data/click_map_data_{now}")
        self.recorder = ConqDataRecorder(root, self.clients, sources=ALL_SOURCES, map_directory_path=upload_path)
        self.recorder_started = False
        self.initialized_waypoint = None

        self._list_graph_waypoint_and_edge_ids()
        self._upload_graph_and_snapshots() # option 5
        self.print_controls()

        self.waypoint_to_timestamp, _ , self.name_to_id = find_waypoint_to_timestamp(self._current_graph)

        # Commands for map recording/editing
        self._command_dictionary = {
            '0': self._clear_map,
            '1': self._start_recording,
            '2': self._stop_recording,
            '3': self._get_recording_status,
            '4': self._create_default_waypoint,
            '5': self._download_full_graph,
            '6': self._list_graph_waypoint_and_edge_ids,
            '7': self._create_new_edge,
            '8': self._create_loop,
            '9': self._auto_close_loops_prompt,
            'a': self._optimize_anchoring
        }
    
    def onKeyPressEvent(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        actor = self.highlight_keypress_location()
        if key == '1':
            self._get_localization_state()
        elif key == '2':
            if actor:
                print(f"initializing localization to nearest fiducial")
                self._set_initial_localization_fiducial()
                self.print_controls()
        elif key == '4':
            if actor:
                print(f"initializing localization to waypoint {actor.waypoint_id}")
                self.initialized_waypoint = actor.waypoint_id
                self._set_initial_localization_waypoint([actor.waypoint_id])
                self.print_controls()
        elif key == '5':
            print(f"(Re)uploading graph and snapshots")
            self._upload_graph_and_snapshots()
            self.print_controls()
        elif key == '6':
            if actor:
                print(f"navigating to: {actor.waypoint_id}")
                self._navigate_to([actor.waypoint_id])
        elif key == '8':
            self._list_graph_waypoint_and_edge_ids()
            self.print_controls()
        elif key == '9':
            print(f"clearing graph")
            self._clear_graph()
            self.print_controls()
        elif key == 'q':
            # TODO: figure out how to do same as 'e' for exit
            self.recorder.stop() # Stop recording on exit
            self._on_quit()
        elif key == "n":
            # Navigate in loop
            self.navigate_in_loop()
        elif key == "s":
            # Robot returns to seed/origin point
            self.return_to_seed()


        #  Forward events
        self.OnKeyPress()
        return key, actor

    def print_controls(self):
        print("""
            Controls:
              (Right-Click)  Zoom
              (Left-Click)   Rotate
              (Scroll-Click) Pan
              (r) reset the camera
              (e) exit the program
              (f) set a new camera focal point and fly towards that point
              (u) invokes the user event
              (3) toggles between stereo and non-stero mode
              (l) toggles on/off a latitude/longitude markers that can be used to estimate/control position.
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (4) Initialize localization to a specific waypoint (must be exactly at the waypoint).
            (5) (Re)Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (8) List the waypoint ids and edge ids of the map on the robot.
            (9) Clear the current graph.
            (n) Navigate in loop of current waypoints
            (s) Return to origin/seed pose that robot was initally localized to
            (q) Exit.
            """)
        
    def start_recording(self):
        if self.recorder_started:
            self.recorder.next_episode()
        else:
            self.recorder_started = True

        Timer(20, self.start_recording).start()
        self.recorder.start_episode(mode="localization", instruction="no instruction", save_interval=200)

    def navigate_in_loop(self):
        self._navigate_route([waypoint[0] for waypoint in self.waypoint_to_timestamp])

    def return_to_seed(self):
        self._navigate_to([self.initialized_waypoint])

    """These functions are taken from the spot sdk"""
    def should_we_start_recording(self):
        # Before starting to record, check the state of the GraphNav system.
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            # Check that the graph has waypoints. If it does, then we need to be localized to the graph
            # before starting to record
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    # Not localized to anything in the map. The best option is to clear the graph or
                    # attempt to localize to the current map.
                    # Returning false since the GraphNav system is not in the state it should be to
                    # begin recording.
                    return False
        # If there is no graph or there exists a graph that we are localized to, then it is fine to
        # start recording, so we return True.
        return True
    
    def _start_recording(self, *args):
        """Start recording a map."""
        should_start_recording = self.should_we_start_recording()
        if not should_start_recording:
            print('The system is not in the proper state to start recording.'
                  'Try using the graph_nav_command_line to either clear the map or'
                  'attempt to localize to the map.')
            return
        try:
            status = self._recording_client.start_recording(
                recording_environment=self._recording_environment)
            print('Successfully started recording a map.')
        except Exception as err:
            print(f'Start recording failed: {err}')

    def _stop_recording(self, *args):
        """Stop or pause recording a map."""
        first_iter = True
        while True:
            try:
                status = self._recording_client.stop_recording()
                print('Successfully stopped recording a map.')
                break
            except bosdyn.client.recording.NotReadyYetError as err:
                # It is possible that we are not finished recording yet due to
                # background processing. Try again every 1 second.
                if first_iter:
                    print('Cleaning up recording...')
                first_iter = False
                time.sleep(1.0)
                continue
            except Exception as err:
                print(f'Stop recording failed: {err}')
                break
    
    def _create_default_waypoint(self, *args):
        """Create a default waypoint at the robot's current location."""
        resp = self._recording_client.create_waypoint(waypoint_name='default')
        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            print('Successfully created a waypoint.')
        else:
            print('Could not create a waypoint.')

    def _download_full_graph(self, *args):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Failed to download the graph.')
            return
        self._write_full_graph(graph)
        print(
            f'Graph downloaded with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges')
        # Download the waypoint and edge snapshots.
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)

    def _write_full_graph(self, graph):
        """Download the graph from robot to the specified, local filepath location."""
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, 'graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        """Download the waypoint snapshots from robot to the specified, local filepath location."""
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id)
            except Exception:
                # Failure in downloading waypoint snapshot. Continue to next snapshot.
                print(f'Failed to download waypoint snapshot: {waypoint.snapshot_id}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'waypoint_snapshots'),
                              str(waypoint.snapshot_id), waypoint_snapshot.SerializeToString())
            num_waypoint_snapshots_downloaded += 1
            print(
                f'Downloaded {num_waypoint_snapshots_downloaded} of the total {len(waypoints)} waypoint snapshots.'
            )

    def _download_and_write_edge_snapshots(self, edges):
        """Download the edge snapshots from robot to the specified, local filepath location."""
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                # Failure in downloading edge snapshot. Continue to next snapshot.
                print(f'Failed to download edge snapshot: {edge.snapshot_id}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'edge_snapshots'),
                              str(edge.snapshot_id), edge_snapshot.SerializeToString())
            num_edge_snapshots_downloaded += 1
            print(
                f'Downloaded {num_edge_snapshots_downloaded} of the total {num_to_download} edge snapshots.'
            )

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), 'wb+') as f:
            f.write(data)
            f.close()

    def _update_graph_waypoint_and_edge_ids(self, do_print=False):
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = update_waypoints_and_edges(
            graph, localization_id, do_print)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""
        self._update_graph_waypoint_and_edge_ids(do_print=True)

    def _create_new_edge(self, *args):
        """Create new edge between existing waypoints in map."""

        if len(args[0]) != 2:
            print('ERROR: Specify the two waypoints to connect (short code or annotation).')
            return

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        from_id = find_unique_waypoint_id(args[0][0], self._current_graph,
                                                         self._current_annotation_name_to_wp_id)
        to_id = find_unique_waypoint_id(args[0][1], self._current_graph,
                                                       self._current_annotation_name_to_wp_id)

        print(f'Creating edge from {from_id} to {to_id}.')

        from_wp = self._get_waypoint(from_id)
        if from_wp is None:
            return

        to_wp = self._get_waypoint(to_id)
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_id
        new_edge.id.to_waypoint = to_id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print(f'edge transform = {new_edge.from_tform_to}')

        # Send request to add edge to map
        self._recording_client.create_edge(edge=new_edge)

    def _create_loop(self, *args):
        """Create edge from last waypoint to first waypoint."""

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        if len(self._current_graph.waypoints) < 2:
            self._add_message(
                f'Graph contains {len(self._current_graph.waypoints)} waypoints -- at least two are '
                f'needed to create loop.')
            return False

        sorted_waypoints = sort_waypoints_chrono(self._current_graph)
        edge_waypoints = [sorted_waypoints[-1][0], sorted_waypoints[0][0]]

        self._create_new_edge(edge_waypoints)

    def _auto_close_loops_prompt(self, *args):
        print("""
        Options:
        (0) Close all loops.
        (1) Close only fiducial-based loops.
        (2) Close only odometry-based loops.
        (q) Back.
        """)
        try:
            inputs = input('>')
        except NameError:
            return
        req_type = str.split(inputs)[0]
        close_fiducial_loops = False
        close_odometry_loops = False
        if req_type == '0':
            close_fiducial_loops = True
            close_odometry_loops = True
        elif req_type == '1':
            close_fiducial_loops = True
        elif req_type == '2':
            close_odometry_loops = True
        elif req_type == 'q':
            return
        else:
            print('Unrecognized command. Going back.')
            return
        self._auto_close_loops(close_fiducial_loops, close_odometry_loops)

    def _auto_close_loops(self, close_fiducial_loops, close_odometry_loops, *args):
        """Automatically find and close all loops in the graph."""
        response = self._map_processing_client.process_topology(
            params=map_processing_pb2.ProcessTopologyRequest.Params(
                do_fiducial_loop_closure=wrappers.BoolValue(value=close_fiducial_loops),
                do_odometry_loop_closure=wrappers.BoolValue(value=close_odometry_loops)),
            modify_map_on_server=True)
        print(f'Created {len(response.new_subgraph.edges)} new edge(s).')

    def _optimize_anchoring(self, *args):
        """Call anchoring optimization on the server, producing a globally optimal reference frame for waypoints to be expressed in."""
        response = self._map_processing_client.process_anchoring(
            params=map_processing_pb2.ProcessAnchoringRequest.Params(),
            modify_anchoring_on_server=True, stream_intermediate_results=False,
            apply_gps_results=self.use_gps)
        if response.status == map_processing_pb2.ProcessAnchoringResponse.STATUS_OK:
            print(f'Optimized anchoring after {response.iteration} iteration(s).')
            # If we are using GPS, the GPS coordinates in the graph have been changed, so we need
            # to re-download the graph.
            if self.use_gps:
                print(f'Downloading updated graph...')
                self._current_graph = self._graph_nav_client.download_graph()
        else:
            print(f'Error optimizing {response}')

    def _get_waypoint(self, id):
        """Get waypoint from graph (return None if waypoint not found)"""

        if self._current_graph is None:
            self._current_graph = self._graph_nav_client.download_graph()

        for waypoint in self._current_graph.waypoints:
            if waypoint.id == id:
                return waypoint

        print(f'ERROR: Waypoint {id} not found in graph.')
        return None

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""

        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x, from_se3.position.y, from_se3.position.z,
            Quat(w=from_se3.rotation.w, x=from_se3.rotation.x, y=from_se3.rotation.y,
                 z=from_se3.rotation.z))

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x, to_se3.position.y, to_se3.position.z,
            Quat(w=to_se3.rotation.w, x=to_se3.rotation.x, y=to_se3.rotation.y,
                 z=to_se3.rotation.z))

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print("""
            Options:
            (0) Clear map.
            (1) Start recording a map.
            (2) Stop recording a map.
            (3) Get the recording service's status.
            (4) Create a default waypoint in the current robot's location.
            (5) Download the map after recording.
            (6) List the waypoint ids and edge ids of the map on the robot.
            (7) Create new edge between existing waypoints using odometry.
            (8) Create new edge from last waypoint to first waypoint using odometry.
            (9) Automatically find and close loops.
            (a) Optimize the map's anchoring.
            (q) Exit.
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                break

            if req_type not in self._command_dictionary:
                print('Request not in the known command dictionary.')
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)

    # Making another function that is a routine for editing the map. We will really only decide
        # to edit once we know there is some waypoint we want to add.
    def edit_map(self):
        # First start the recorder
        self._start_recording()

        # Waypoints are added by just walking around. Make sure to create a copy of the
        # graph before navigating.

        # TODO: add some vision routine to detect certain objects.

        # Stop recording and reupload the new graph to the robot

        # Make sure to localize the robot
        
        pass



def main(argv):
    """Run the click_map graph_nav interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    parser.add_argument('-a', '--anchoring', action='store_true',
                        help='Draw the map according to the anchoring (in seed frame).')
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    spot_map = SpotMap(options.upload_filepath)
    vtk_engine = VTKEngine()

        # Create an interface to create actors from the map datastructure
    bosdyn_vtk_interface = BosdynVTKInterface(spot_map, vtk_engine.renderer)
    # Display map objects extracted from file
    if options.anchoring:
        if len(spot_map.graph.anchoring.anchors) == 0:
            print('No anchors to draw.')
            sys.exit(-1)
        avg_pos = bosdyn_vtk_interface.create_anchored_graph_objects()
    else:
        avg_pos = bosdyn_vtk_interface.create_graph_objects()
    vtk_engine.set_camera(avg_pos + np.array([-1.0, 0.0, 5.0]))

    silhouette, silhouetteActor = bosdyn_vtk_interface.make_silhouette_actor()
    # TODO: can all this lease client boilerplate be handled by a ArmlabRobot class? loop below seems it could be reused
    # How do we want to manage all of the clients and ensuring the clients?
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    clients = Clients(lease=lease_client, state=None, manipulation=None,
                          image=image_client, graphnav=None, raycast=None, command=None, robot=robot)
    style = ClickMapInterface(robot, options.upload_filepath, clients, silhouette, silhouetteActor)
    vtk_engine.set_interactor_style(style)

    style.start_recording()

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                vtk_engine.start()
                style.recorder.stop()
                return True
            except Exception as exc:  # pylint: disable=broad-except
                print(exc)
                print('Graph nav command line client threw an error.')
                return False
    except ResourceAlreadyClaimedError:
        print(
            'The robot\'s lease is currently in use. Check for a tablet connection or try again in a few seconds.'
        )
        return False


if __name__ == '__main__':
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
