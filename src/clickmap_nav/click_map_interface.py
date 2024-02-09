from graph_nav_interface import GraphNavInterface
from conq.data_recorder import ConqDataRecorder
from conq.clients import Clients
from conq.cameras_utils import RGB_SOURCES, ALL_SOURCES

import argparse
import os
import sys
import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.image import ImageClient
import numpy as np
import time
from pathlib import Path
from threading import Timer

from view_map_highlighted import SpotMap, VTKEngine, BosdynVTKInterface, HighlightInteractorStyle

class ClickMapInterface(GraphNavInterface, HighlightInteractorStyle): 
    def __init__(self, robot, upload_path, clients: Clients=None, silhouette=None, silhouetteActor=None):
        GraphNavInterface.__init__(self,robot, upload_path)
        HighlightInteractorStyle.__init__(self, silhouette, silhouetteActor)
        
        self.clients = clients
        self.clients.graphnav = self._graph_nav_client
        self.clients.state = self._robot_state_client
        now = int(time.time())
        root = Path(f"data/click_map_data_{now}")
        self.recorder = ConqDataRecorder(root, self.clients, sources=ALL_SOURCES, map_directory_path=upload_path)
        self.recorder_started = False

        self._list_graph_waypoint_and_edge_ids()
        self._upload_graph_and_snapshots() # option 5
        self.print_controls()

    
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
            (q) Exit.
            """)
        
    def start_recording(self):
        if self.recorder_started:
            self.recorder.next_episode()
        else:
            self.recorder_started = True

        Timer(20, self.start_recording).start()
        self.recorder.start_episode(mode="localization", instruction="no instruction", save_interval=200)



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

    # graph_nav_interface = ClickMapInterface(robot, options.upload_filepath)

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
