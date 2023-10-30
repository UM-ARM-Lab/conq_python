from graph_nav_interface import GraphNavInterface
import argparse
import os
import sys
import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
import numpy as np

from view_map_with_highlight import SpotMap, VTKEngine, BosdynVTKInterface, HighlightInteractorStyle

class ClickMapInterface(GraphNavInterface, HighlightInteractorStyle): 
    def __init__(self, robot, upload_path, silhouette=None, silhouetteActor=None):
        GraphNavInterface.__init__(self,robot, upload_path)
        HighlightInteractorStyle.__init__(self, silhouette, silhouetteActor)

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
            self._on_quit()


        #  Forward events
        self.OnKeyPress()
        return

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
    style = ClickMapInterface(robot, options.upload_filepath, silhouette, silhouetteActor)
    vtk_engine.set_interactor_style(style)

    # graph_nav_interface = ClickMapInterface(robot, options.upload_filepath)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)


    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                vtk_engine.start()
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
