import argparse
import os
import sys

import bosdyn.client.util
import numpy as np
from arm_segmentation.predictor import Predictor
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from click_map_interface import ClickMapInterface
from view_map_highlighted import BosdynVTKInterface, SpotMap, VTKEngine


# TODO: Make class that will loop around in a circle of waypoints
class ToolDetectorInterface(ClickMapInterface):
    def __init__(
        self, robot, upload_path, model_path=None, silhouette=None, silhouetteActor=None
    ):
        super().__init__(robot, upload_path, silhouette, silhouetteActor)
        # self.predictor = None
        # if model_path is not None:
        #     self.predictor = Predictor(model_path)

    def onKeyPressEvent(self, obj, event):
        key, actor = super().onKeyPressEvent(obj, event)

        if key == "n":
            # Navigate in loop
            self.navigate_in_loop()
        elif key == "r":
            # Robot returns to seed/origin point
            self.return_to_seed()

    def navigate_in_loop(self):
        self._navigate_route(self.graph.waypoints)

    def return_to_seed(self):
        self._navigate_to([self.graph.waypoints[0]])

    def print_controls(self):
        print(
            """
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
            (n) Navigate in loop of current waypoints
            (s) Stop the robot at current pose
            (r) Return to origin/seed pose
        """
        )


def main(argv):
    """Run tool detector interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-u",
        "--upload-filepath",
        help="Full filepath to graph and snapshots to be uploaded.",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--anchoring",
        action="store_true",
        help="Draw the map according to the anchoring (in seed frame).",
    )
    parser.add_argument(
        "--model-filepath",
        type=str,
        required=False,
        help="Full Filepath to model file",
    )
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Setup and authenticate robot
    sdk = bosdyn.client.create_standard_sdk("GraphNavClient")
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    spot_map = SpotMap(options.upload_filepath)
    vtk_engine = VTKEngine()

    # Create an interface to create actors from the map datastructure
    bosdyn_vtk_interface = BosdynVTKInterface(spot_map, vtk_engine.renderer)
    # Display map objects extracted from file
    if options.anchoring:
        if len(spot_map.graph.anchoring.anchors) == 0:
            print("No anchors to draw.")
            sys.exit(-1)
        avg_pos = bosdyn_vtk_interface.create_anchored_graph_objects()
    else:
        avg_pos = bosdyn_vtk_interface.create_graph_objects()
    vtk_engine.set_camera(avg_pos + np.array([-1.0, 0.0, 5.0]))

    silhouette, silhouetteActor = bosdyn_vtk_interface.make_silhouette_actor()
    style = ToolDetectorInterface(
        robot,
        options.upload_filepath,
        options.model_filepath,
        silhouette,
        silhouetteActor,
    )
    vtk_engine.set_interactor_style(style)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                vtk_engine.start()
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


if __name__ == "__main__":
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
