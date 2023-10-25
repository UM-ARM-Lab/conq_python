from graph_nav_interface import GraphNavInterface
import argparse
import os
import sys
import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError

from clickmap_nav import SpotMap, VTKEngine

class ClickMapGraphNavInterface(GraphNavInterface):
    def __init__(self, robot, upload_path):
        super().__init__(robot, upload_path)
        self.spot_map = SpotMap(upload_path)
        self.vtk_engine = VTKEngine(self.spot_map)
        self.interactor_style = self.vtk_engine.GetInteractorStyle()
        self.interactor_style.add_external_observer('space', self._navigate_to()) 
        self._upload_graph_and_snapshots() # option 5

    def run(self):
        """Main loop for the click-map interface."""
        # Controls determined in SpotCommandInteractorStyle
        print("""
            Controls:
              (Right-Click)  Zoom
              (Left-Click)   Rotate
              (Scroll-Click) Pan
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint).
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) (Re)Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (q) Exit.
            """)
        self.vtk_engine.start()
        # vtk engine is callback-based, but I need it to return the waypoint id
        # and the key that was pressed every time a key is pressed, then use i
        # that to trigger the corresponding function in the GraphNavInterface class



def main(argv):
    """Run the click_map graph_nav interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    graph_nav_interface = ClickMapGraphNavInterface(robot, options.upload_filepath)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                graph_nav_interface.run()
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
