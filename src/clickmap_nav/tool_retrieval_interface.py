from click_map_interface import ClickMapInterface
from arm_segmentation.predictor import Predictor
import argparse
import os
import sys
import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
import numpy as np

from view_map_highlighted import SpotMap, VTKEngine, BosdynVTKInterface
from conq.cameras_utils import get_color_img #, get_depth_img
from conq.manipulation import grasp_point_in_image_basic

# TODO: Implement a state machine to deal with edge cases in a structured way

class ToolRetrievalInferface(ClickMapInterface):
    def __init__(self, robot, upload_path, model_path, silhouette=None, silhouetteActor=None):
        super().__init__(robot, upload_path, silhouette, silhouetteActor)
        self.predictor = Predictor(model_path)
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)

        # Cameras through which to look for objects
        self.cameras = [
            'back_fisheye_image',
            'frontleft_fisheye_image',
            'frontright_fisheye_image',
            'left_fisheye_image',
            'right_fisheye_image',
        ] # See conq/cameras_utils.py RGB_SOURCES


    def onKeyPressEvent(self, obj, event):
        key, actor =  super().onKeyPressEvent(obj, event)
        if key == 'g':
            object_class = 'hose_handle'
            # Go to a waypoint and pick up the tool, then come back
            if actor:
                localization_state = self._get_localization_state()
                initial_waypoint_id = localization_state.localization.waypoint_id
                print(f"navigating to: {actor.waypoint_id}")
                self._navigate_to([actor.waypoint_id])
                print(f"Looking for {object_class}...")
                # pixel_xy, rgb_response = self.find_object(object_class) # may have to reorient the robot / run multiple times
                # self.pick_up_object(rgb_response, pixel_xy)
                print(f"navigating to {initial_waypoint_id}")
                self._navigate_to([initial_waypoint_id])
            else:
                print("No waypoint selected")

    def find_object(self, object_class: str):
        """
        Looks through each of the cameras for an object of type object_class
        Input: 
            object_class: string corresponding to the class of object to look for
        Output:
            the pixel centroid of the object of type object_class with the highest confidence
            the ImageResponse object corresponding to the camera that saw the object (so the robot can deduce the pose of the object)
        """
        pixels_and_confidences = []
        # get images from each camera
        for camera in self.cameras:
            rgb_np, rgb_response = get_color_img(self.image_client, camera)
            # depth_np, depth_res = get_depth_img(self.image_client, 'hand_depth_in_hand_color_frame')
            predictions = self.predictor.predict(rgb_np)
            # TODO: Get centroid
            pixel_xy = predictions[object_class]
            confidence = None
            pixels_and_confidences.append((pixel_xy, confidence, rgb_response))
            
        # find the pixel with the highest confidence
        max_confidence = 0
        best_pixel_xy = None
        best_rgb_response = None
        for pixel_xy, confidence, rgb_response in pixels_and_confidences:
            if confidence > max_confidence:
                max_confidence = confidence
                best_pixel_xy = pixel_xy
                best_rgb_response = rgb_response

        print(f"Found {object_class} at {pixel_xy} in {best_rgb_response}")

        return best_pixel_xy, best_rgb_response


    def pick_up_object(self, image_response, pixel_xy):
        """ Pick up the object at the given location."""
        # See the find_plant_demo for an (unorganized) example of how to do this
        # or continuous_hose_regrasping_demo for a more structured example
        grasp_success = False
        for _ in range(3):
            # first just try the auto-grasp
            grasp_success = grasp_point_in_image_basic(self.clients, image_response, pixel_xy)
            if grasp_success:
                return grasp_success
        return grasp_success


def main(argv):
    """Run the click_map graph_nav interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    parser.add_argument('-a', '--anchoring', action='store_true',
                        help='Draw the map according to the anchoring (in seed frame).')
    parser.add_argument('--model-filepath', type=str, default='./models/model.pth', 
                        help='Full Filepath to model file')
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
    style = ToolRetrievalInferface(robot, options.upload_filepath, options.model_filepath, silhouette, silhouetteActor)
    vtk_engine.set_interactor_style(style)

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
