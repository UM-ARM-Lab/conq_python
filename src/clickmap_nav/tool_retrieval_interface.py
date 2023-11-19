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
from conq.cameras_utils import get_color_img , display_image, annotate_frame#, get_depth_img
from conq.manipulation import grasp_point_in_image_basic, \
                                move_gripper_to_pose, \
                                blocking_gripper_open_fraction, \
                                blocking_arm_stow, \
                                follow_gripper_trajectory, \
                                arm_stow, gripper_open_fraction
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
            print(f"Looking for {object_class}...")
            # Go to a waypoint and pick up the tool, then come back
            if actor:
                localization_state = self._get_localization_state()
                initial_waypoint_id = localization_state.localization.waypoint_id
                self.toggle_power(should_power_on=True)
                
                print(f"navigating to: {actor.waypoint_id}")
                self._navigate_to([actor.waypoint_id])

                print(f"Looking for {object_class}...")
                pixel_xy, rgb_response = self.find_object(object_class) # may have to reorient the robot / run multiple times
                
                if rgb_response is not None and pixel_xy is not None:
                    print(f"Picking up {object_class} at {pixel_xy} in {rgb_response.source.name}")
                    grasp_success = self.pick_up_object(pixel_xy, rgb_response)
                    drop_position = [-0.25, 0.0, 0.5]
                    drop_orientation = [0.0,0.0,1.0,0.0]
                    self.drop_object(drop_position, drop_orientation)
                else:
                    print("No objects found")

                print(f"navigating to {initial_waypoint_id}")
                self._navigate_to([initial_waypoint_id])
                self.toggle_power(should_power_on=False)

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
            rgb_np = np.array(rgb_np, dtype=np.uint8)
            # depth_np, depth_res = get_depth_img(self.image_client, 'hand_depth_in_hand_color_frame')
            predictions = self.predictor.predict(rgb_np)
            # List of dictionaries, where each dict contains a confidence, class, and 
            
            for prediction in predictions:
                confidence = prediction["confidence"]
                predicted_class = prediction["class"]
                confidence_array = prediction["mask"]
                if predicted_class == object_class:
                    binary_mask = (confidence_array >= confidence).astype(np.uint8)
                    # print(f"rgb_np shape: {rgb_np.shape}, rgb_np.type {type(rgb_np)}, {type(rgb_np[0,0,0])}")
                    color = (255,0,0) # class_name_to_color[class_name]
                    label = f"{confidence} {object_class}"
                    # TODO: .astype(np.uint16)
                    center_x, center_y = annotate_frame(rgb_np, binary_mask, mask_label=label, color=color)

                    pixels_and_confidences.append(((center_x, center_y), confidence, rgb_response))

            display_image(rgb_np, window_name=camera, seconds_to_show=2)
        
        # find the pixel with the highest confidence
        max_confidence = 0
        best_pixel_xy = None
        best_rgb_response = None
        for pixel_xy, confidence, rgb_response in pixels_and_confidences:
            if confidence > max_confidence:
                max_confidence = confidence
                best_pixel_xy = pixel_xy
                best_rgb_response = rgb_response

        # TODO best_rgb_response.frame_name_image_sensor
        if best_pixel_xy is None or best_rgb_response is None:
            print(f"Did not find {object_class}")
        else:
            print(f"Found {object_class} at {best_pixel_xy} in {best_rgb_response.source.name} with confidence {max_confidence}")

        return best_pixel_xy, best_rgb_response


    def pick_up_object(self, pixel_xy,image_response ):
        """ Pick up the object at the given location."""
        # See the find_plant_demo for an (unorganized) example of how to do this
        # or continuous_hose_regrasping_demo for a more structured example
        grasp_success = False
        for _ in range(2):
            # first just try the auto-grasp
            grasp_success = grasp_point_in_image_basic(self.manipulation_client, self._robot_state_client, image_response, pixel_xy)
            if grasp_success:
                return grasp_success
        
        return grasp_success
    
    def drop_object(self, position=None, orientation=None):
        """ Assuming there is an object in the gripper, move the arm above
         the bucket and open the gripper to drop the object in the bucket"""
        if position is None:
            position = [0.5, 0.0, 0.0]
        if orientation is None:
            orientation = [1.0, 0.0, 0.0, 0.0]
        # Nx8 list of xyx, quat, time
        trajectory_points = [[0.80, 0.0, 0.0,   1.0, 0.0, 0.0, 0.0,               2.0 ], # a reasonable position in front of the robot
                            [0.25, 0.35, 0.55,   1.0, 0.0, 0.0, 0.0,   4.0 ], 
                            #  [0.25, 0.35, 0.5,   0.7071068, 0.0, 0.0, 0.7071068,   8.0 ], # this pose doesn't match my expectations
                             [-0.2, 0.0, 0.55,   0.0, 0.0, 1.0, 0.0,               6.0]
                            #  [-0.2, 0.0, 0.5,  0.0, -0.7071068, 0.7071068, 0.0,   16.0] # rotate sideways
                            ]
                            
        # if move_gripper_to_pose(self._robot_command_client, position, orientation):
        follow_gripper_trajectory(self._robot_command_client, trajectory_points, timeout_sec=10.0)
        gripper_open_fraction(self._robot_command_client, fraction=1.0)
        arm_stow(self._robot_command_client)

        # success = True
        # print(f"Follower trajectory success: {success}")
        # success = success and follow_gripper_trajectory(self._robot_command_client, trajectory_points, timeout_sec=10.0)
        # success = success and blocking_gripper_open_fraction(self._robot_command_client, fraction=1.0, timeout_sec=3.0)
        # print(f"Open gripper success: {success}")
        # success = success and blocking_arm_stow(self._robot_command_client, timeout_sec=10.0)
        # print(f"Stow arm success: {success}")
        # if success:
        #     print("Successfully stowed object")
        #     return True
        # else:
        #     print("Failed to stow object")
        #     return False
        return True


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
