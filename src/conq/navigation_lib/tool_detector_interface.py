import argparse
import os
import sys

import bosdyn.client.util
import numpy as np
from arm_segmentation.predictor import Predictor
from bosdyn.api import geometry_pb2 as geom
from bosdyn.api import world_object_pb2 as wo
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.util import now_timestamp
from bosdyn.client.world_object import WorldObjectClient, make_add_world_object_req
from dotenv import load_dotenv
from openai import AzureOpenAI

from clickmap_nav.click_map_interface import ClickMapInterface
from clickmap_nav.view_map_highlighted import BosdynVTKInterface, SpotMap, VTKEngine
from conq.cameras_utils import annotate_frame, display_image, get_color_img
from conq.clients import Clients


# TODO: Make class that will loop around in a circle of waypoints
class ToolDetectorInterface(ClickMapInterface):
    def __init__(
        self, robot, upload_path, model_path=None, silhouette=None, silhouetteActor=None
    ):
        # Sets the current working directory to be the same as the file.
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        try:
            if load_dotenv(".env") is False:
                raise TypeError
        except TypeError:
            print("Unable to load .env file.")
            quit()

        # Create Azure client
        self.llm_client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["API_VERSION"],
            azure_endpoint=os.environ["openai_api_base"],
            organization=os.environ["OPENAI_organization"],
        )
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_client = robot.ensure_client(
            ManipulationApiClient.default_service_name
        )
        self.world_object_client = robot.ensure_client(
            WorldObjectClient.default_service_name
        )
        self.clients = Clients(
            image=self.image_client,
            manipulation=self.manipulation_client,
            world_object=self.world_object_client,
        )
        super().__init__(
            robot=robot,
            upload_path=upload_path,
            clients=self.clients,
            silhouette=silhouette,
            silhouetteActor=silhouetteActor,
        )
        self.predictor = None
        if model_path is not None:
            self.predictor = Predictor(model_path)
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_client = robot.ensure_client(
            ManipulationApiClient.default_service_name
        )
        self.world_object_client = robot.ensure_client(
            WorldObjectClient.default_service_name
        )
        self.cameras = [
            "back_fisheye_image",
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
            "left_fisheye_image",
            "right_fisheye_image",
        ]

    def onKeyPressEvent(self, obj, event):
        key, actor = super().onKeyPressEvent(obj, event)

        if key == "n":
            # Navigate in loop
            self.navigate_in_loop()
        elif key == "r":
            # Robot returns to seed/origin point
            self.return_to_seed()
        elif key == "j":
            self.navigate_to_clippers()

        print("Key press!")
        self.print_controls()

    def navigate_in_loop(self):
        # self._navigate_route(self.graph.waypoints)
        print("Navigating in loop!")
        self.toggle_power(should_power_on=True)
        localization_state = self._get_localization_state()
        initial_waypoint_id = localization_state.localization.waypoint_id
        for waypoint in self.waypoint_to_timestamp:
            self._navigate_to([waypoint[0]])
            self.find_object()

    def return_to_seed(self):
        self._navigate_to([self.waypoint_to_timestamp[0]])

    def navigate_to_clippers(self):
        """Navigating based on likely hood of where to find the clippers."""
        # Call function to get probability of where something is
        likely_locations = self.find_likely_locations_for_object(object="clippers")
        best_location = ""
        best_location_probability = 0.0

        for location, probability in likely_locations.items():
            if probability > best_location_probability:
                best_location_probability = probability
                best_location = location
        
        # Find and navigate to the location
        self._navigate_to([self.name_to_id[best_location]])

    def find_likely_locations_for_object(self, object: str):
        """Function will call an LLM in the future to get probabilities of where something likely is."""
        return {"shed": 0.75}

    def find_object(self, min_confidence_thresh=0.65):
        """
        Looks through each of the cameras for an object of type object_class
        Input:
            object_class: string corresponding to the class of object to look for
        Output:
            the pixel centroid of the object of type object_class with the highest confidence
            the ImageResponse object corresponding to the camera that saw the object (so the robot can deduce the pose of the object)
        """
        object_classes = ["hose_handel", "trowel", "clippers", "shovel"]
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
                if (
                    predicted_class in object_classes
                    and confidence >= min_confidence_thresh
                ):
                    binary_mask = (confidence_array >= confidence).astype(np.uint8)
                    color = (255, 0, 0)  # class_name_to_color[class_name]
                    label = f"{confidence} {predicted_class}"
                    center_x, center_y = annotate_frame(
                        rgb_np, binary_mask, mask_label=label, color=color
                    )

                    pixels_and_confidences.append(
                        (
                            (center_x, center_y),
                            confidence,
                            rgb_response,
                            predicted_class,
                            camera,
                        )
                    )

            display_image(rgb_np, window_name=camera, seconds_to_show=1)

        # find the pixel with the highest confidence
        max_confidence = 0
        best_pixel_xy = None
        best_rgb_response = None
        detected_object_class = None
        for (
            pixel_xy,
            confidence,
            rgb_response,
            predicted_class,
            camera_source,
        ) in pixels_and_confidences:
            if confidence > max_confidence:
                max_confidence = confidence
                best_pixel_xy = pixel_xy
                best_rgb_response = rgb_response
                detected_object_class = predicted_class

        # TODO best_rgb_response.frame_name_image_sensor
        if best_pixel_xy is None or best_rgb_response is None:
            print(f"Did not find any objects at this waypoint")
        else:
            print(
                f"Found {detected_object_class} at {best_pixel_xy} in {best_rgb_response.source.name} with confidence {max_confidence}"
            )
            print("Adding object to world objects...")
            self.add_world_object_to_map(pixel_xy, rgb_response)

        return best_pixel_xy, best_rgb_response

    def add_world_object_to_map(self, pixel_xy, image_response):
        # Given a pixel where the object is and using transforms, compute where the
        # object is relative to the robot and then add as a world objet.
        # img_coord = wo.ImageProperties(
        #     camera_source=camera,
        #     image_source=image_response.source,
        #     image_capture=image_response.shot,
        # )
        # wo_obj = wo.WorldObject(
        #     id=2,
        #     name=,
        #     acquisition_time=image_response.shot.acquisition_time,
        #     image_properties=img_coord,
        # )

        # Find transform for vision to body
        vision_tform_body = get_a_tform_b(
            image_response.shot.transforms_snapshot, "vision", "body"
        )

        if vision_tform_body is not None:
            import pdb

            pdb.set_trace()
            pixels_in_body = vision_tform_body.to_matrix() @ np.array(
                [pixel_xy[0], pixel_xy[1], 0, 1]
            )

            # For now, draw a sphere of the pixel coords in body frame
            x = pixels_in_body[0]
            y = pixels_in_body[1]
            z = pixels_in_body[2]

            frame = "body"
            color = (255, 0, 0, 1)
            radius = 0.05

            resp = self.world_object_client.draw_sphere(
                "debug_sphere", x, y, z, frame, radius, color
            )
            print(f"Added a world object sphere at ({x}, {y}, {z})")

            # Get the world object ID set by the service.
            sphere_id = resp.mutated_object_id

            # Add world object

            # List all world objects in the scene after the mutation was applied. Find the sphere in the list
            # and see the transforms added into the frame tree snapshot by Spot in addition to the custom frame.
            world_objects = self.world_object_client.list_world_objects().world_objects
            for world_obj in world_objects:
                if world_obj.id == sphere_id:
                    print(f"Found sphere named {world_obj.name}")
                    full_snapshot = world_obj.transforms_snapshot
                    for edge in full_snapshot.child_to_parent_edge_map:
                        print(
                            f"Child frame name: {edge}. Parent frame name: "
                            f"{full_snapshot.child_to_parent_edge_map[edge].parent_frame_name}"
                        )

            return True
        else:
            print(
                "transform from vision to body is None! Not drawing sphere for object"
            )
            return False

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
            (r) Return to origin/seed pose
            (j) navigate to the clippers
        """
        )

    def add_test_world_object(self):
        # TODO: add world object
        test_wo = wo.WorldObject(name="test_world_object")


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
