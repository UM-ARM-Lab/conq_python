from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client.image import build_image_request
import os
import cv2

from conq.navigation.graph_nav.graph_nav_utils import GraphNav

from PIL import Image
import numpy as np
from dotenv import load_dotenv
import time

class WaypointPhotographer:

    def __init__(self, robot, is_debug=False):
        
        load_dotenv('.env.local')
        self._upload_filepath = os.getenv('GRAPH_NAV_GRAPH_FILEPATH')

        self.MEMORY_IMAGE_PATH = os.getenv('MEMORY_IMAGE_PATH')

        self._robot = robot

        self._graph_nav = GraphNav(self._robot, is_debug=is_debug)

        self._image_client = self._robot.ensure_client(ImageClient.default_service_name)

        self._img_sources = ['right_fisheye_image', 'left_fisheye_image', 'back_fisheye_image']

        self.ROTATION_ANGLE = {
                                'back_fisheye_image':               0,
                                'frontleft_fisheye_image':          -78,
                                'frontright_fisheye_image':         -102,
                                'frontleft_depth_in_visual_frame':  -78,
                                'frontright_depth_in_visual_frame': -102,
                                'hand_depth_in_hand_color_frame':   -90,
                                'hand_depth':                       0,
                                'hand_color_image':                 0,
                                'left_fisheye_image':               0,
                                'right_fisheye_image':              180
                            }        

    def _rotate_image(self, img, angle):
        img = np.asarray(Image.fromarray(img).rotate(angle, expand=True))
        return img

    def _image_to_opencv(self, image, auto_rotate=False):
        """Convert an image proto message to an openCV image."""
        num_channels = 1  # Assume a default of 1 byte encodings.
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                num_channels = 3
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                num_channels = 4
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                num_channels = 1
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                num_channels = 1
                dtype = np.uint16

        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into a RGB rows X cols shape.
                img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
            except ValueError:
                # Unable to reshape the image data, trying a regular decode.
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            img = img[:, :, ::-1]

        if auto_rotate:
            angle = self.ROTATION_ANGLE[image.source.name]
            if angle != 0:
                img = self._rotate_image(img, angle)

        return img

    def _take_photos_at_waypoint(self, waypoint_str):

        for src in self._img_sources:
            rgb_request = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
            rgb_response= self._image_client.get_image([rgb_request])[0]
            rgb_np = self._image_to_opencv(rgb_response, auto_rotate=True)
            image = np.array(rgb_np,dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Printing out the shape of the image
            dims = np.shape(image)
            print(f"Saving an image of size: {dims[0]} {dims[1]} {dims[2]} from source {src}")
                
            # Save the image
            cv2.imwrite(self.MEMORY_IMAGE_PATH + src + f"_{waypoint_str}_.jpg", image)    

    def go_to_waypoint_and_take_photos(self, waypoint_name):

        self._graph_nav.navigate_to(waypoint_number=waypoint_name, sit_down_after_reached=False)
        self._take_photos_at_waypoint(waypoint_name)
        # self._graph_nav.toggle_power(should_power_on=False)

    def take_photos_of_full_map(self):

        for waypoint_num in range(0, len(self._graph_nav._current_graph.waypoints)):
            waypoint_name = f'waypoint_{waypoint_num}'
            self.go_to_waypoint_and_take_photos(waypoint_name=waypoint_name)
        self._graph_nav.navigate_to(waypoint_number='waypoint_0')
        self._graph_nav.toggle_power(should_power_on=False)
