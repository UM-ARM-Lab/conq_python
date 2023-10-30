from typing import Tuple

import numpy as np
import cv2
from bosdyn.client.robot import Robot
from bosdyn.client.image import ImageClient
from bosdyn.api import geometry_pb2, image_pb2

from .base import CameraWeedDetectorBase, PixelCoord

class CameraWeedDetectorBDWrapper(CameraWeedDetectorBase):
    image_client: ImageClient

    def __init__(self, robot: Robot, camera_name: str, subscribe_rgb: bool=True,
                 subscribe_depth: bool=False,
                 save_imgs: bool=False):
        """Creates a weed detector that communicates with Spot via the Boston Dynamics SDK API

        NOTE: Assumes the robot has already been constructed with sdk.create_robot(config.hostname)
        and has been authenticated with bosdyn.client.util.authenticate(robot)

        NOTE: `camera_name` is the "image_source" that the API is expecting, e.g.
        frontleft_fisheye_image.
        """
        super().__init__(camera_name, subscribe_rgb, subscribe_depth, save_imgs)
        self.robot = robot
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)

    def get_weed_coords_and_transforms(self) -> Tuple[image_pb2.ImageResponse, PixelCoord]:
        # Grab image(s).
        image_responses = self.image_client.get_image_from_sources([self.camera_name])
        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False

        # Verify image response data.
        image_msg = image_responses[0]
        if image_msg.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            # I believe this is for depth data.
            dtype = np.uint16
        else:
            # I believe this is for RGB data.
            dtype = np.uint8

        img = np.fromstring(image_msg.shot.image.data, dtype=dtype)
        if image_msg.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image_msg.shot.image.rows, image_msg.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        cent, bbox = self.detect_weed(img)

        vec_to_weed = geometry_pb2.Vec2(x=cent.x, y=cent.y)

        return image_msg, vec_to_weed


