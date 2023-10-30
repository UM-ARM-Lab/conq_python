import os

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from .base import CameraWeedDetectorBase

BBOX_COLOR = (75, 0, 130)

class CameraWeedDetectorRosWrapper(CameraWeedDetectorBase):
    """A weed detector for a *single camera*, meaning that each camera would construct one of these
    objects to listen for images and process images of potential weeds.
    """
    def __init__(self, camera_name: str, img_bridge: CvBridge, subscribe_rgb: bool=True,
                 subscribe_depth: bool=False, save_imgs: bool=False):
        super().__init__(camera_name, subscribe_rgb, subscribe_depth, save_imgs)
        self.img_bridge = img_bridge

        # if self.subscribe_rgb:
        #     self.rgb_topic = f"camera/{self.camera_name}/image"
        # else:
        #     self.rgb_topic = None

        # if self.subscribe_depth:
        #     self.depth_topic = f"depth/{self.camera_name}/image"
        # else:
        #     self.depth_topic = None

    def subscribe(self):
        """Subscribes the detector to the given topics"""
        # TODO: How to synchronize depth and RGB?
        if self.subscribe_rgb:
            print(f"Subscribing to {self.camera_name} RGB topic: '{self.rgb_topic}'")
            rgb_subscriber = rospy.Subscriber(self.rgb_topic, Image, self.detect_weed_callback,
                                              queue_size=1)
        else:
            rgb_subscriber = None
            print(f"NOT subscribing {self.camera_name} to RGB")

        if self.subscribe_depth:
            print(f"Subscribing to {self.camera_name} Depth topic: '{self.rgb_topic}'")
            # rgb_subscriber = rospy.Subscriber(self.rgb_topic, Image, self.detect_weed_callback,
            #                                   queue_size=1)
            raise RuntimeError("Haven't implemented synchronous RGB/depth subscription!")
        else:
            print(f"NOT subscribing {self.camera_name} to Depth")
            depth_subscriber = None

        subscribers = {
            "rgb": rgb_subscriber,
            "depth": depth_subscriber
        }
        return subscribers

    def detect_weed_callback(self, img_msg):
        """Detects 'weeds' in an image using simple HSV segmentation"""
        # Transform image message to OpenCV
        img = self.img_bridge.imgmsg_to_cv2(img_msg)

        # TODO: Grab the last synchronized depth image as well.

        # Save the raw images.
        if self.save_imgs:
            raw_rgb_path = os.path.join(self._get_rgb_img_dir(), "raw.png")
            cv2.imwrite(raw_rgb_path, img)
            print(f"Wrote raw RGB from {self.camera_name} camera to:", raw_rgb_path)

            # TODO: Depth

        # Call actual weed detection segmentation routine.
        cent, bbox = self.detect_weed(img)

        # Draw output on the image and save if desired.
        if self.save_imgs:
            img_processed = img.copy()
            cv2.rectangle(img_processed, bbox.top_left(), bbox.bottom_right(), BBOX_COLOR, 3)
            cv2.circle(img_processed, (cent.x, cent.y), 4, BBOX_COLOR, -1)

            processed_rgb_path = os.path.join(self._get_rgb_img_dir(), "processed.png")
            cv2.imwrite(processed_rgb_path, img_processed)
            print(f"Wrote processed RGB from {self.camera_name} camera to:", raw_rgb_path)

        # Publish the centroid pixel?
        # Perhaps publish the centroid pixel along with "weed confidence"?
        # Could just make a ROS message for weed centroid, bounding box coordinates, and the confidence
        # Get the frame name for the camera so that we can later use the frame name with the grasping
        # API.
        # self.camera_name


        self.img_num += 1