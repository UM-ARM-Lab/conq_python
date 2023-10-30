import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#### NOTE: For the find_plant_demo.py, all the functionality from detect_weed2() and below is not used. 
# Furthermore, the bd_wrapper and ros_wrapper are not used.

class BoundingBox:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def top_left(self):
        return (self.x, self.y)

    def bottom_right(self):
        return (self.x + self.width, self.y + self.height)

class PixelCoord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class CameraWeedDetectorBase:
    def __init__(self, camera_name: str, subscribe_rgb: bool=True, subscribe_depth: bool=False,
                 save_imgs: bool=False):
        self.camera_name = camera_name
        self.subscribe_rgb = subscribe_rgb
        self.subscribe_depth = subscribe_depth
        self.save_imgs = save_imgs
        # For saving images.
        self.img_num = 0

        # TODO: This seems pretty ROS-specific. Does the Boston Dynamics API require this type of
        # setup?
        if self.subscribe_rgb:
            self.rgb_topic = f"camera/{self.camera_name}/image"
        else:
            self.rgb_topic = None

        if self.subscribe_depth:
            self.depth_topic = f"depth/{self.camera_name}/image"
        else:
            self.depth_topic = None

    def detect_weed(self, img):
        """Detects the 'weed' in the image by segmenting and returning the single largest contour"""
        # Call weed detection function which returns a contour of the "weed"
        # Rescale the images to 0-1 for each channel
        # rescaled = img.astype(np.float32) / 255
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define variables to store the best HSV filter values
        lower_h = 38
        upper_h = 90
        lower_s = 79
        upper_s = 201
        lower_v = 37
        upper_v = 255

        lower_range = np.array([lower_h, lower_s, lower_v])
        upper_range = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
    
        # Erode the images to get rid of noise
        erosion_size = 3
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        eroded = cv2.erode(mask.astype(np.uint8), el)

        # Now dilate by a slightly larger kernel to close gaps
        dilation_size = erosion_size + 2
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated = cv2.dilate(eroded, el)

        # Now get contours of the weeds.
        cnt, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(cnt))
        max_cnt = max(cnt, key=cv2.contourArea)

        # find the centroid pixel of the weed.
        M = cv2.moments(max_cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        contour_centroid = PixelCoord(cx, cy)

        # find the bounding box of the weed.
        x, y, w, h  = cv2.boundingRect(max_cnt)
        bbox = BoundingBox(x, y, w, h)

        return contour_centroid, bbox
    

    def detect_weed2(self, img, debug=False):
        """Detects the 'weed' in the image by segmenting and returning the single largest contour"""
        # Call weed detection function which returns a contour of the "weed"
        # Rescale the images to 0-1 for each channel
        rescaled = img.astype(np.float32) / 255
        hsv = cv2.cvtColor(rescaled, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        if debug:
            fig, ax = plt.subplots(ncols=3)
            ax[0].imshow(hue)
            ax[1].imshow(sat)
            ax[2].imshow(val)

        h_seg = np.logical_and(hue > 70, hue < 170)
        seg = np.logical_and(h_seg, sat > 0.2)
        if debug:
            plt.figure()
            plt.imshow(seg)

        # Erode the images to get rid of noise
        erosion_size = 3
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))

        eroded = cv2.erode(seg.astype(np.uint8), el)
        if debug:
            plt.figure()
            plt.imshow(eroded)

        # Now dilate by a slightly larger kernel to close gaps
        dilation_size = erosion_size + 2
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        if debug:
            plt.figure()
            plt.imshow(el)

        dilated = cv2.dilate(eroded, el)
        if debug:
            plt.figure()
            plt.imshow(dilated)

        # Now get contours of the weeds.
        cnt, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(cnt))
        max_cnt = max(cnt, key=cv2.contourArea)

        # find the centroid pixel of the weed.
        M = cv2.moments(max_cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        contour_centroid = PixelCoord(cx, cy)

        # find the bounding box of the weed.
        x, y, w, h  = cv2.boundingRect(max_cnt)
        bbox = BoundingBox(x, y, w, h)

        return contour_centroid, bbox

    def _get_saved_imgs_root(self):
        # TODO: Better specification of directory to save images.
        return os.path.join(self.camera_name, f"img_{self.img_num}")

    def _get_rgb_img_dir(self):
        rgb_path = os.path.join(self._get_saved_imgs_root(), "rgb")
        os.makedirs(rgb_path, exist_ok=True)
        return rgb_path

    def _get_depth_img_dir(self):
        depth_path = os.path.join(self._get_saved_imgs_root(), "depth")
        os.makedirs(depth_path, exist_ok=True)
        return depth_path