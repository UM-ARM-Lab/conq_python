import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request
from scipy import ndimage

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontleft_depth_in_visual_frame': -78,
    'frontright_fisheye_image': -102,
    'frontright_depth_in_visual_frame': -102,
    'hand_depth_in_hand_color_frame': 0,
    'hand_depth': 0,
    'hand_color_image': 0,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def image_to_opencv(image, auto_rotate=True):
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
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

    return img


def get_color_img(image_client, src):
    rgb_req = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
    rgb_res = image_client.get_image([rgb_req])[0]
    rgb_np = image_to_opencv(rgb_res, auto_rotate=True)
    return rgb_np, rgb_res


def get_depth_img(image_client, src):
    depth_req = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16)
    depth_res = image_client.get_image([depth_req])[0]
    depth_np = image_to_opencv(depth_res, auto_rotate=True)
    return depth_np, depth_res
