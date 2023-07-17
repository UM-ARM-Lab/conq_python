import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client.image import build_image_request
from scipy import ndimage
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME

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
        angle = ROTATION_ANGLE[image.source.name]
        if angle != 0:
            img = ndimage.rotate(img, angle)

    return img


def get_color_img(image_client, src):
    rgb_req = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
    rgb_res: ImageResponse = image_client.get_image([rgb_req])[0]
    rgb_np = image_to_opencv(rgb_res, auto_rotate=True)
    return rgb_np, rgb_res


def get_depth_img(image_client, src):
    depth_req = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16)
    depth_res: ImageResponse = image_client.get_image([depth_req])[0]
    depth_np = image_to_opencv(depth_res, auto_rotate=True)
    return depth_np, depth_res


def rotate_image_coordinates(pts, width, height, angle):
    """
    Rotate image coordinates by rot degrees around the center of the image.

    Args:
        pts: Nx2 array of image coordinates
        width: width of image
        height: height of image
        angle: rotation in degrees
    """
    center = np.array([width / 2, height / 2])
    new_pts = center + (pts - center) @ rot_2d(np.deg2rad(angle)).T
    return new_pts


def rot_2d(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])


def camera_space_to_pixel(image_proto, x, y, z):
    """ Inverse of pixel_to_camera_space """
    focal_x = image_proto.source.pinhole.intrinsics.focal_length.x
    principal_x = image_proto.source.pinhole.intrinsics.principal_point.x

    focal_y = image_proto.source.pinhole.intrinsics.focal_length.y
    principal_y = image_proto.source.pinhole.intrinsics.principal_point.y

    pixel_x = (x * focal_x) / z + principal_x
    pixel_y = (y * focal_y) / z + principal_y

    return pixel_x, pixel_y


# FIXME: this functions may be wrong or misleadingly named???
def pos_in_cam_to_pos_in_hand(p_in_cam):
    """
    Convert a point in camera space to a point in hand space.
    Camera space is like image space, with +X pointing along the columns and +Y pointing along the rows.
    Hand space is defined by BD's hand frame, as shown here:
        https://dev.bostondynamics.com/docs/concepts/arm/arm_concepts.html?#hand-frame
    This does not account for the offset between the camera sensor and the hand frame,
    and gives only an approximation of the rotation.
    """
    return np.array([-p_in_cam[1], -p_in_cam[0]])

# Returns the ground plane estimate in the hand frame 
def gpe_frame_in_cam(robot_state_client, rgb_res):
    transforms_hand = rgb_res.shot.transforms_snapshot
    transforms_body = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    # se3 transform from the cam to the ground plane
    return get_a_tform_b(transforms_hand, rgb_res.shot.frame_name_image_sensor, ODOM_FRAME_NAME) * get_a_tform_b(transforms_body, ODOM_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
