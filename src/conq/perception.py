import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME, GROUND_PLANE_FRAME_NAME
from bosdyn.client.image import pixel_to_camera_space


def project_points_in_gpe(px: np.ndarray, rgb_res, cam2gpe: math_helpers.SE3Pose):
    """
    Project points in camera frame into the ground plane (XY plane) in the frame defined by cam2gpe
    Inputs:
        px: [b, 2] pixels to be projected into ground plane in [col, row] order
        rgb_res: a bosdyn image_pb2.ImageResponse from the image protobuf containing information about the rgb_np
        cam2gpe: a bosdyn math_helpers.SE3Pose representing the transform from the camera frame to the GPE frame

    Returns:
        points: the projected points in R3 in the GPE frame
    """
    valid_points_in_cam = project_points_in_cam(px, rgb_res, cam2gpe)

    cam2gpe_np = cam2gpe.inverse().to_matrix()
    xyz_in_cam_homo = np.concatenate((valid_points_in_cam, np.ones([valid_points_in_cam.shape[0], 1])), 1)
    xyz_in_gpe = (cam2gpe_np @ xyz_in_cam_homo.T)[:3].T

    return xyz_in_gpe


def project_points_in_cam(px, rgb_res, cam2gpe):
    """
    Project px into the ground plane defined by the XY plane of cam2gpe,
    and return the points still in the camera frame.

    Args:
        px: [b, 2] pixels to be projected into ground plane in [col, row] order
        rgb_res: a bosdyn image_pb2.ImageResponse from the image protobuf containing information about the rgb_np
        cam2gpe: a bosdyn math_helpers.SE3Pose representing the transform from the camera frame to the GPE frame

    """
    p0 = np.array([cam2gpe.position.x, cam2gpe.position.y, cam2gpe.position.z])
    rot_mat_gpe = cam2gpe.rotation.to_matrix()
    n = rot_mat_gpe[0:3, 2]
    zs = np.ones(px.shape[0])
    l = np.array([*pixel_to_camera_space(rgb_res, px[:, 0], px[:, 1], depth=zs)])
    l = np.transpose(l)
    l = l / np.linalg.norm(l, axis=1, keepdims=True)
    l0 = np.array([0, 0, 0])
    l_dot_n = np.dot(l, n)
    d = np.dot((p0 - l0), n) / l_dot_n
    points = l0 + l * d[:, np.newaxis]
    # some points won't intersect the ground plane because they are "above" the horizon
    valid_points_in_cam = points[np.where(np.abs(l_dot_n) > 1e-3)]
    return valid_points_in_cam


def get_gpe_in_cam(rgb_res, robot_state_client):
    transforms_hand = rgb_res.shot.transforms_snapshot
    transforms_body = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    odon_in_cam = get_a_tform_b(transforms_hand, rgb_res.shot.frame_name_image_sensor, ODOM_FRAME_NAME)
    gpe_in_odom = get_a_tform_b(transforms_body, ODOM_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    gpe_in_cam = odon_in_cam * gpe_in_odom
    return gpe_in_cam
