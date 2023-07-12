#!/usr/bin/env python

from pathlib import Path
import matplotlib.pyplot as plt
import rerun as rr
import numpy as np
from PIL import Image
from arm_segmentation.predictor import Predictor
from arm_segmentation.viz import viz_predictions
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd, METERS_TO_MILLIMETERS

from bosdyn.client.image import pixel_to_camera_space
from bosdyn.client.math_helpers import Vec3

from hose_gpe_recorder import load_data

def project_hose(rgb_np, rgb_res, gpe_in_hand):
    '''
    Projects the hose in an image into the ground plane.

    TODO: modify these to be more generic types
    Inputs:
        rgb_np: an rgb image of the hose expressed as a numpy array.
        rgb_res: a bosdyn image_pb2.ImageResponse from the image protobuf containing information about the rgb_np
        gpe_in_hand: a bosdyn math_helpers.SE3Pose representing the transform from the camera frame to the GPE frame
    
    Returns:
        ordered_hose_points: the pixel space points on the hose that are to be projected 
        intersection: a numpy array containing 3D points of the hose projected onto the ground plane
    '''

    # Vec3 describing a point on the plane
    p0 = np.array([gpe_in_hand.position.x, gpe_in_hand.position.y, gpe_in_hand.position.z])
    # 4x4 rotation matrix of gpe
    rot_mat_gpe = gpe_in_hand.rotation.to_matrix()
    plane_q = np.array([gpe_in_hand.rotation.x, gpe_in_hand.rotation.y, gpe_in_hand.rotation.z, gpe_in_hand.rotation.w])
    # normal vector of gpe, this is a numpy array
    n = rot_mat_gpe[0:3,2]

    # get the prediction values for the hose
    predictor = Predictor(Path("hose_regrasping.pth"))
    predictions = predictor.predict(rgb_np)
    # n x 2 array with n points and their u, v pixel coordinates
    ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    rr.log_arrow("plane/n", p0, n)
    rr.log_obb("plane/obb", position=p0, rotation_q=plane_q, half_size=[3.5, 3.5, 0.005], label="ground plane")

    l = np.array([*pixel_to_camera_space(rgb_res, ordered_hose_points[:,0], ordered_hose_points[:,1])[0:2]])
    l = np.transpose(l)
    l = np.column_stack((l, np.ones(ordered_hose_points[:,0].shape)))
    l = l / np.linalg.norm(l, axis=1, keepdims=True)
    l0 = np.array([0,0,0])

    d = np.dot((p0 - l0), n) / np.dot(l,n)
    intersection = l0 + l * d[:, np.newaxis]
    return ordered_hose_points, intersection

def main():
    rr.init("hose_gpe")
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [0.4, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 0.4, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 0.4], color=(0, 0, 255), width_scale=0.02)
    info_dict_loaded = load_data("/home/aliryckman/conq_python/scripts/data/info_1689099849.pkl")

    rgb_np = info_dict_loaded["rgb_np"]
    rgb_res = info_dict_loaded["rgb_res"]
    gpe_in_hand = info_dict_loaded["gpe_in_hand"]

    ordered_hose_points, intersection = project_hose(rgb_np, rgb_res, gpe_in_hand)
    rr.log_line_strip("rope", intersection, stroke_width=0.02)
    for i, point in enumerate(intersection):
        rr.log_point(f"intersection_point_{i}", point, radius=0.03)
    rr.log_point(f"intersection_point_{i}", point, radius=0.03, label="hose")
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.scatter(ordered_hose_points[:, 0], ordered_hose_points[:, 1], c='yellow', zorder=2)
    plt.show()

if __name__ == "__main__":
    main()
 