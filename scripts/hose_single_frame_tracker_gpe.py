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

def main():
    rr.init("hose_gpe")
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [0.2, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 0.2, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 0.2], color=(0, 0, 255), width_scale=0.02)
    info_dict_loaded = load_data("/home/aliryckman/conq_python/scripts/data/info_1689099583.pkl")

    rgb_np = info_dict_loaded["rgb_np"]
    rgb_res = info_dict_loaded["rgb_res"]
    gpe_in_hand = info_dict_loaded["gpe_in_hand"]

    # Vec3 describing a point on the plane
    p0 = np.array([gpe_in_hand.position.x, gpe_in_hand.position.y, gpe_in_hand.position.z])
    # 4x4 rotation matrix of gpe
    rot_mat_gpe = gpe_in_hand.rotation.to_matrix()
    plane_q = np.array([gpe_in_hand.rotation.x, gpe_in_hand.rotation.y, gpe_in_hand.rotation.z, gpe_in_hand.rotation.w])
    # normal vector of gpe, this is a numpy array
    n = rot_mat_gpe[0:3,2]
    # n = Vec3(x=n[0],y=n[1],z=n[2])
    
    # A point on the line, in this case the line projected onto camera frame

    # get the prediction values for the hose
    predictor = Predictor(Path("hose_regrasping.pth"))
    predictions = predictor.predict(rgb_np)
    saved_fig_name = "cdcpd_output.png"
    # n x 2 array with n points and their u, v pixel coordinates
    ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    rr.log_arrow("plane/n", p0, n)
    rr.log_obb("plane/obb", position=p0, rotation_q=plane_q, half_size=[3.5, 3.5, 0.005])

    l = np.array([*pixel_to_camera_space(rgb_res, ordered_hose_points[:,0], ordered_hose_points[:,1])[0:2]])
    l = np.transpose(l)
    l = np.column_stack((l, np.ones(ordered_hose_points[:,0].shape)))
    l = l / np.linalg.norm(l, axis=1, keepdims=True)
    l0 = np.array([0,0,0])

    d = np.dot((p0 - l0), n) / np.dot(l,n)
    intersection = l0 + l * d[:, np.newaxis]

    rr.log_line_strip("rope", intersection, stroke_width=0.1)
    for i, point in enumerate(intersection):
        rr.log_point(f"intersection_point_{i}", point, radius=0.05)
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax, legend=False)
    ax.scatter(ordered_hose_points[:, 0], ordered_hose_points[:, 1], c='yellow', zorder=2)
    plt.show()

if __name__ == "__main__":
    main()
 