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
    rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02)
    info_dict_loaded = load_data("/home/aliryckman/conq_python/scripts/data/info_1689099583.pkl")

    rgb_np = info_dict_loaded["rgb_np"]
    rgb_res = info_dict_loaded["rgb_res"]
    gpe_in_hand = info_dict_loaded["gpe_in_hand"]

    # Vec3 describing a point on the plane
    p0 = Vec3.from_proto(gpe_in_hand.position)
    # 4x4 rotation matrix of gpe
    rot_mat_gpe = gpe_in_hand.rotation.to_matrix()
    plane_q = np.array([gpe_in_hand.rotation.x, gpe_in_hand.rotation.y, gpe_in_hand.rotation.z, gpe_in_hand.rotation.w])
    # normal vector of gpe
    n_np = rot_mat_gpe[0:3,2]
    n = Vec3(x=n_np[0],y=n_np[1],z=n_np[2])
    
    # A point on the line, in this case the line projected onto camera frame

    # get the prediction values for the hose
    predictor = Predictor(Path("hose_regrasping.pth"))
    predictions = predictor.predict(rgb_np)
    saved_fig_name = "cdcpd_output.png"
    ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    rr.log_arrow("plane/n", [p0.x, p0.y,p0.z], [n.x,n.y,n.z])
    rr.log_obb("plane/obb", position=[p0.x, p0.y,p0.z], rotation_q=plane_q, half_size=[2, 2, 0.01])
    for point in ordered_hose_points:
        l = Vec3(*pixel_to_camera_space(rgb_res, point[0], point[1]))
        # normalize so that d can be interpreted in meters
        l = l / l.length()
        rr.log_point("l", [l.x, l.y, l.z], radius=0.05)
        
        # A point on the line, in this case the line projected onto camera frame
        l0 = Vec3(x=0,y=0,z=0)

        # distance to point in camera frame
        # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        d = Vec3.dot((p0-l0), n) / Vec3.dot(l,n)
        print(f"point: {point}, dist: {d}")

        intersection = l0 + l * d
        rr.log_point("intersection_point", [intersection.x, intersection.y, intersection.z], radius=0.05)
    
    #rr.log_line_strip('rope', )
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax, legend=False)
    ax.scatter(ordered_hose_points[:, 0], ordered_hose_points[:, 1], c='yellow', zorder=2)
    plt.show()

    # for every pixel in the image (or once we have the hose, every corresponding point in the hose)    
    
    '''
    
    data_dir = Path("conq_hose")
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue
        img_path_dict = {
            "rgb": "rgb.png",
            "depth": "depth.png",
            "pred": "pred.json"
        }
        all_found = True
        for k, filename in img_path_dict.items():
            img_path_dict[k] = subdir / filename
            if not img_path_dict[k].exists():
                all_found = False
                break
        if not all_found:
            print("skipping ", subdir)
            continue

        rgb_np = np.array(Image.open(img_path_dict["rgb"].as_posix()))

        # Not using the depth image as the depth horizontal FOV is too tight. Using single depth value method instead.
        depth_img = np.ones((rgb_np.shape[0], rgb_np.shape[1]), dtype=float) * METERS_TO_MILLIMETERS

        predictions = predictor.predict(rgb_np)

        saved_fig_name = subdir / "cdcpd_output.png"
        ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

        fig, ax = plt.subplots()
        ax.imshow(rgb_np, zorder=0)
        viz_predictions(rgb_np, predictions, predictor.colors, fig, ax, legend=False)
        ax.scatter(ordered_hose_points[:, 0], ordered_hose_points[:, 1], c='yellow', zorder=2)
        plt.show()
        '''

if __name__ == "__main__":
    main()
