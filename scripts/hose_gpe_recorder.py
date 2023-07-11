#!/usr/bin/env python

import sys
import argparse
import time
import json
import pickle

from pathlib import Path
import matplotlib.pyplot as plt
import rerun as rr

import numpy as np
from PIL import Image
from arm_segmentation.predictor import Predictor
from arm_segmentation.viz import viz_predictions
# from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd, METERS_TO_MILLIMETERS

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, GROUND_PLANE_FRAME_NAME, get_a_tform_b
from bosdyn.client.math_helpers import quat_to_eulerZYX
from bosdyn.api import geometry_pb2

from conq.cameras_utils import get_color_img
from conq.roboflow_utils import get_predictions

def save_data(dictionary):
    now = int(time.time())
    with open(f"data/info_{now}.pkl", 'wb') as f:
        pickle.dump(dictionary, f)
    f.close()
    return f"data/info_{now}.pkl"

def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)
        return data_dict
    
def main(argv):
    # Run full pipeline on all data
    np.seterr(all='raise')
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    args = parser.parse_args(argv)
    rr.init("rope_pull")
    rr.connect()

    bosdyn.client.util.setup_logging(args.verbose)

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('RopePullClient')
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, such as the" \
                                    " estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)

    # Collect image and pose information 
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    
    transforms_hand = rgb_res.shot.transforms_snapshot
    transforms_body = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    # se3 transform from the hand to the ground plane
    gpe_in_hand = get_a_tform_b(transforms_hand, rgb_res.shot.frame_name_image_sensor, ODOM_FRAME_NAME) * get_a_tform_b(transforms_body, ODOM_FRAME_NAME, GROUND_PLANE_FRAME_NAME)

    info_dict = {"rgb_np": rgb_np, "rgb_res": rgb_res, "gpe_in_hand": gpe_in_hand}

    file_name = save_data(info_dict)

    info_dict_loaded = load_data(file_name)
    print(type(info_dict_loaded["rgb_np"]))
    print(type(info_dict_loaded["rgb_res"]))
    print(type(info_dict_loaded["gpe_in_hand"]))
    '''
    # Vec3 describing a point on the plane
    p0 = gpe_in_hand.position
    # 4x4 rotation matrix of gpe
    rot_mat_gpe = gpe_in_hand.rotation.to_matrix()
    # normal vector of gpe
    n = rot_mat_gpe[0:3,2]

    # sample point in camera frame
    point_in_cam = pixel_to_camera_space(rgb_res.shot.image, 40, 50, 1)
    
    #TODO: modify this to be the appropriate type
    # A point on the line intersecting the plane, in this case the origin of the camera frame
    l0 = geometry_pb2.Vec3(0,0,0)

    # vector of arbitrary length to point in cam
    l = geometry_pb2.Vec3(point_in_cam(0), point_in_cam(1), point_in_cam(2))
    predictions = get_predictions(rgb_np)

    save_data(info_dict)
    # for every pixel in the image (or once we have the hose, every corresponding point in the hose)    
    

    predictor = Predictor(Path("hose_regrasping.pth"))
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
    if not main(sys.argv[1:]):
        sys.exit(1)
