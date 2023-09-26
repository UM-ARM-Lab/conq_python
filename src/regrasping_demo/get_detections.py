import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from bosdyn.api import image_pb2, geometry_pb2
from bosdyn.client import math_helpers
from diffcp import SolverError

from arm_segmentation.predictor import get_combined_mask
from arm_segmentation.viz import viz_predictions
from conq.cameras_utils import get_color_img, get_depth_img
from conq.exceptions import DetectionError
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_masks, detect_object_center, detect_regrasp_point_from_hose, \
    get_center_of_mass

DEFAULT_IDEAL_DIST_TO_OBS = 70


@dataclass
class GetRetryResult:
    rgb_res: image_pb2.ImageResponse
    depth_res: image_pb2.ImageResponse
    depth_np: np.ndarray
    hose_points: np.ndarray
    best_idx: int
    best_vec2: geometry_pb2.Vec2


def np_to_vec2(a):
    return geometry_pb2.Vec2(x=a[0], y=a[1])


def vec3_to_np(v):
    return np.array([v.x, v.y, v.z])


def save_data(rgb_np, depth_np, predictions):
    now = int(time.time())
    Path(f"data/{now}").mkdir(exist_ok=True, parents=True)

    Image.fromarray(rgb_np).save(f"data/{now}/rgb.png")
    Image.fromarray(np.squeeze(depth_np)).save(f"data/{now}/depth.png")

    data_dict = {
        'rgb':         rgb_np,
        'depth':       depth_np,
        'predictions': predictions,
    }
    with open(f"data/{now}.pkl", 'wb') as f:
        pickle.dump(data_dict, f)


def detect_object_points(predictions, class_name):
    """"
    Returns a set of points in a N x 2 np array representing the mask
    Basically takes the mask and splits it into a grid
    """
    mask = get_combined_mask(predictions, class_name)
    # take every 50th pixel to be a part of the mask
    obj_mask_idxs = np.fliplr(np.transpose(np.nonzero(mask)))
    grid_idxs = obj_mask_idxs[0::50]
    return grid_idxs


def get_mess(predictor, image_client):
    get_object_on_floor(predictor, image_client, 'mess')


def get_object_on_floor(predictor, image_client, object_name: str):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')

    predictions = predictor.predict(rgb_np)

    # save_data(rgb_np, depth_np, predictions)
    save_all_rgb(image_client)

    object_masks = get_masks(predictions, desired_class_name=object_name)

    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.imshow(depth_np, alpha=0.3, zorder=1)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
    fig.show()

    if len(object_masks) == 0:
        raise DetectionError("No object detected")

    if len(object_masks) != 1:
        print(f"Error: expected 1 object, got {len(object_masks)}")

    object_px, object_py = get_center_of_mass(object_masks[0])

    return GetRetryResult(rgb_res, depth_res, depth_np, None, None, math_helpers.Vec2(object_px, object_py))


def get_hose_and_head_point(predictor, image_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = predictor.predict(rgb_np)
    # save_data(rgb_np, depth_np, predictions)
    save_all_rgb(image_client)

    try:
        hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
    except SolverError:
        raise DetectionError("CDCPD Solver failed")
    head_px = detect_object_center(predictions, "vacuum_head")
    head_xy = head_px[::-1]

    # TODO do this in ground plane space instead of image space
    dists = np.linalg.norm(hose_points - head_xy, axis=-1)
    best_idx = int(np.argmin(dists))
    best_px = hose_points[best_idx]
    if best_idx == 0:
        prev_idx = 1
    else:
        prev_idx = best_idx - 1
    prev_px = hose_points[prev_idx]

    # interpolate from best_px to prev_px
    alpha = 15  # in pixels
    iterp_dir = (prev_px - best_px).astype(float)
    iterp_dir /= np.linalg.norm(iterp_dir)  # normalize
    iterp_dir *= alpha
    best_px += iterp_dir.astype(int)

    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.imshow(depth_np, alpha=0.3, zorder=1)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
    ax.plot(hose_points[:, 0], hose_points[:, 1], c='w', linewidth=4)
    ax.scatter(best_px[0], best_px[1], c='m', marker='*', s=100, zorder=10)
    ax.scatter(prev_px[0], prev_px[1], c='m', marker='*', s=100, zorder=10)
    fig.show()

    best_vec2 = np_to_vec2(best_px)

    return GetRetryResult(rgb_res, depth_res, depth_np, hose_points, best_idx, best_vec2)


def save_all_rgb(image_client):
    rgb_sources = [
        'back_fisheye_image',
        'frontleft_fisheye_image',
        'frontright_fisheye_image',
        'left_fisheye_image',
        'right_fisheye_image',
    ]
    now = int(time.time())
    for src in rgb_sources:
        rgb_np, rgb_res = get_color_img(image_client, src)
        filename = Path(f"data/all_rgb/{now}_{src}.png")
        Image.fromarray(rgb_np).save(filename)


def get_hose_and_regrasp_point(predictor, image_client, ideal_dist_to_obs=DEFAULT_IDEAL_DIST_TO_OBS):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = predictor.predict(rgb_np)

    # save_data(rgb_np, depth_np, predictions)
    save_all_rgb(image_client)

    hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    best_idx, best_px = detect_regrasp_point_from_hose(predictions, hose_points, ideal_dist_to_obs)

    # visualize
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.imshow(depth_np, alpha=0.3, zorder=1)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
    ax.scatter(best_px[0], best_px[1], s=200, marker='*', c='orange', zorder=4)
    fig.show()

    best_vec2 = np_to_vec2(best_px)
    return GetRetryResult(rgb_res, depth_res, depth_np, hose_points, best_idx, best_vec2)


def get_body_goal_se2_from_hose_points(hose_points, best_idx):
    """
    Construct a SE2Pose from the hose points and the best index

    Args:
        hose_points: a Nx2 np array of points on the hose, in GPE frame
        best_idx: the index of the best point on the hose
    """
    x = hose_points[best_idx][0]
    y = hose_points[best_idx][1]

    # generate rotation from the rate of change
    # if the selected point is not the first or last on the hose
    if best_idx == 0:
        p_n1 = hose_points[best_idx]
        p_n2 = hose_points[best_idx + 1]
    elif best_idx == len(hose_points) - 1:
        p_n1 = hose_points[best_idx - 1]
        p_n2 = hose_points[best_idx]
    else:
        # rotation is calculated using the two adjacent hose points
        p_n1 = hose_points[best_idx - 1]
        p_n2 = hose_points[best_idx + 1]

    # slope of line connecting neighbors
    m = (p_n2[1] - p_n1[1]) / (p_n2[0] - p_n1[0])

    # Construct a vector perpendicular to this line
    inv_m = -1.0 / m
    dir_vec = np.array([1, inv_m])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    # find the angle between the vector and the x-axis
    angle = np.arctan(dir_vec[1] / dir_vec[0])

    # If start angle and the calculated pose are more than 90 degrees apart, flip the final pose
    start_to_goal_vec = np.array(
        [hose_points[best_idx][0], hose_points[best_idx][1]])
    start_to_goal_vec = start_to_goal_vec / np.linalg.norm(dir_vec)
    if np.dot(dir_vec, start_to_goal_vec) < 0:
        angle = (angle + np.pi) % (2 * np.pi)

    return x, y, angle
