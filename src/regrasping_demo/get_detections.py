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


def get_mess(predictor, image_client):
    get_object_on_floor(predictor, image_client, 'mess')


def get_object_on_floor(predictor, image_client, object_name: str):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')

    predictions = predictor.predict(rgb_np)

    save_data(rgb_np, depth_np, predictions)

    object_masks = get_masks(predictions, desired_class_name=object_name)

    fig, ax = plt.subplots()
    ax.imshow(rgb_np, alpha=0.5, zorder=0)
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
    save_data(rgb_np, depth_np, predictions)

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
    ax.imshow(rgb_np, alpha=0.5, zorder=0)
    ax.imshow(depth_np, alpha=0.3, zorder=1)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
    ax.plot(hose_points[:, 0], hose_points[:, 1], c='w', linewidth=4)
    ax.scatter(best_px[0], best_px[1], c='m', marker='*', s=100, zorder=10)
    ax.scatter(prev_px[0], prev_px[1], c='m', marker='*', s=100, zorder=10)
    fig.show()

    best_vec2 = np_to_vec2(best_px)

    return GetRetryResult(rgb_res, depth_res, depth_np, hose_points, best_idx, best_vec2)


def get_hose_and_regrasp_point(predictor, image_client, ideal_dist_to_obs=DEFAULT_IDEAL_DIST_TO_OBS):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = predictor.predict(rgb_np)
    save_data(rgb_np, depth_np, predictions)

    hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    best_idx, best_px = detect_regrasp_point_from_hose(predictions, hose_points, ideal_dist_to_obs)

    # visualize
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, alpha=0.5, zorder=0)
    ax.imshow(depth_np, alpha=0.3, zorder=1)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
    ax.scatter(best_px[0], best_px[1], s=200, marker='*', c='orange', zorder=4)
    fig.show()

    best_vec2 = np_to_vec2(best_px)
    return GetRetryResult(rgb_res, depth_res, depth_np, hose_points, best_idx, best_vec2)
