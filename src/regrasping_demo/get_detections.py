import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from bosdyn.api import image_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.image import pixel_to_camera_space

from conq.cameras_utils import get_color_img, get_depth_img
from conq.exceptions import DetectionError
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_polys, detect_object_center, detect_regrasp_point_from_hose
from conq.roboflow_utils import get_predictions

DEFAULT_IDEAL_DIST_TO_OBS = 70


@dataclass
class GetRetryResult:
    image_res: image_pb2.ImageResponse
    hose_points: np.ndarray
    best_idx: int
    best_vec2: geometry_pb2.Vec2


def np_to_vec2(a):
    return geometry_pb2.Vec2(x=a[0], y=a[1])


def save_data(rgb_np, depth_np, predictions):
    now = int(time.time())
    Path(f"data/{now}").mkdir(exist_ok=True, parents=True)

    Image.fromarray(rgb_np).save(f"data/{now}/rgb.png")
    Image.fromarray(np.squeeze(depth_np)).save(f"data/{now}/depth.png")
    with open(f"data/{now}/pred.json", 'w') as f:
        json.dump(predictions, f)


def get_mess(image_client):
    time.sleep(1)  # reduces motion blur?
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')

    predictions = get_predictions(rgb_np)

    save_data(rgb_np, depth_np, predictions)

    mess_polys = get_polys(predictions, "mess")

    if len(mess_polys) == 0:
        raise DetectionError("No mess detected")

    if len(mess_polys) != 1:
        print(f"Error: expected 1 mess, got {len(mess_polys)}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, alpha=0.5, zorder=0)
    ax.imshow(depth_np, alpha=0.5, zorder=1)
    for mess_poly in mess_polys:
        ax.plot(mess_poly[:, 0], mess_poly[:, 1], zorder=2, linewidth=3)
    fig.show()

    # NOTE: we would need to handle the rotate if we used the body cameras
    mess_mask = np.zeros(depth_np.shape[:2])
    cv2.drawContours(mess_mask, mess_polys, -1, (1), 1)
    # expand the mask a bit
    mess_mask = cv2.dilate(mess_mask, np.ones((5, 5), np.uint8), iterations=1)
    depths_m = depth_np[np.where(mess_mask == 1)] / 1000
    nonzero_depths_m = depths_m[np.where(np.logical_and(depths_m > 0, np.isfinite(depths_m)))]

    depth_m = get_mess_depth(nonzero_depths_m)

    M = cv2.moments(mess_polys[0])
    mess_px = int(M["m10"] / M["m00"])
    mess_py = int(M["m01"] / M["m00"])

    mess_pos_in_cam = np.array(pixel_to_camera_space(rgb_res, mess_px, mess_py, depth=depth_m))  # [x, y, z]

    mess_in_cam = math_helpers.SE3Pose(*mess_pos_in_cam, math_helpers.Quat())
    # FIXME: why can't we use "GRAV_ALIGNED_BODY_FRAME_NAME" here?
    cam2body = get_a_tform_b(rgb_res.shot.transforms_snapshot, BODY_FRAME_NAME,
                             rgb_res.shot.frame_name_image_sensor)
    mess_in_body = cam2body * mess_in_cam
    mess_x = mess_in_body.x
    mess_y = mess_in_body.y

    print(f"Mess detected at {mess_x:.2f}, {mess_y:.2f}")
    return mess_x, mess_y


def get_mess_depth(nonzero_depths_m):
    if len(nonzero_depths_m) > 0:
        depth_m = nonzero_depths_m.mean()
        if np.isfinite(depth_m):
            return depth_m
    return 3.5


def get_hose_and_head_point(image_client):
    time.sleep(1)  # reduces motion blur?
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = get_predictions(rgb_np)
    save_data(rgb_np, depth_np, predictions)

    hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
    head_detection = detect_object_center(predictions, "vacuum_head")

    # fig, ax = viz_detection(rgb_np, head_detection)
    # ax.plot(hose_points[:, 0], hose_points[:, 1], c='w', linewidth=4)
    # fig.show()

    dists = np.linalg.norm(hose_points - head_detection.grasp_px, axis=-1)
    best_idx = int(np.argmin(dists))
    best_px = hose_points[best_idx]
    best_vec2 = np_to_vec2(best_px)

    # ax.scatter(best_px[0], best_px[1], c='m', marker='*', s=100, zorder=10)
    # fig.show()

    return GetRetryResult(rgb_res, hose_points, best_idx, best_vec2)


def get_hose_and_regrasp_point(image_client, ideal_dist_to_obs=DEFAULT_IDEAL_DIST_TO_OBS):
    time.sleep(1)  # reduces motion blur?
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = get_predictions(rgb_np)
    save_data(rgb_np, depth_np, predictions)

    hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    best_idx, best_px = detect_regrasp_point_from_hose(rgb_np, predictions, ideal_dist_to_obs, hose_points)
    best_vec2 = np_to_vec2(best_px)
    return GetRetryResult(rgb_res, hose_points, best_idx, best_vec2)
