import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from bosdyn.api import image_pb2, geometry_pb2, ray_cast_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.image import pixel_to_camera_space

from arm_segmentation.viz import viz_predictions
from conq.cameras_utils import get_color_img, get_depth_img
from conq.exceptions import DetectionError
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_masks, detect_object_center, detect_regrasp_point_from_hose, \
    get_center_of_mass

DEFAULT_IDEAL_DIST_TO_OBS = 70


@dataclass
class GetRetryResult:
    image_res: image_pb2.ImageResponse
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
    with open(f"data/{now}/pred.json", 'w') as f:
        json.dump(predictions, f)


def get_mess(predictor, rc_client, image_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')

    predictions = predictor.predict(rgb_np)

    save_data(rgb_np, depth_np, predictions)

    mess_masks = get_masks(predictions, "mess")

    if len(mess_masks) == 0:
        raise DetectionError("No mess detected")

    if len(mess_masks) != 1:
        print(f"Error: expected 1 mess, got {len(mess_masks)}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, alpha=0.5, zorder=0)
    ax.imshow(depth_np, alpha=0.5, zorder=1)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
    fig.show()

    mess_px, mess_py = get_center_of_mass(mess_masks[0])

    # compute position in camera frame of the pixel given a fixed depth, which has the right direction in 3D
    # but may be the wrong length.
    mess_dir_in_cam = np.array(pixel_to_camera_space(rgb_res, mess_px, mess_py, 1))
    mess_dir_in_cam = mess_dir_in_cam / np.linalg.norm(mess_dir_in_cam)
    cam_in_body = get_a_tform_b(rgb_res.shot.transforms_snapshot, BODY_FRAME_NAME, rgb_res.shot.frame_name_image_sensor)
    mess_dir_in_body = vec3_to_np(cam_in_body.rot * math_helpers.Vec3(*mess_dir_in_cam))
    mess_origin_in_body = vec3_to_np(cam_in_body.position)

    # Ray-cast to the floor (or whatever is nearest based on the robot's perception of the world)
    response = rc_client.raycast(mess_origin_in_body, mess_dir_in_body,
                                 [ray_cast_pb2.RayIntersection.Type.TYPE_GROUND_PLANE],
                                 min_distance=1.0, frame_name=BODY_FRAME_NAME)

    if len(response.hits) == 0:
        raise DetectionError("No raycast hits")

    hit = response.hits[0]
    mess_x = hit.hit_position_in_hit_frame.x
    mess_y = hit.hit_position_in_hit_frame.y

    print(f"Mess detected at {mess_x:.2f}, {mess_y:.2f}")
    return mess_x, mess_y


def get_hose_and_head_point(predictor, image_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = predictor.predict(rgb_np)
    save_data(rgb_np, depth_np, predictions)

    hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
    head_px = detect_object_center(predictions, "vacuum_head")

    # fig, ax = viz_detection(rgb_np, head_detection)
    # ax.plot(hose_points[:, 0], hose_points[:, 1], c='w', linewidth=4)
    # fig.show()

    dists = np.linalg.norm(hose_points - head_px, axis=-1)
    best_idx = int(np.argmin(dists))
    best_px = hose_points[best_idx]
    best_vec2 = np_to_vec2(best_px)

    # ax.scatter(best_px[0], best_px[1], c='m', marker='*', s=100, zorder=10)
    # fig.show()

    return GetRetryResult(rgb_res, hose_points, best_idx, best_vec2)


def get_hose_and_regrasp_point(predictor, image_client, ideal_dist_to_obs=DEFAULT_IDEAL_DIST_TO_OBS):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = predictor.predict(rgb_np)
    save_data(rgb_np, depth_np, predictions)

    hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    best_idx, best_px = detect_regrasp_point_from_hose(predictions, hose_points, ideal_dist_to_obs)
    best_vec2 = np_to_vec2(best_px)
    return GetRetryResult(rgb_res, hose_points, best_idx, best_vec2)
