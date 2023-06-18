import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from conq.exceptions import DetectionError
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_masks
from regrasping_demo.homotopy_planner import poly_to_mask, get_filenames


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=2, suppress=True)

    rng = np.random.RandomState(0)

    data_dir = Path("homotopy_test_data/")
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue
        img_path_dict = get_filenames(subdir)
        if not img_path_dict:
            print("skipping ", subdir)
            continue

        rgb_pil = Image.open(img_path_dict["rgb"])
        rgb_np = np.asarray(rgb_pil)
        with open(img_path_dict["pred"]) as f:
            predictions = json.load(f)

        delta_px = center_object_step(rgb_np, predictions, rng)
        print(delta_px)


def center_object_step(rgb_np, predictions, rng, padding=25):
    """
    Returns a delta_px that shows the direction the camera should move to bring the center of the camera
    closer to the center of obstacles that are touching the edge of the frame.

    It also checks for bad perception of the hose, in which case small random delta is used.
    """
    h, w = rgb_np.shape[:2]
    # filter out small instances of obstacles
    obstacle_masks = get_obs_masks_near_hose(predictions, h, w, min_dist_thresh=150)

    # check if any obstacle is touching the edge
    for mask in obstacle_masks:
        ys, xs = np.where(mask)
        x_edge = np.logical_or(xs < padding, xs > w - padding)
        y_edge = np.logical_or(ys < padding, ys > h - padding)
        if np.any(x_edge) or np.any(y_edge):
            centroid = np.stack([xs, ys], -1).mean(0)
            delta_px = centroid - np.array([w / 2, h / 2])

            return delta_px

    # Check that we can see an instance of the hose that is long enough, otherwise we move a small random amount
    length = 0
    try:
        ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
        deltas = ordered_hose_points[1:] - ordered_hose_points[:-1]
        length = sum(np.linalg.norm(deltas, axis=-1))
    except DetectionError:
        pass

    if length < 350:
        delta_px = rng.uniform(-100, 100, [2]).astype(int)
        return delta_px

    return None


def get_obs_masks_near_hose(predictions: Dict, h, w, min_dist_thresh=250):
    obstacle_polys = get_masks(predictions, "battery")
    hose_polys = get_masks(predictions, ['vacuum_hose', 'vacuum_neck', 'vacuum_head'])
    if len(hose_polys) == 0:
        raise DetectionError("No hose detected")

    obs_near_hose = []
    for obstacle_poly in obstacle_polys:
        min_dist = min([dist_to_nearest_poly(obstacle_p, hose_polys) for obstacle_p in obstacle_poly])
        mask = poly_to_mask([obstacle_poly], h, w)
        if min_dist < min_dist_thresh:
            obs_near_hose.append(mask)
    return obs_near_hose


def dist_to_nearest_poly(p, polys):
    return min([-cv2.pointPolygonTest(poly, tuple(map(int, p)), True) for poly in polys])


if __name__ == '__main__':
    main()
