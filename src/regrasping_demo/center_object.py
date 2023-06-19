from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from arm_segmentation.predictor import get_combined_mask, Predictor
from arm_segmentation.viz import viz_predictions
from conq.exceptions import DetectionError
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_masks, min_dist_to_mask
from regrasping_demo.homotopy_planner import get_filenames


def center_object_step(rgb_np, predictions, rng, padding=25):
    """
    Returns a delta_px that shows the direction the camera should move to bring the center of the camera
    closer to the center of obstacles that are touching the edge of the frame.

    It also checks for bad perception of the hose, in which case small random delta is used.
    """
    h, w = rgb_np.shape[:2]
    # filter out small instances of obstacles
    obstacle_masks = get_obsacles_near_hose(predictions, min_dist_thresh=150)

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


def get_obsacles_near_hose(predictions: Dict, min_dist_thresh=250):
    obstacle_masks = get_masks(predictions, "battery")
    hose_mask = get_combined_mask(predictions, ['vacuum_hose', 'vacuum_neck', 'vacuum_head'])
    if hose_mask is None:
        raise DetectionError("No hose detected")

    obstacles_near_hose = []
    for obstacle_mask in obstacle_masks:
        # Look at all points on all contours, since we only are about the "outside" points of the obstacle
        binary_obstacle_mask = (obstacle_mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary_obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        obstacle_points = np.concatenate(contours, 0).squeeze()
        min_dist = min([min_dist_to_mask(hose_mask, obstacle_p) for obstacle_p in obstacle_points])
        if min_dist < min_dist_thresh:
            obstacles_near_hose.append(obstacle_mask)
    return obstacles_near_hose


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=2, suppress=True)

    rng = np.random.RandomState(0)
    predictor = Predictor()

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

        predictions = predictor.predict(rgb_np)

        delta_px = center_object_step(rgb_np, predictions, rng)
        print(delta_px)

        fig, ax = plt.subplots()
        ax.imshow(rgb_np, zorder=0)
        viz_predictions(rgb_np, predictions, predictor.colors, fig, ax, legend=False)
        if delta_px is not None:
            ax.arrow(rgb_np.shape[1] / 2, rgb_np.shape[0] / 2, delta_px[0], delta_px[1], width=5, color='r', zorder=2)
        fig.show()


if __name__ == '__main__':
    main()
