from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from arm_segmentation.predictor import get_combined_mask
from conq.exceptions import DetectionError
from conq.roboflow_utils import get_predictions
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.viz import pred_to_poly


@dataclass
class DetectionResult:
    grasp_px: np.ndarray
    candidates_pxs: np.ndarray
    predictions: List[Dict[str, np.ndarray]]


def get_masks(predictions, desired_class_name):
    masks = []
    for pred in predictions:
        class_name = pred["class"]
        mask = pred["mask"]

        if isinstance(desired_class_name, list):
            if class_name in desired_class_name:
                masks.append(mask)
        elif class_name == desired_class_name:
            masks.append(mask)
    return masks


def detect_object_center(predictions, class_name):
    mask = get_combined_mask(predictions, class_name),
    grasp_point = np.array(get_center_of_mass(mask))
    detection = DetectionResult(grasp_point, np.array([grasp_point]), predictions)
    return detection


def get_center_of_mass(x):
    """
    Returns the center of mass of the given map of probabilities.
    This works by weighting each pixel coordinates by its probability,
    then summing all of these weighted coordinates and dividing by the sum of the probabilities.
    """
    total_p_mass = x.sum()
    coordinates = np.indices(x.shape)
    # sum over the x and y coordinates
    com = np.sum(np.sum(coordinates * x, 1), 1) / total_p_mass

    # Check the probability of the center of mass, if it's low that indicates a problem!
    if x[int(com[0]), int(com[1])] < 0.5:
        plt.imshow(x)
        plt.show()
        raise DetectionError("The COM has low probability!")

    return com


def detect_regrasp_point(rgb_np, predictions, ideal_dist_to_obs):
    ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
    min_cost_idx, best_px = detect_regrasp_point_from_hose(rgb_np, predictions, ordered_hose_points, ideal_dist_to_obs)

    return DetectionResult(best_px, ordered_hose_points, predictions)


def detect_regrasp_point_from_hose(rgb_np, predictions, ordered_hose_points, ideal_dist_to_obs, viz=True):
    n = ordered_hose_points.shape[0]

    dist_costs = np.zeros(n)

    obstacles_mask = get_combined_mask(predictions, "battery")
    if obstacles_mask is None:
        raise DetectionError(f"No obstacles detected")

    for i, p in enumerate(ordered_hose_points):
        min_d_to_any_obstacle = min_dist_to_obstacles(obstacles_mask, p)
        dist_costs[i] = abs(min_d_to_any_obstacle - ideal_dist_to_obs)

    total_cost = dist_costs
    min_cost_idx = np.argmin(total_cost)
    best_px = ordered_hose_points[min_cost_idx]

    if viz:
        fig, ax = plt.subplots()
        ax.imshow(rgb_np)
        for pred in predictions:
            points = pred_to_poly(pred)
            ax.plot(points[:, 0], points[:, 1])
        for p in ordered_hose_points:
            ax.scatter(p[0], p[1], color='y', zorder=3)
        ax.scatter(best_px[0], best_px[1], s=100, marker='*', c='m', zorder=4)
        fig.show()

    return min_cost_idx, best_px


def min_dist_to_obstacles(obstacles_mask, p):
    obstacle_pixels = np.argwhere(obstacles_mask)
    distances = np.sqrt((obstacle_pixels[:, 0] - p[0]) ** 2 + (obstacle_pixels[:, 1] - p[1]) ** 2)
    return distances.min()


def min_angle_to_x_axis(delta):
    angle = np.arctan2(delta[1], delta[0])
    neg_angle = np.arctan2(-delta[1], -delta[0])
    if abs(angle) < abs(neg_angle):
        return angle
    else:
        return neg_angle


def main():
    test_image_filename = "data/1686856761/rgb.png"

    rgb_pil = Image.open(test_image_filename)
    rgb_np = np.asarray(rgb_pil)

    predictions = get_predictions(rgb_np)

    detection = detect_regrasp_point(rgb_np, predictions, ideal_dist_to_obs=50)
    print(detection)


if __name__ == "__main__":
    main()
