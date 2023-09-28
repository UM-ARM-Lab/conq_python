import matplotlib.pyplot as plt
import numpy as np

from arm_segmentation.predictor import get_combined_mask
from conq.exceptions import DetectionError


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
    mask = get_combined_mask(predictions, class_name)
    if mask is None:
        raise DetectionError(f"No {class_name} detected")
    com = get_center_of_mass(mask)
    return com


def get_center_of_mass(x):
    """
    Returns the center of mass of the given map of probabilities.
    This works by weighting each pixel coordinates by its probability,
    then summing all of these weighted coordinates and dividing by the sum of the probabilities.

    Args:
        x: A map of probabilities.

    Returns:
        The center of mass of the given map of probabilities, like [row, col]
    """
    total_p_mass = x.sum()
    coordinates = np.indices(x.shape)
    # sum over the x and y coordinates
    com = np.sum(np.sum(coordinates * x, 1), 1) / total_p_mass
    row, col = com

    # Check the probability of the center of mass, if it's low that indicates a problem!
    if x[int(com[0]), int(com[1])] < 0.5:
        plt.imshow(x)
        plt.show()
        raise DetectionError("The COM has low probability!")

    x = col
    y = row
    return x, y


def detect_regrasp_point_from_hose(predictions, ordered_hose_points, ideal_dist_to_obs):
    n = ordered_hose_points.shape[0]

    dist_costs = np.zeros(n)

    obstacles_mask = get_combined_mask(predictions, "battery")
    if obstacles_mask is None:
        raise DetectionError(f"No obstacles detected")

    for i, p in enumerate(ordered_hose_points):
        min_d_to_any_obstacle = min_dist_to_mask(obstacles_mask, p)
        dist_costs[i] = abs(min_d_to_any_obstacle - ideal_dist_to_obs)

    total_cost = dist_costs
    min_cost_idx = np.argmin(total_cost)
    best_px = ordered_hose_points[min_cost_idx]

    return min_cost_idx, best_px


def min_dist_to_mask(mask, p, threshold=0.5):
    """
    Returns the minimum distance from the given point to any obstacle in the given mask.

    Args:
        mask: A mask of probabilities.
    """
    mask_ys, mask_xs = np.where(mask > threshold)
    distances = np.sqrt((mask_xs - p[0]) ** 2 + (mask_ys - p[1]) ** 2)
    return distances.min()


def min_angle_to_x_axis(delta):
    angle = np.arctan2(delta[1], delta[0])
    neg_angle = np.arctan2(-delta[1], -delta[0])
    if abs(angle) < abs(neg_angle):
        return angle
    else:
        return neg_angle
