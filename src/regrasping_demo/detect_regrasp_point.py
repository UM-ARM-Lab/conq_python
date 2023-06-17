from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List, Dict

import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image

from conq.exceptions import DetectionError
from conq.roboflow_utils import get_predictions
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.viz import pred_to_poly


@dataclass
class DetectionResult:
    grasp_px: np.ndarray
    candidates_pxs: np.ndarray
    predictions: List[Dict[str, np.ndarray]]


def get_polys(predictions, desired_class_name):
    polys = []
    for pred in predictions:
        class_name = pred["class"]
        points = pred_to_poly(pred)

        if isinstance(desired_class_name, list):
            if class_name in desired_class_name:
                polys.append(points)
        elif class_name == desired_class_name:
            polys.append(points)
    return polys


def detect_object_center(predictions, class_name):
    polygons = get_polys(predictions, class_name)

    if len(polygons) == 0:
        raise DetectionError(f"No {class_name} detected")
    if len(polygons) > 1:
        print("Warning: multiple objects detected")

    poly = polygons[0]
    grasp_point = np.array(get_poly_centroid(poly))

    detection = DetectionResult(grasp_point, np.array([grasp_point]), predictions)

    return detection


def get_poly_centroid(x):
    M = cv2.moments(x)
    mess_px = int(M["m10"] / M["m00"])
    mess_py = int(M["m01"] / M["m00"])
    return mess_px, mess_py


def detect_regrasp_point(rgb_np, predictions, ideal_dist_to_obs):
    ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
    min_cost_idx, best_px = detect_regrasp_point_from_hose(rgb_np, predictions, ideal_dist_to_obs, ordered_hose_points)

    return DetectionResult(best_px, ordered_hose_points, predictions)


def detect_regrasp_point_from_hose(rgb_np, predictions, ideal_dist_to_obs, ordered_hose_points, viz=True):
    n = ordered_hose_points.shape[0]

    dist_costs = np.zeros(n)

    obstacle_class_name = "battery"
    obstacle_polygons = get_polys(predictions, obstacle_class_name)
    if len(obstacle_polygons) == 0:
        raise DetectionError(f"No {obstacle_class_name} detected")
    for i, p in enumerate(ordered_hose_points):
        min_d_to_any_obstacle = min_dist_to_obstacles(obstacle_polygons, p)
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
        cost_normalized = (total_cost - total_cost.min()) / (total_cost.max() - total_cost.min())
        for cost, p in zip(cost_normalized, ordered_hose_points):
            color = cm.hsv(cost)
            ax.scatter(p[0], p[1], color=color, zorder=3)
        ax.scatter(best_px[0], best_px[1], s=100, marker='*', c='m', zorder=4)
        fig.show()

    return min_cost_idx, best_px


def min_dist_to_obstacles(obstacle_polygons, p):
    min_d_to_any_obstacle = np.inf
    for obstacle_poly in obstacle_polygons:
        # dist is positive if the point is outside the polygon
        dist = -cv2.pointPolygonTest(obstacle_poly, p.tolist(), True)
        if dist < min_d_to_any_obstacle:
            min_d_to_any_obstacle = dist
    return min_d_to_any_obstacle


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
