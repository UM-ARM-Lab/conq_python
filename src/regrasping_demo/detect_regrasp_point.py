import itertools
from dataclasses import dataclass
from typing import List, Dict

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from conq.roboflow_utils import get_predictions


class DetectionError(Exception):
    pass


@dataclass
class DetectionResult:
    grasp_px: np.ndarray
    candidates_pxs: np.ndarray
    predictions: List[Dict[str, np.ndarray]]


def viz_detection(rgb_np, detection):
    fig, ax = plt.subplots()
    ax.imshow(rgb_np)
    rng = np.random.RandomState(0)
    class_colors = {}
    for pred in detection.predictions:
        points = pred["points"]
        class_name = pred["class"]
        if class_name not in class_colors:
            class_colors[class_name] = cm.hsv(rng.uniform())
        x = [p['x'] for p in points]
        y = [p['y'] for p in points]
        c = class_colors[class_name]
        ax.plot(x, y, c=c, linewidth=2, zorder=1)
    ax.scatter(detection.candidates_pxs[:, 0], detection.candidates_pxs[:, 1], color="y", marker="x", s=100,
               label='candidates',
               zorder=2)
    ax.scatter(detection.grasp_px[0], detection.grasp_px[1], color="green", marker="o", s=100, label='grasp point',
               zorder=3)
    ax.legend()
    fig.show()

    return fig, ax


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


def pred_to_poly(pred):
    points = pred["points"]
    points = np.array([(p['x'], p['y']) for p in points], dtype=int)
    return points


def detect_object_center(predictions, class_name):
    polygons = get_polys(predictions, class_name)

    if len(polygons) == 0:
        raise DetectionError(f"No {class_name} detected")
    if len(polygons) > 1:
        print("Warning: multiple objects detected")

    poly = polygons[0]
    M = cv2.moments(poly)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    grasp_point = np.array([cx, cy])

    detection = DetectionResult(grasp_point, np.array([grasp_point]), predictions)

    return detection


def fit_hose_model(hose_polygons, n_clusters=8):
    from sklearn.cluster import KMeans

    # 8 is the max we can do exhaustively, since it grows exponentially!
    hose_points = np.concatenate(hose_polygons, 0)
    n_clusters = int(min(n_clusters, hose_points.shape[0] / 2))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(hose_points)
    clusters = kmeans.cluster_centers_

    # organize the points into line segments with the shortest total length
    def _len_cost(points):
        deltas = points[1:] - points[:-1]
        lengths = np.linalg.norm(deltas, axis=-1)
        return lengths.sum()

    min_length = np.inf
    best_ordered_hose_points = None
    for permutation in itertools.permutations(range(clusters.shape[0])):
        ordered_hose_points = clusters[list(permutation)]
        cost = _len_cost(ordered_hose_points)
        if cost < min_length:
            min_length = cost
            best_ordered_hose_points = ordered_hose_points

    return best_ordered_hose_points


def hose_points_from_predictions(predictions):
    # FIXME: remove blue_rope class!
    hose_polys = get_polys(predictions, ["vacuum_hose", "vacuum_neck", "blue_rope"])
    if len(hose_polys) == 0:
        raise DetectionError("No hose detected")
    ordered_hose_points = fit_hose_model(hose_polys)
    return ordered_hose_points


def detect_regrasp_point(rgb_np, predictions, ideal_dist_to_obs):
    ordered_hose_points = hose_points_from_predictions(predictions)
    min_cost_idx, best_px = detect_regrasp_point_from_hose(rgb_np, predictions, ideal_dist_to_obs, ordered_hose_points)

    return DetectionResult(best_px, ordered_hose_points, predictions)


def detect_regrasp_point_from_hose(rgb_np, predictions, ideal_dist_to_obs, ordered_hose_points):
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

    import matplotlib.pyplot as plt
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
