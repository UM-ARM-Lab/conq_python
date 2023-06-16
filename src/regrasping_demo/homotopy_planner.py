import json
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_polys, detect_regrasp_point_from_hose
from regrasping_demo.viz import viz_predictions


def make_tau(start, end, waypoint):
    def _tau(t):
        """
        Represents the piecewise linear path: start --> waypoint --> end.
        The "time" is arbitrary, so we place 0.5 at the waypoint.
        """
        if t <= 0.5:
            return start + 2 * t * (waypoint - start)
        else:
            return waypoint + 2 * (t - 0.5) * (end - waypoint)

    return _tau


def angle_between(w, v):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    return np.arctan2(w[1] * v[0] - v[1] * w[0], w[0] * v[0] + w[1] * v[1])


def relative_distance_deviation(dist_to_start0, sample_px, start_px):
    dist_to_start = np.linalg.norm(sample_px - start_px)
    return abs((dist_to_start - dist_to_start0) / dist_to_start0)


def check_is_homotopy_diff(start, end, waypoint1, waypoint2, obstacle_centers):
    tau1 = make_tau(start, end, waypoint1)
    tau2 = make_tau(start, end, waypoint2)
    windings = []
    # compute winding number for each obstacle
    for object_center in obstacle_centers:
        # to compute the winding number for a given obstacle,
        # go from 0 to 1 and compute the angle between
        # the vector from the obstacle to the path and the +X axis,
        # then integrate these angles, and you'll get a total rotation of either 0 or 2pi
        dt = 0.05
        integral_angle = 0
        last_vec = None
        for t in np.arange(0, 2, dt):
            if t < 1:
                z = tau1(t)
            else:
                z = tau2(2 - t)
            vec = z - object_center
            if last_vec is not None:
                angle = angle_between(vec, last_vec)
                integral_angle += angle
            last_vec = vec

        # round to nearest multiple of 2 pi
        winding_number = np.round(integral_angle / (2 * np.pi)) * 2 * np.pi
        windings.append(winding_number)

    windings = np.array(windings)
    return not np.allclose(windings, 0)


def get_obstacle_centers(predictions, h, w):
    obstacle_polys = get_polys(predictions, "battery")
    obstacle_centers = []
    inflated_obstacles_mask = np.zeros([h, w], dtype=np.uint8)
    for poly in obstacle_polys:
        obstcle_mask = np.zeros([h, w], dtype=np.uint8)
        cv2.drawContours(obstcle_mask, [poly], -1, (255), cv2.FILLED)
        inflated_obstacle_mask = cv2.dilate(obstcle_mask, np.ones([20, 20], dtype=np.uint8))
        inflated_obstacles_mask = cv2.bitwise_or(inflated_obstacle_mask, obstcle_mask)
        center = np.mean(np.stack(np.where(obstcle_mask == 255)), axis=1)
        # switch from [row, col] to [x, y]
        center = center[::-1]
        obstacle_centers.append(center)
    obstacle_centers = np.array(obstacle_centers)
    return obstacle_centers, inflated_obstacles_mask


def is_in_collision(inflated_obstacles_mask, sample_px):
    # Assume that OOB points are not in collision
    if sample_px[0] < 0 or sample_px[0] >= inflated_obstacles_mask.shape[1]:
        return False
    if sample_px[1] < 0 or sample_px[1] >= inflated_obstacles_mask.shape[0]:
        return False
    return bool(inflated_obstacles_mask[int(sample_px[1]), int(sample_px[0])])


def main():
    rgb_pil = Image.open("data/1686855846/rgb.png")
    rgb_np = np.asarray(rgb_pil)
    with open("data/1686855846/pred.json") as f:
        predictions = json.load(f)

    # run CDCPD to get the hose state
    ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    # detect regrasp point
    min_cost_idx, best_px = detect_regrasp_point_from_hose(rgb_np, predictions, 50, ordered_hose_points)

    t0 = time.time()
    plan(rgb_np, predictions, ordered_hose_points, best_px)
    print("Planning took %.3f seconds" % (time.time() - t0))


def plan(rgb_np, predictions, ordered_hose_points, best_px,
         dist_tol_px=0.8, robot_reach_px=700, robot_px=np.array([700, -200])):
    h, w = rgb_np.shape[:2]
    obstacle_centers, inflated_obstacles_mask = get_obstacle_centers(predictions, h, w)

    start_px = ordered_hose_points[0]
    end_px = ordered_hose_points[-1]
    dist_to_start0 = np.linalg.norm(best_px - start_px)
    dist_to_end0 = np.linalg.norm(best_px - end_px)

    rng = np.random.RandomState(0)
    while True:
        sample_px = sample_point(rng, h, w, extend_px=100)
        # sample_px = np.array([250, 50])

        is_diff = check_is_homotopy_diff(start_px, end_px, best_px, sample_px, obstacle_centers)

        # Check against inflated obstacles
        is_collision_free = not is_in_collision(inflated_obstacles_mask, sample_px)

        # Check against robot reachability
        is_near_robot = np.linalg.norm(robot_px - sample_px) < robot_reach_px

        is_near_start = relative_distance_deviation(dist_to_start0, sample_px, start_px) < dist_tol_px
        is_near_end = relative_distance_deviation(dist_to_end0, sample_px, end_px) < dist_tol_px

        if all([is_collision_free, is_diff, is_near_start, is_near_end, is_near_robot]):
            print(sample_px)

            fig, ax = plt.subplots()
            ax.imshow(rgb_np, zorder=0)
            viz_predictions(rgb_np, predictions, fig, ax)
            ax.scatter(ordered_hose_points[:, 0], ordered_hose_points[:, 1], c='yellow', zorder=2)
            ax.scatter(best_px[0], best_px[1], c='orange', marker='*', s=200, zorder=3)
            # ax.set_xlim(-100, 900)
            # ax.set_ylim(800, -100)
            ax.plot([start_px[0], best_px[0], end_px[0]], [start_px[1], best_px[1], end_px[1]], c='b')
            ax.plot([start_px[0], sample_px[0], end_px[0]], [start_px[1], sample_px[1], end_px[1]], c='r')
            ax.scatter(obstacle_centers[:, 0], obstacle_centers[:, 1], c='m')
            fig.show()
            break


def sample_point(rng, h, w, extend_px):
    sample_px = rng.uniform(-extend_px, w + extend_px)
    sample_py = rng.uniform(-extend_px, h + extend_px)
    sample_px = np.array([sample_px, sample_py])
    return sample_px


if __name__ == '__main__':
    main()
