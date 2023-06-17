import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Circle

from conq.exceptions import DetectionError, PlanningException
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
    return np.square((dist_to_start - dist_to_start0) / dist_to_start0)


def is_homotopy_diff(start, end, waypoint1, waypoint2, obstacle_centers):
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


def get_obstacles(predictions, h, w):
    obstacle_polys = get_polys(predictions, "battery")
    obstacle_centers = []
    inflated_obstacles_mask = np.zeros([h, w], dtype=np.uint8)
    for poly in obstacle_polys:
        obstacle_mask = np.zeros([h, w], dtype=np.uint8)
        cv2.drawContours(obstacle_mask, [poly], -1, (255), cv2.FILLED)
        inflated_obstacle_mask = cv2.dilate(obstacle_mask, np.ones([20, 20], dtype=np.uint8))
        inflated_obstacles_mask = cv2.bitwise_or(inflated_obstacles_mask, inflated_obstacle_mask)
        center = np.mean(np.stack(np.where(obstacle_mask == 255)), axis=1)
        center = center[::-1]  # switch from [row, col] to [x, y]
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


def sample_point(rng, h, w, extend_px):
    sample_px = rng.uniform(-extend_px, w + extend_px)
    sample_py = rng.uniform(-extend_px, h + extend_px)
    sample_px = np.array([sample_px, sample_py])
    return sample_px


def plan(rgb_np, predictions, ordered_hose_points, regrasp_px, robot_px, robot_reach_px=700, extend_px=100,
         near_tol=0.4):
    h, w = rgb_np.shape[:2]
    obstacle_centers, inflated_obstacles_mask = get_obstacles(predictions, h, w)

    start_px = ordered_hose_points[0]
    end_px = ordered_hose_points[-1]
    dist_to_start0 = np.linalg.norm(regrasp_px - start_px)
    dist_to_end0 = np.linalg.norm(regrasp_px - end_px)

    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    viz_predictions(rgb_np, predictions, fig, ax)
    ax.scatter(ordered_hose_points[:, 0], ordered_hose_points[:, 1], c='yellow', zorder=2)
    ax.scatter(regrasp_px[0], regrasp_px[1], c='orange', marker='*', s=200, zorder=3)
    ax.add_patch(Circle((robot_px[0], robot_px[1]), robot_reach_px, color='g', fill=False, linewidth=4, alpha=0.5))
    ax.add_patch(Circle((start_px[0], start_px[1]), dist_to_start0, color='g', fill=False, linewidth=4, alpha=0.5))
    ax.add_patch(Circle((end_px[0], end_px[1]), dist_to_end0, color='g', fill=False, linewidth=4, alpha=0.5))
    ax.plot([start_px[0], regrasp_px[0], end_px[0]], [start_px[1], regrasp_px[1], end_px[1]], c='b')
    ax.scatter(obstacle_centers[:, 0], obstacle_centers[:, 1], c='m')
    ax.set_xlim(-extend_px, w + extend_px)
    ax.set_ylim(h + extend_px, -extend_px)

    if np.allclose(regrasp_px, start_px):
        fig.show()
        raise PlanningException("start_px is too close to regrasp_px")
    if np.allclose(regrasp_px, end_px):
        fig.show()
        raise PlanningException("end_px is too close to regrasp_px")

    rng = np.random.RandomState(0)
    for _ in range(5000):
        sample_px = sample_point(rng, h, w, extend_px)

        homotopy_diff = is_homotopy_diff(start_px, end_px, regrasp_px, sample_px, obstacle_centers)
        in_collision = is_in_collision(inflated_obstacles_mask, sample_px)
        reachable = np.linalg.norm(robot_px - sample_px) < robot_reach_px
        near_start = relative_distance_deviation(dist_to_start0, sample_px, start_px) < near_tol
        near_end = relative_distance_deviation(dist_to_end0, sample_px, end_px) < near_tol

        if all([homotopy_diff, not in_collision, near_start, near_end, reachable]):
            ax.plot([start_px[0], sample_px[0], end_px[0]], [start_px[1], sample_px[1], end_px[1]], c='r')
            fig.show()
            return sample_px

    fig.show()
    raise PlanningException("Failed to find a solution")


def get_filenames(subdir):
    img_path_dict = {
        "rgb": "rgb.png",
        "depth": "depth.png",
        "pred": "pred.json"
    }
    for k, filename in img_path_dict.items():
        img_path_dict[k] = subdir / filename
        if not img_path_dict[k].exists():
            return None
    return img_path_dict


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=2, suppress=True)
    rgb_pil = Image.open("data/1686855846/rgb.png")
    rgb_np = np.asarray(rgb_pil)
    with open("data/1686855846/pred.json") as f:
        predictions = json.load(f)
    robot_px = np.array([320, 650])

    # # run CDCPD to get the hose state
    # ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
    #
    # # detect regrasp point
    # min_cost_idx, regrasp_px = detect_regrasp_point_from_hose(rgb_np, predictions, 50, ordered_hose_points)
    #
    #
    # t0 = time.time()
    # plan(rgb_np, predictions, ordered_hose_points, regrasp_px, robot_px)
    # print("Planning took %.3f seconds" % (time.time() - t0))

    rng = np.random.RandomState(0)
    data_dir = Path("homotopy_test_data/")
    n_total = 0
    n_success = 0
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue
        img_path_dict = get_filenames(subdir)
        if not img_path_dict:
            print("skipping ", subdir)
            continue

        n_total += 1
        rgb_pil = Image.open(img_path_dict["rgb"])
        rgb_np = np.asarray(rgb_pil)
        with open(img_path_dict["pred"]) as f:
            predictions = json.load(f)

        try:
            t0 = time.time()
            ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
            min_cost_idx, regrasp_px = detect_regrasp_point_from_hose(rgb_np, predictions, 50, ordered_hose_points,
                                                                      viz=False)
            plan(rgb_np, predictions, ordered_hose_points, regrasp_px, robot_px + rng.uniform(-25, 25, 2).astype(int))
            print("Planning took %.3f seconds" % (time.time() - t0))
            n_success += 1
        except DetectionError as e:
            print(f"detection error {subdir}, {e}")
            continue
        except PlanningException as e:
            print(f"planning error {subdir}, {e}")
            continue

    print(f"success rate: {n_success / n_total:%}")


if __name__ == '__main__':
    main()
