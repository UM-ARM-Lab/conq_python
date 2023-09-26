import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from arm_segmentation.predictor import get_combined_mask
from conq.exceptions import DetectionError, PlanningException
from regrasping_demo import testing
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_masks, detect_regrasp_point_from_hose, get_center_of_mass


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


def poly_to_mask(polys, h, w):
    mask = np.zeros([h, w], dtype=np.uint8)
    cv2.drawContours(mask, polys, -1, (255), cv2.FILLED)
    return mask


def inflate_mask(mask):
    return cv2.dilate(mask, np.ones((20, 20), mask.dtype), iterations=1)


def get_obstacle_coms(predictions):
    obstacles_masks = get_masks(predictions, "battery")
    obstacle_coms = []
    for obstacle_mask in obstacles_masks:
        com = get_center_of_mass(obstacle_mask)
        com = com[::-1]  # switch from [row, col] to [x, y]
        obstacle_coms.append(com)
    obstacle_coms = np.array(obstacle_coms)
    return obstacle_coms


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


def plan(rgb_np, predictions, ordered_hose_points, regrasp_px, robot_px, robot_reach_px=750, extend_px=100,
         near_tol=0.30):
    h, w = rgb_np.shape[:2]
    obstacles_mask = get_combined_mask(predictions, "battery")
    inflated_obstacles_mask = inflate_mask(obstacles_mask)
    obstacle_coms = get_obstacle_coms(predictions)

    start_px = ordered_hose_points[0]
    end_px = ordered_hose_points[-1]
    dist_to_start0 = np.linalg.norm(regrasp_px - start_px)
    dist_to_end0 = np.linalg.norm(regrasp_px - end_px)

    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.set_facecolor("gray")
    # viz_predictions(rgb_np, predictions, class_colors, fig, ax, legend=False)
    ax.scatter(ordered_hose_points[:, 0], ordered_hose_points[:, 1], c='yellow', zorder=2)
    ax.scatter(regrasp_px[0], regrasp_px[1], c='orange', marker='*', s=200, zorder=3)
    ax.scatter(robot_px[0], robot_px[1], c='k', s=500)
    ax.add_patch(
        Circle((robot_px[0], robot_px[1]), robot_reach_px, color=(0.1, 0.6, 0.6), fill=False, linewidth=4, alpha=0.4))
    ax.add_patch(Circle((start_px[0], start_px[1]), dist_to_start0, color='g', fill=False, linewidth=75, alpha=0.2))
    ax.add_patch(Circle((end_px[0], end_px[1]), dist_to_end0, color='b', fill=False, linewidth=75, alpha=0.2))
    line = ax.plot([start_px[0], regrasp_px[0], end_px[0]],
                   [start_px[1], regrasp_px[1], end_px[1]], c='r', linestyle='--', zorder=3)[0]
    scatt = ax.scatter(regrasp_px[0], regrasp_px[1], c='r', marker='*', s=200, zorder=3)
    ax.scatter(obstacle_coms[:, 0], obstacle_coms[:, 1], c='m')
    ax.set_xlim(-extend_px, w + extend_px)
    ax.set_ylim(h + extend_px, -extend_px)
    ax.axis("off")

    random_place_px = regrasp_px + np.random.uniform(-150, 150, [2])
    random_place_px = np.clip(random_place_px, 0, [w, h])
    if np.allclose(regrasp_px, start_px) or np.allclose(regrasp_px, end_px):
        scatt.set_offsets(random_place_px)
        line.set_data([start_px[0], random_place_px[0], end_px[0]],
                      [start_px[1], random_place_px[1], end_px[1]])
        fig.show()
        print("Planning failed, using random place point")
        return False, random_place_px

    rng = np.random.RandomState(0)
    for i in range(5000):
        sample_px = sample_point(rng, h, w, extend_px)

        homotopy_diff = is_homotopy_diff(start_px, end_px, regrasp_px, sample_px, obstacle_coms)
        in_collision = is_in_collision(inflated_obstacles_mask, sample_px)
        reachable = np.linalg.norm(robot_px - sample_px) < robot_reach_px
        near_start = relative_distance_deviation(dist_to_start0, sample_px, start_px) < near_tol
        near_end = relative_distance_deviation(dist_to_end0, sample_px, end_px) < near_tol

        scatt.set_offsets(sample_px)
        line.set_data([start_px[0], sample_px[0], end_px[0]],
                      [start_px[1], sample_px[1], end_px[1]])
        # fig.savefig(f"homotopy_planning/{i:04d}.png")

        if all([homotopy_diff, not in_collision, near_start, near_end, reachable]):
            fig.show()
            return True, sample_px

    print("Failed to find a place point, using random place point")
    scatt.set_offsets(random_place_px)
    line.set_data([start_px[0], random_place_px[0], end_px[0]],
                  [start_px[1], random_place_px[1], end_px[1]])
    fig.show()
    return False, random_place_px


def main():
    robot_px = np.array([320, 650])

    # from PIL import Image
    # predictor = Predictor()
    # from pathlib import Path
    # subdir = Path('homotopy_test_data/ex_for_vid')
    # img_path_dict = get_filenames(subdir)
    # rgb_pil = Image.open(img_path_dict["rgb"])
    # rgb_np = np.asarray(rgb_pil)
    # predictions = predictor.predict(rgb_np)
    # ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
    # min_cost_idx, regrasp_px = detect_regrasp_point_from_hose(predictions, ordered_hose_points, 50)
    # plan(rgb_np, predictions, predictor.colors, ordered_hose_points, regrasp_px, robot_px)

    rng = np.random.RandomState(0)
    n_success = 0
    n_total = 0
    for predictor, subdir, rgb_np, predictions in testing.get_test_examples():
        try:
            n_total += 1
            t0 = time.time()
            ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)
            min_cost_idx, regrasp_px = detect_regrasp_point_from_hose(predictions, ordered_hose_points, 50)
            success, _ = plan(rgb_np, predictions, ordered_hose_points, regrasp_px,
                              robot_px + rng.uniform(-25, 25, 2).astype(int))
            print("Planning took %.3f seconds" % (time.time() - t0))
            if success:
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
