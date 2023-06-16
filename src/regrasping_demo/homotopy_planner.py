import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from regrasping_demo.cdcpd_hose_state_predictor import do_single_frame_cdcpd_prediction
from regrasping_demo.detect_regrasp_point import get_polys, detect_regrasp_point


def make_tau(start, goal, waypoint):
    def _tau(t):
        """
        Represents the piecewise linear path: start --> waypoint --> goal.
        The "time" is arbitrary, so we place 0.5 at the waypoint.
        """
        if t <= 0.5:
            return start + 2 * t * (waypoint - start)
        else:
            return waypoint + 2 * (t - 0.5) * (goal - waypoint)

    return _tau


def angle_between(w, v):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    return np.arctan2(w[1] * v[0] - v[1] * w[0], w[0] * v[0] + w[1] * v[1])


def compare_waypoint_homotopy(start, goal, waypoint1, waypoint2, obstacle_centers):
    tau1 = make_tau(start, goal, waypoint1)
    tau2 = make_tau(start, goal, waypoint2)
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
    return np.allclose(windings, 0)


def main():
    rgb = Image.open("data/1686855846/rgb.png")
    rgb_np = np.asarray(rgb)
    h, w = rgb_np.shape[:2]
    with open("data/1686855846/pred.json") as f:
        predictions = json.load(f)

    # Convert the image and predictions into the obstacle, start, goal, and midpoint representations
    regrasp_detection = detect_regrasp_point(rgb_np, predictions, 50)
    regrasp_px = regrasp_detection.grasp_px

    # NOTE: everything is in pixel space here
    obstacle_polys = get_polys(predictions, "battery")
    obstacle_centers = []
    for poly in obstacle_polys:
        m = np.zeros([h, w], dtype=np.uint8)
        cv2.drawContours(m, [poly], -1, (255), cv2.FILLED)
        center = np.mean(np.stack(np.where(m == 255)), axis=1)
        # for some reason x/y are swapped?
        center = center[::-1]
        obstacle_centers.append(center)
    obstacle_centers = np.array(obstacle_centers)

    # run CDCPD
    ordered_hose_points = do_single_frame_cdcpd_prediction(bgr_img, depth_img, masks)

    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.set_ylim(-100, h + 100)
    ax.set_xlim(-100, w + 100)
    for poly in obstacle_polys:
        ax.fill(poly[:, 0], poly[:, 1], c='k')
    for obstacle_center in obstacle_centers:
        ax.scatter(obstacle_center[0], obstacle_center[1], c='m')
    ax.scatter(start_px[0], start_px[1], c='g')
    ax.scatter(regrasp_px[0], regrasp_px[1], c='b')
    ax.scatter(goal_px[0], goal_px[1], c='g')
    fig.show()

    rng = np.random.RandomState(0)
    while True:
        # TODO: sample candidate place points such that tau() is entirely collision free
        candidate_px = rng.uniform(-100, 700)
        candidate_py = rng.uniform(-100, 600)
        candidate_place_px = np.array([candidate_px, candidate_py])

        fig, ax = plt.subplots()
        ax.axis("equal")
        ax.set_xlim(-100, 700)
        ax.set_ylim(-100, 600)
        ax.plot([start_px[0], regrasp_px[0], goal_px[0]],
                [start_px[1], regrasp_px[1], goal_px[1]],
                c='b')
        ax.plot([start_px[0], candidate_place_px[0], goal_px[0]],
                [start_px[1], candidate_place_px[1], goal_px[1]],
                c='r')
        ax.scatter(obstacle_centers[:, 0], obstacle_centers[:, 1], c='m')
        fig.show()

        is_same = compare_waypoint_homotopy(start_px, goal_px, regrasp_px, candidate_place_px,
                                            obstacle_centers)
        print(is_same)

        is_collision_free = True

        if is_collision_free and not is_same:
            print(candidate_place_px)
            break


if __name__ == '__main__':
    main()
