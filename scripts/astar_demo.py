""" Plan a path in (X, Y, Theta) space from a start to a goal. """
import numpy as np
from matplotlib import pyplot as plt
import rerun as rr
from pathlib import Path

from conq.astar import find_path, AStar

from regrasping_demo.get_detections import project_hose, cartesian_to_se2, np_to_vec2
from arm_segmentation.predictor import Predictor
from hose_gpe_recorder import load_data
from bosdyn.client import math_helpers
from regrasping_demo.detect_regrasp_point import detect_object_center
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME

def yaw_diff(yaw1, yaw2):
    """
    Compute the absolute difference between two yaw angles, accounting for wraparound.
    EX:
        yaw_diff(0, 2*np.pi) = 0
        yaw_diff(0, 0.1) = 0.1
        yaw_diff(0, -0.1) = 0.1
    """
    return min(abs(yaw1 - yaw2), abs(yaw1 - yaw2 + 2 * np.pi), abs(yaw1 - yaw2 - 2 * np.pi))


def round_node(n):
    """ round to be sure that node equality with floats isn't an issue """
    return round(n[0], 2), round(n[1], 2), round(n[2], 2)


class ConqAStar(AStar):

    def __init__(self, hose_in_body):
        # Circles are defined as (x, y, radius)
        # This set of obstacles is still a np array of R3 points
        # We only care about x
        self.obstacles = hose_in_body
        self.xy_tol = 0.30 # 30cm?
        self.yaw_tol = np.deg2rad(5)

        # A weight of 1 here would mean that 1 radian of yaw error is equivalent to 1 meter of position error.
        # This is not realistic, so instead we weight yaw error such that a full rotation (2 pi) is equivalent to 1 meter.
        self.yaw_weight = 4 / (np.pi)

        self.alignment_weight = 1

    def neighbors(self, n):
        x, y, yaw = n
        d_x = 0.2
        d_y = 0.2
        d_yaw = np.pi / 8
        for dx in [-d_x, 0, d_x]:
            for dy in [-d_y, 0, d_y]:
                for dyaw in [-d_yaw, 0, d_yaw]:
                    if dx == 0 and dy == 0 and dyaw == 0:
                        continue
                    neighbor = (x + dx, y + dy, yaw + dyaw)
                    neighbor = round_node(neighbor)
                    if not self.in_bounds(neighbor):
                        continue
                    if self.in_collision(neighbor):
                        continue
                    yield neighbor

    def is_goal_reached(self, n, goal):
        return abs(n[0] - goal[0]) < self.xy_tol and abs(n[1] - goal[1]) < self.xy_tol and abs(
            n[2] - goal[2]) < self.yaw_tol

    def distance_between(self, n1, n2):
        # The alignment between the edge and the yaw of the node is important.
        xy_dist = np.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)
        yaw_dist = yaw_diff(n1[2], n2[2])
        alignment_dist = yaw_diff(np.arctan2(n2[1] - n1[1], n2[0] - n1[0]), n1[2])
        return sum([
            xy_dist,
            yaw_dist * self.yaw_weight,
            alignment_dist * self.alignment_weight,
        ])

    def heuristic_cost_estimate(self, current, goal):
        return self.distance_between(current, goal)

    def in_bounds(self, n):
        # TODO: Fix these so they aren't hard coded
        return -3 <= n[0] <= 3 and -3 <= n[1] <= 3 and -2 * np.pi <= n[2] <= 2 * np.pi

    def in_collision(self, n):
        for obstacle in self.obstacles:
            dist_to_obstacle = np.sqrt((n[0] - obstacle[0]) ** 2 + (n[1] - obstacle[1]) ** 2)
            if dist_to_obstacle < obstacle[2]:
                return True
        return False

    def viz(self, start, goal, path):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # TODO: These shouldn't be hardcoded either
        ax.set_xlim(-1, 3)
        ax.set_ylim(-2, 1)

        xs = [n[0] for n in path]
        ys = [n[1] for n in path]
        yaws = [n[2] for n in path]

        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='r')
            ax.add_artist(circle)

        ax.plot(xs, ys, 'b-')
        ax.quiver(xs, ys, np.cos(yaws), np.sin(yaws))
        ax.scatter([start[0]], [start[1]], c='b')
        ax.scatter([goal[0]], [goal[1]], c='g')

        fig.show()


def simple_graph_demo():
    neighbors = {
        'A': ['B', 'C'],
        'C': ['D'],
        'D': ['B']
    }
    costs = {
        ('A', 'B'): 100,
        ('A', 'C'): 10,
        ('C', 'D'): 10,
        ('D', 'B'): 20,
    }

    def _neighbors(node):
        if node in neighbors:
            return neighbors[node]
        return []

    def _cost(n1, n2):
        if (n1, n2) in costs:
            return costs[(n1, n2)]
        return 1e9

    def _heuristic(n1, n2):
        return 0

    path = find_path('A', 'B', neighbors_fnct=_neighbors, distance_between_fnct=_cost,
                     heuristic_cost_estimate_fnct=_heuristic)
    print(list(path))
    return path


def main():
    rr.init("hose_gpe")
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [0.4, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 0.4, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 0.4], color=(0, 0, 255), width_scale=0.02)
    info_dict_loaded = load_data("/home/aliryckman/conq_python/scripts/data/info_1689099862.pkl")
    
    predictor = Predictor(Path("hose_regrasping.pth"))

    rgb_np = info_dict_loaded["rgb_np"]
    rgb_res = info_dict_loaded["rgb_res"]
    gpe_in_cam = info_dict_loaded["gpe_in_hand"]
    # gpe_in_body = info_dict_loaded["gpe_in_body"]

    hose_points, projected_points_in_cam, predictions = project_hose(predictor, rgb_np, rgb_res, gpe_in_cam)

    # For now, duplicating code from get_detections/get_hose_and_head_point. Eventually astar will be
    # called by that function

    head_px = detect_object_center(predictions, "vacuum_head")
    rr.log_line_strip("rope", projected_points_in_cam, stroke_width=0.02)

    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.scatter(hose_points[:,0], hose_points[:, 1], c='yellow', zorder=2)
    fig.show()
    
    transforms_cam = rgb_res.shot.transforms_snapshot
    frame_name_shot = rgb_res.shot.frame_name_image_sensor
    projected_points_in_body = []
    for pt_in_cam in projected_points_in_cam:
        vec_in_cam = math_helpers.Vec3(*pt_in_cam)
        # Don't have transform from body frame to gpe, so for now do planning 
        vec_in_gpe = get_a_tform_b(transforms_cam, BODY_FRAME_NAME, frame_name_shot) * vec_in_cam
        projected_points_in_body.append([vec_in_gpe.x, vec_in_gpe.y, vec_in_gpe.z])

    projected_points_in_body = np.array(projected_points_in_body)
    projected_points_in_body = projected_points_in_body[:,:2]
    # TODO: Distances are measured in pixel space. Fix to be measured in R2
    dists = np.linalg.norm(hose_points - head_px, axis=-1)
    best_idx = int(np.argmin(dists))
    best_se2 = cartesian_to_se2(best_idx, projected_points_in_body)

    best_px = hose_points[best_idx]
    best_vec2 = np_to_vec2(best_px)
    
    # delete the hose point that we're trying to walk to from the obstacle list
    # TODO: eventually when we offset from the hose this is irrelevant
    projected_points_in_body = np.delete(projected_points_in_body, (best_idx), axis=0)

    # Currently, the hose is the only obstacle
    # Eventually want to add battery box, stuff from depth?
    projected_t = np.column_stack((projected_points_in_body, 0.15 * np.ones_like(projected_points_in_body[:,0])))
    projected_t = projected_t.T
    projected_t = list(zip(projected_t[0], projected_t[1], projected_t[2]))
    a = ConqAStar(projected_t)
    start = (0,0,0.15)
    # goal is the point on the hose closest to the vacuum head, perhaps create a modified ver of get_hose_and_head_point
    goal = ([best_se2.position.x, best_se2.position.y, 0.15])
    path = list(a.astar(projected_points_in_body, start=start, goal=goal))
    print(path)
    a.viz(start, goal, path)
    return path


if __name__ == '__main__':
    main()
