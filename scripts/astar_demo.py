""" Plan a path in (X, Y, Theta) space from a start to a goal. """
import numpy as np
from matplotlib import pyplot as plt
import rerun as rr
from pathlib import Path

from conq.astar import find_path, AStar

from regrasping_demo.get_detections import project_hose, construct_se2_from_points, project_points
from regrasping_demo.occupancy_grid import OccupancyGrid
from arm_segmentation.predictor import Predictor
from hose_gpe_recorder import load_data
from bosdyn.client import math_helpers
from regrasping_demo.detect_regrasp_point import detect_object_center, detect_object_points
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

    def __init__(self):
        # Circles are defined as (x, y, radius)
        # This set of obstacles is still a np array of R3 points
        # We only care about x
        self.occupancy_grid = OccupancyGrid()
        self.xy_tol = 0.30 # 30cm?
        self.yaw_tol = np.deg2rad(8)

        # A weight of 1 here would mean that 1 radian of yaw error is equivalent to 1 meter of position error.
        # This is not realistic, so instead we weight yaw error such that a full rotation (2 pi) is equivalent to 1 meter.
        self.yaw_weight = 1 / (2 * np.pi)

        self.alignment_weight = 1

    def neighbors(self, n):
        x, y, yaw = n
        d_x = 0.2
        d_y = 0.2
        d_yaw = np.pi / 12
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

    # TODO: Fix this to use the occupancy grid instead
    def in_collision(self, n):
        return self.occupancy_grid.get_robot_intersection(n)

    def add_obstacle(self, obstacle_x, obstacle_y, radius):
        self.occupancy_grid.add_obstacle(obstacle_x, obstacle_y, radius)

    # the indexing is reversed in this function so that x and y are swapped when graphing
    def viz(self, start, goal, path):
        fig, ax = plt.subplots()
        ax.set_title("A* Planned Path")
        ax.set_aspect('equal')
        ax.set_ylim(min(start[0], goal[0]) - 1, max(start[0], goal[0]) + 1)
        ax.set_xlim(min(start[1], goal[1]) - 1, max(start[1], goal[1]) + 1)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # need to reflect vertically otherwise the graph is mirrored
        ax.invert_xaxis()
        xs = [n[0] for n in path]
        ys = [n[1] for n in path]
        yaws = np.array([n[2] for n in path])
        scale_dim = self.occupancy_grid.get_scaled_dim()
        ax.imshow(self.occupancy_grid.get_grid(), extent=[-scale_dim/2, scale_dim/2, -scale_dim/2, scale_dim/2])

        # for obstacle in self.obstacles:
        #     circle = plt.Circle((obstacle[1], obstacle[0]), obstacle[2], color='r')
        #     ax.add_artist(circle)

        ax.plot(ys, xs, 'b-')
        ax.quiver(ys, xs, np.sin(2 * np.pi - yaws), np.cos(2 * np.pi - yaws))
        ax.scatter([start[1]], [start[0]], c='b')
        ax.scatter([goal[1]], [goal[0]], c='g')
        ax.quiver(goal[1], goal[0], np.sin(2 * np.pi - goal[2]), np.cos(2 * np.pi - goal[2]), color='g')

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
    info_dict_loaded = load_data("/home/aliryckman/conq_python/scripts/data/info_1689099826.pkl")
    
    predictor = Predictor(Path("hose_regrasping.pth"))

    rgb_np = info_dict_loaded["rgb_np"]
    rgb_res = info_dict_loaded["rgb_res"]
    gpe_in_cam = info_dict_loaded["gpe_in_hand"]
    # gpe_in_body = info_dict_loaded["gpe_in_body"]

    a = ConqAStar()

    # In the future, want to record transform between ground plane and gpe to get 'start'. For now, we hard code it
    # to be 0,0,0 since we're planning in the body frame
    # start must be a tuple
    start = (0,0,0)

    # Use arm segmentation's predictor to determine key features of the hose and battery
    hose_points, projected_points_in_cam, predictions = project_hose(predictor, rgb_np, rgb_res, gpe_in_cam)
    battery_px_points = detect_object_points(predictions, "battery")
    projected_battery_in_cam = project_points(battery_px_points, rgb_res, gpe_in_cam)

    # graph hose and battery in rr
    rr.log_line_strip("rope", projected_points_in_cam, stroke_width=0.02)
    rr.log_points("battery", projected_battery_in_cam)

    projected_points_in_cam = np.append(projected_points_in_cam, projected_battery_in_cam, axis=0)

    battery_px = detect_object_center(predictions, "battery")

    # Draw cdcpd hose points prediction on top of the hand RGB image
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.scatter(battery_px[0], battery_px[1], c='orange', zorder=2)
    ax.scatter(hose_points[:,0], hose_points[:, 1], c='yellow', zorder=2)
    fig.show()

    # For now, duplicating code from get_detections/get_hose_and_head_point. Eventually astar will be
    # called by that function instead
    head_px = detect_object_center(predictions, "vacuum_head")

    transforms_cam = rgb_res.shot.transforms_snapshot
    frame_name_shot = rgb_res.shot.frame_name_image_sensor

    # Project the points on the hose 
    projected_points_in_body = []
    for pt_in_cam in projected_points_in_cam:
        vec_in_cam = math_helpers.Vec3(*pt_in_cam)
        # Don't have transform from body frame to gpe, so for now do planning in body frame
        # Once we record new pickles that contain gpe_in_body, we can do planning in gpe instead
        vec_in_gpe = get_a_tform_b(transforms_cam, BODY_FRAME_NAME, frame_name_shot) * vec_in_cam
        projected_points_in_body.append([vec_in_gpe.x, vec_in_gpe.y, vec_in_gpe.z])

    # Trim points so they're in 2D instead of 3D
    projected_points_in_body = np.array(projected_points_in_body)
    projected_points_in_body = projected_points_in_body[:,:2]
    # TODO: Distances are measured in pixel space. Fix to be measured in R2
    dists = np.linalg.norm(hose_points - head_px, axis=-1)
    best_idx = int(np.argmin(dists))
    best_se2 = construct_se2_from_points(best_idx, start, projected_points_in_body)

    # delete the hose point that we're trying to walk to from the obstacle list
    # TODO: eventually when we offset from the hose this is irrelevant
    projected_points_in_body = np.delete(projected_points_in_body, (best_idx), axis=0)
    projected_t = np.column_stack((projected_points_in_body, 0.15 * np.ones_like(projected_points_in_body[:,0])))

    # Add fake obstacles to test A*
    # These points are in (y, x, angle)
    projected_t = np.append(projected_t, [[0.5, -0.5, 0.2],[0.5, -0.75, 0.2],[0.5, 0, 0.2],[0.5,-0.25,0.2]], axis=0)
    projected_t = np.append(projected_t, [[1.5, 1, 0.2],[1.5, 0.75, 0.2],[1.5, 0.5, 0.2],[1.5, 0.25, 0.2],[1.5,0, 0.2],[1.5,0.75,0.2]], axis=0)
    for ob in projected_t:
        a.add_obstacle(ob[0], ob[1], ob[2])
    # projected_t = projected_t.T
    # projected_t = list(zip(projected_t[0], projected_t[1], projected_t[2]))

    # goal is the point on the hose closest to the vacuum head, perhaps create a modified ver of get_hose_and_head_point
    goal = ([best_se2.position.x, best_se2.position.y, best_se2.angle])
    path = list(a.astar(projected_points_in_body, start=start, goal=goal))
    print(path)
    a.viz(start, goal, path)
    return path


if __name__ == '__main__':
    main()
