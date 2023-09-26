""" Plan a path in (X, Y, Theta) space from a start to a goal. """
from typing import Tuple

import numpy as np
import rerun as rr
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from conq.astar import AStar
from conq.perception import project_points_in_gpe
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.get_detections import detect_object_points
from regrasping_demo.occupancy_grid import OccupancyGrid


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
        self.xy_tol = 0.1
        self.yaw_tol = np.deg2rad(10)
        # A weight of 1 here would mean that 1 radian of yaw error is equivalent to 1 meter of position error.
        # This is not realistic, so instead we weight yaw error such that a full rotation (2 pi) is equal to 1m.
        self.yaw_weight = 1 / (2 * np.pi)

        self.alignment_weight = 0.25

        # This makes the planner more greedy, by valuing optimistic cost-to-go over cost-to-come
        self.h_weight = 4

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

    def draw_obstacles(self, fig, ax):
        for obs in self.occupancy_grid.obstacles_list:
            # draw a circle
            x, y, r = obs
            circle = plt.Circle((x, y), r, color='r')
            ax.add_artist(circle)

    def is_goal_reached(self, n, goal):
        return abs(n[0] - goal[0]) < self.xy_tol and abs(n[1] - goal[1]) < self.xy_tol and yaw_diff(
            n[2], goal[2]) < self.yaw_tol

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
        return self.distance_between(current, goal) * self.h_weight

    def in_bounds(self, n):
        # TODO: Fix these so they aren't hard coded
        return -5 <= n[0] <= 5 and -5 <= n[1] <= 5 and -2 * np.pi <= n[2] <= 2 * np.pi

    # TODO: Fix this to use the occupancy grid instead
    def in_collision(self, n):
        return self.occupancy_grid.get_robot_intersection(n)

    def add_obstacle(self, obstacle_x, obstacle_y, radius):
        self.occupancy_grid.add_obstacle(obstacle_x, obstacle_y, radius)

    # the indexing is reversed in this function so that x and y are swapped when graphing
    def viz(self, start, goal, path):
        obstacle_points = []
        # TODO: make these same dim as grid
        for x in np.linspace(-4, 4, 100):
            for y in np.linspace(-4, 4, 100):
                if self.occupancy_grid.is_point_occupied(x, y):
                    obstacle_points.append([x, y, 0])
        rr.log_points("obstacles", obstacle_points)

        self.rr_se2('start', start, [0, 1., 0])
        rr.log_obb('start_pose', half_size=[0.5, 0.25, 0.01], position=[start[0], start[1], 0],
                   rotation_q=self.get_quat_from_se2(start))
        for i, point in enumerate(path):
            self.rr_se2(f'path/{i}', point, color=[1., 1., 0])
        self.rr_se2('goal', goal, [0, 1., 0])
        end = path[-1]
        rr.log_obb('end_pose', half_size=[0.5, 0.25, 0.01], position=[end[0], end[1], 0],
                   rotation_q=self.get_quat_from_se2(end))

    def get_quat_from_se2(self, se2):
        rot = Rotation.from_euler('xyz', [0, 0, se2[2]])
        rot_quat = rot.as_quat()
        return rot_quat

    def rr_se2(self, n, se2, color):
        origin = [se2[0], se2[1], 0]
        direction = [np.cos(se2[2]) * self.occupancy_grid.res,
                     np.sin(se2[2]) * self.occupancy_grid.res, 0]
        rr.log_arrow(n, origin, direction, width_scale=0.02, color=color)

    def offset_from_hose(self, se2: Tuple, dist: float):
        """ Offsets the destination pose from the hose so that the robot isn't standing directly over it """
        xa = np.cos(se2[2])
        ya = np.sin(se2[2])
        if abs(xa) < 0.0001:
            dx = 0
            dy = ya * dist
        elif abs(ya) < 0.0001:
            dx = xa * dist
            dy = 0
        else:
            dx = xa * dist
            dy = ya * dist
        return se2[0] - dx, se2[1] - dy, se2[2]


def astar(predictor, rgb_np, rgb_res, gpe_in_cam):
    a = ConqAStar()

    start = (0, 0, 0)

    predictions = predictor.predict(rgb_np)
    hose_pixels = single_frame_planar_cdcpd(rgb_np, predictions)
    hose_points_in_gpe = project_points_in_gpe(hose_pixels, rgb_res, gpe_in_cam)
    battery_px_points = detect_object_points(predictions, "battery")
    battery_in_gpe = project_points_in_gpe(battery_px_points, rgb_res, gpe_in_cam)

    obstacle_points_in_gpe = np.concatenate((hose_points_in_gpe, battery_in_gpe), axis=0)

    # rr.log_line_strip("rope", projected_points_in_cam, stroke_width=0.02)
    # rr.log_points("battery", projected_battery_in_cam)
    # projected_points_in_cam = np.append(projected_points_in_cam, projected_battery_in_cam, axis=0)

    obstacle_point_radius = 0.15
    for ob in obstacle_points_in_gpe:
        a.add_obstacle(ob[0], ob[1], obstacle_point_radius)

    # test_x = np.float64(-0.05)
    # test_y = np.float64(-0.45)
    # a.add_obstacle(test_x, test_y, a.occupancy_grid.res)
    # print(a.occupancy_grid.is_point_occupied(test_x, test_y))

    # Add fake obstacles to test A*
    # (x,y, radius)
    # projected_t = np.append(projected_t, [[0.9, -0.5, 0.2],[0.9, -0.75, 0.2],[0.9, 0, 0.2],[0.9,-0.25,0.2]], axis=0)
    # projected_t = np.append(projected_t, [[1.5, 1, 0.2],[1.5, 0.75, 0.2],[1.5, 0.5, 0.2],[1.5, 0.25, 0.2],[1.5,0, 0.2],[1.5,0.75,0.2]], axis=0)
    # projected_t = projected_t.T
    # projected_t = list(zip(projected_t[0], projected_t[1], projected_t[2]))

    # goal is the point on the hose closest to the vacuum head, perhaps create a modified ver of get_hose_and_head_point
    goal = ([best_se2.position.x, best_se2.position.y, best_se2.angle])
    goal = a.offset_from_hose(goal, 0.3)
    path = list(a.astar(projected_points_in_body, start=start, goal=goal))
    print(path)
    a.viz(start, goal, path)
    return path
