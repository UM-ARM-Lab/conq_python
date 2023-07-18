""" Plan a path in (X, Y, Theta) space from a start to a goal. """
import numpy as np
from matplotlib import pyplot as plt

from conq.astar import find_path, AStar


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
        self.obstacles = [
            (0.3, 0.1, 0.15),
            (0.3, 0.3, 0.15),
            (0.3, 0.5, 0.15),
            (0.3, 0.7, 0.15),
            (0.8, 0.3, 0.15),
            (0.8, 0.5, 0.15),
            (0.8, 0.7, 0.15),
            (0.8, 0.9, 0.15),
        ]
        self.xy_tol = 0.01
        self.yaw_tol = np.deg2rad(5)

        # A weight of 1 here would mean that 1 radian of yaw error is equivalent to 1 meter of position error.
        # This is not realistic, so instead we weight yaw error such that a full rotation (2 pi) is equivalent to 1 meter.
        self.yaw_weight = 1 / (2 * np.pi)

        self.alignment_weight = 1

    def neighbors(self, n):
        x, y, yaw = n
        d_x = 0.1
        d_y = 0.1
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
        return 0 <= n[0] <= 1 and 0 <= n[1] <= 1 and -2 * np.pi <= n[2] <= 2 * np.pi

    def in_collision(self, n):
        for obstacle in self.obstacles:
            dist_to_obstacle = np.sqrt((n[0] - obstacle[0]) ** 2 + (n[1] - obstacle[1]) ** 2)
            if dist_to_obstacle < obstacle[2]:
                return True
        return False

    def viz(self, start, goal, path):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

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
    a = ConqAStar()
    start = (0, 0, 0)
    goal = (1.0, 1.0, np.deg2rad(90))
    path = list(a.astar(start, goal))
    print(path)
    a.viz(start, goal, path)

    return path


if __name__ == '__main__':
    main()
