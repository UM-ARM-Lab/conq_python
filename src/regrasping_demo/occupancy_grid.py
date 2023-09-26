import matplotlib.pyplot as plt
import cv2
import rerun as rr

import numpy as np
from math import cos, sin


class OccupancyGrid():
    def __init__(self, dim: int = 300, res: float = 0.05) -> None:
        """
        The occupancy grid covers (dim * res)^2 meters, centered at (0,0)
        dim must be divisible by 2
        """
        self.obstacles_list = []
        assert (dim % 2 == 0), "Occupancy grid dimensions must be divisible by 2."
        self.dim = dim
        self.res = res
        self.grid = np.zeros([dim, dim])

    def is_point_occupied(self, x: float, y: float):
        u, v = self.get_bucket(x, y)
        return self.grid[u, v]

    def get_bucket(self, x: float, y: float):
        """
        Given a coordinate in the frame of the occupancy grid, return the corresponding bucket's grid coordinates and occupancy        
        """
        u = -1 * (x / self.res) + self.dim / 2
        v = self.dim / 2 - (y / self.res)
        # TODO: double check these boundary conditions are right
        assert np.any((0 <= u) & (u < self.dim)), "Assertion Error: Calculated bucket would be outside grid dimensions"
        assert np.any((0 <= v) & (v < self.dim)), "Assertion Error: Calculated bucket would be outside grid dimensions"
        return u.astype(int), v.astype(int)

    def get_robot_intersection(self, se2, robot_width: float = 0.5, robot_height: float = 1, res: int = 15):
        """
        Given a particular robot se2, project robot into the occupancy grid and determine if it intersects with any obstacles
        robot_width: the width of the robot in meters
        robot_height: the length of the robot in meters
        res: resolution (points per meter) of the grid that spans the robot body
        """

        robot_mask_x = np.linspace(-robot_height / 2, robot_height / 2, num=res)
        robot_mask_y = np.linspace(-robot_width / 2, robot_width / 2, num=res)
        xv, yv = np.meshgrid(robot_mask_x, robot_mask_y)

        mesh_in_base = np.transpose(np.vstack([xv.flatten(), yv.flatten()]))

        # Adding a third dimension as required by transformation matrix
        mesh_in_base = np.append(mesh_in_base, np.ones_like(xv.flatten())[..., np.newaxis], axis=1)
        trans_mat = np.array([[cos(se2[2]), -sin(se2[2]), se2[0]],
                              [sin(se2[2]), cos(se2[2]), se2[1]],
                              [0, 0, 1]])

        translated_mask = np.transpose(np.dot(trans_mat, np.transpose(mesh_in_base)))

        buckets_u, buckets_v = self.get_bucket(translated_mask[:, 0], translated_mask[:, 1])

        intersection = self.grid[buckets_u, buckets_v]

        is_intersection = np.any(intersection)

        map_img = (np.flip(self.get_grid(), 1) * 255).astype(np.uint8)
        map_img = (self.get_grid() * 255).astype(np.uint8)

        map_img_color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2RGB)

        color_channel = 0 if is_intersection else 2
        map_img_color[buckets_u, buckets_v, color_channel] = 255
        # rr.log_image("a_star/map", map_img_color)
        # rr.log_text_entry("se2", str(se2[0]) + " " + str(se2[1]) + " " + str(se2[2]))
        return is_intersection

    def add_obstacle(self, center_x, center_y, radius):
        """
        Given a center and radius, add a point obstacle as a circular mask to the occupancy grid
        """
        self.obstacles_list.append((center_x, center_y, radius))
        grid_u, grid_v = self.get_bucket(center_x, center_y)
        grid_radius = radius / self.res
        u = np.arange(0, self.dim)[:, None]
        v = np.arange(0, self.dim)[None, :]
        mask = (u - grid_u) ** 2 + (v - grid_v) ** 2 < grid_radius ** 2
        self.grid[mask] = 1

        map_img = (self.get_grid() * 255).astype(np.uint8)
        rr.log_image("map", map_img)

    def get_grid(self):
        return self.grid

    def get_scaled_dim(self):
        """
        Returns the dimension of the grid in meters instead of grid dimensions
        """
        return self.dim * self.res
