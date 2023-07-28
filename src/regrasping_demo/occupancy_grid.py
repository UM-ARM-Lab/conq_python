import numpy as np
from math import cos, sin

class OccupancyGrid():
    def __init__(self, dim: int = 300, res: float = 0.05) -> None:
        '''
        The occupancy grid covers (dim * res)^2 meters, centered at (0,0)
        dim must be divisible by 2
        '''
        assert (dim % 2 == 0), "Occupancy grid dimensions must be divisible by 2."
        self.dim = dim
        self.res = res
        self.grid = np.zeros([dim, dim])

    def get_bucket(self, x: float, y: float):
        '''
        Given a coordinate in the frame of the occupancy grid, return the corresponding bucket's grid coordinates and occupancy        
        '''
        v = -1 * np.ceil(x / self.res) + self.dim / 2
        u = self.dim / 2 + np.floor(y / self.res)
        # TODO: double check these boundary conditions are right
        assert np.any((0 <= u) & (u < self.dim)), "Assertion Error: Calculated bucket would be outside grid dimensions"
        assert np.any((0 <= v) & (v < self.dim)), "Assertion Error: Calculated bucket would be outside grid dimensions"
        return u.astype(int), v.astype(int)

    def get_robot_intersection(self, se2, robot_width: float = 0.5, robot_height: float = 1, res: int=3):
        '''
        Given a particular robot se2, project robot into the occupancy grid and determine if it intersects with any obstacles
        robot_width: the width of the robot in meters
        robot_height: the length of the robot in meters
        res: resolution (points per meter) of the grid that spans the robot body
        '''
        print(se2)
        if ( 0.45 < se2[1] < 0.75):
            print("break")
        robot_mask_x = np.linspace(-robot_width / 2, robot_width / 2, num=res)
        robot_mask_y = np.linspace(-robot_height / 2, robot_height / 2, num=res)
        xv, yv = np.meshgrid(robot_mask_x,robot_mask_y)
        xv, yv = np.meshgrid(robot_mask_x, robot_mask_y)
 
        mesh_in_base = np.transpose(np.vstack([xv.flatten(), yv.flatten()]))

        # Adding a third dimension as required by transformation matrix
        mesh_in_base = np.append(mesh_in_base, np.ones_like(xv.flatten())[..., np.newaxis], axis=1)

        trans_mat = np.array([[cos(se2[2]), -sin(se2[2]), se2[1]],
                              [sin(se2[2]), cos(se2[2]), se2[0]],
                              [0, 0, 1]])
        

        translated_mask = np.transpose(np.dot(trans_mat, np.transpose(mesh_in_base)))
        buckets_x, buckets_y = self.get_bucket(translated_mask[:,0], translated_mask[:,1])

        intersection = self.grid[buckets_x, buckets_y]
        return np.any(intersection)


    def add_obstacle(self, center_x, center_y, radius):
        '''
        Given an center and radius, add a point obstacle as a circular mask to the occupancy grid
        '''
        grid_x, grid_y = self.get_bucket(center_x, center_y)
        grid_radius = radius / self.res
        x = np.arange(0, self.dim)
        y = np.arange(0, self.dim)
        mask = (x[np.newaxis,:] - grid_x)**2 + (y[:,np.newaxis] - grid_y)**2 < grid_radius**2
        self.grid[mask] = 1

    def get_grid(self):
        return self.grid
    
    def get_scaled_dim(self):
        '''
        Returns the dimension of the grid in meters instead of grid dimensions
        '''
        return self.dim * self.res