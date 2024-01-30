from pathlib import Path

import numpy as np
import rerun as rr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *

from conq.navigation_lib.map import load_map, extract_full_point_cloud_in_seed_frame

RR_TIMELINE_NAME = "stable_time"
WAYPOINT_COLOR = np.array((1.0, 1.0, 1.0), dtype=float).reshape(1, 3)
EDGE_COLOR = WAYPOINT_COLOR.copy()


class MplColorHelper:
    """Adapted from https://stackoverflow.com/a/26109298"""

    def __init__(self, start_val: float, stop_val: float, cmap_name: str = "plasma"):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val: np.ndarray):
        return self.scalarMap.to_rgba(val)


class MapAnchored:

    def __init__(self, map_directory: Path):
        self.map_directory: Path = map_directory
        self._verify_map_path()
        self._map_name: str = "map_anchored"

        # TODO: Make (free) factory function that chooses the correct derived class to construct
        # depending on if the map is anchored or not.
        # - This effects if we can extract the full point cloud from the graph in the seed frame or
        #   if we visualize the point cloud based on relative transforms of the waypoints.
        (current_graph, current_waypoints, current_waypoint_snapshots, current_edge_snapshots,
         current_anchors, current_anchored_world_objects) = load_map(self.map_directory)
        self.current_graph: map_pb2.Graph = current_graph
        self.current_waypoints = current_waypoints
        self.current_waypoint_snapshots = current_waypoint_snapshots
        self.current_edge_snapshots = current_edge_snapshots
        self.current_anchors = current_anchors
        self.current_anchored_world_objects = current_anchored_world_objects

        self._cloud_in_seed_frame: np.ndarray = self._get_cloud_in_seed_frame()
        self._waypoint_cloud_in_seed_frame: np.ndarray = self._get_waypoint_cloud_in_seed_frame()
        # I'm not satisfied with the current data structure for storing edges so I'm going to just
        # extract them when the plotting function is called, at least for now.

    def _verify_map_path(self):
        if not self.map_directory.exists():
            raise FileNotFoundError(f"Map directory not found at: {self.map_directory}")
        # graph_dir = self.map_directory / "downloaded_graph"
        # if not graph_dir.exists():
        #     raise FileNotFoundError(f"Graph directory not found at: {graph_dir}")

    def _get_cloud_in_seed_frame(self) -> np.ndarray:
        """Get the map point cloud in the seed frame

        NOTE: Abstracted this functionality as it will be different for non-anchored graphs.
        """
        return extract_full_point_cloud_in_seed_frame(self.current_graph, self.current_waypoints,
                                                      self.current_waypoint_snapshots,
                                                      self.current_anchors)

    def _get_waypoint_cloud_in_seed_frame(self) -> np.ndarray:
        """Returns cloud of waypoint positions in seed frame"""
        # NOTE: The breadth-first search is for non-anchored graphs.
        # while len(queue) > 0:
        #     curr_element = queue.pop(0)
        #     curr_waypoint = curr_element[0]
        #     if curr_waypoint.id in visited:
        #         continue
        #     visited[curr_waypoint.id] = True

        #     # We now know the global pose of this waypoint, so set the pose.

        waypoint_cloud = []
        for waypoint in self.current_graph.waypoints:
            if waypoint.id in self.current_anchors:
                # waypoint_objects[waypoint.id] = create_waypoint_object(current_waypoints,
                # current_waypoint_snapshots, waypoint.id)
                # Just store the location in seed frame for this waypoint.
                # waypoint_objects[waypoint.id]

                seed_tform_waypoint = SE3Pose.from_proto(
                    self.current_anchors[waypoint.id].seed_tform_waypoint).to_matrix()
                waypoint_pos_in_seed_frame = seed_tform_waypoint[:3, 3]
                waypoint_cloud.append(waypoint_pos_in_seed_frame)
        return np.stack(waypoint_cloud, axis=0)

    def _log_edges(self):
        edge_root = self._map_name + "/edges/"
        edge_num = 0
        for edge in self.current_graph.edges:
            if ((edge.id.from_waypoint in self.current_anchors)
                    and (edge.id.to_waypoint in self.current_anchors)):
                seed_tform_from = SE3Pose.from_proto(
                    self.current_anchors[edge.id.from_waypoint].seed_tform_waypoint).to_matrix()
                from_tform_to = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
                world_tform_to_wp = np.dot(seed_tform_from, from_tform_to)

                # Make line between the current waypoint and the neighbor.
                pos_start = seed_tform_from[:3, 3]
                pos_end = world_tform_to_wp[:3, 3]
                line = np.stack((pos_start, pos_end), axis=0)
                name = edge_root + str(edge_num)
                rr.log(name, rr.LineStrips3D(line, colors=EDGE_COLOR))
                edge_num += 1

    def _get_pc_color(self) -> np.ndarray:
        z_vals = self._cloud_in_seed_frame[:, 2]
        z_min = np.min(z_vals)
        z_max = np.max(z_vals)
        # cmap_vals = z_max - z_vals
        cmap_vals = z_vals
        # turbo was okay but almost too bright.
        # plasma was a nice cool colormap.
        # gnuplot is okay but pretty dark/hard to differentiate.
        color_mapper = MplColorHelper(start_val=z_min, stop_val=z_max, cmap_name="plasma")
        pc_colors = color_mapper.get_rgb(cmap_vals)
        return pc_colors

    def _log_map_cloud(self, cloud_point_radii_meters: float = 0.02):
        cloud_colors = self._get_pc_color()
        rr.log(
            self._map_name + "/point_cloud",
            rr.Points3D(self._cloud_in_seed_frame,
                        colors=cloud_colors,
                        radii=cloud_point_radii_meters))

    def _log_waypoints(self,
                       waypoint_radii_meters: float = 0.1,
                       waypoint_color: np.ndarray = WAYPOINT_COLOR):
        num_points = self._waypoint_cloud_in_seed_frame.shape[0]
        waypoint_colors = waypoint_color.repeat(num_points, 0)
        rr.log(
            self._map_name + "/waypoints",
            rr.Points3D(self._waypoint_cloud_in_seed_frame,
                        colors=waypoint_colors,
                        radii=waypoint_radii_meters))

    def log_rerun(self, cloud_point_radii_meters: float = 0.02, waypoint_radii_meters: float = 0.1):
        """Logs the full map in rerun"""
        self._log_map_cloud(cloud_point_radii_meters)
        self._log_edges()
        self._log_waypoints(waypoint_radii_meters)
