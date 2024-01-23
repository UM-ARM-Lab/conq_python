import os
from pathlib import Path

import numpy as np
import rerun as rr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *

from conq.navigation_lib.map import get_point_cloud_data_in_seed_frame, load_map


class MplColorHelper:
    """Adapted from https://stackoverflow.com/a/26109298"""

    def __init__(self, start_val: float, stop_val: float, cmap_name: str = "plasma"):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val: np.ndarray):
        return self.scalarMap.to_rgba(val)


def main():
    # Note that the map has to be "anchored", meaning that the map has been ran through the
    # anchoring post-processing routines to relate all waypoints to a single frame. Otherwise, the
    # point cloud can't be extracted as it's not topologically consistent.
    # map_path = Path("/home/dcolli23/DataLocker/spot/spot_graphnav_map_of_lab/")
    map_path = Path("/home/dcolli23/DataLocker/spot/maps/collabspace_jacob/")
    # Boston Dynamics API always saves maps in a "downloaded_graph" sub-directory of where you want
    # to save the map.
    map_path = map_path / "downloaded_graph"

    (current_graph, current_waypoints, current_waypoint_snapshots, current_edge_snapshots,
     current_anchors, current_anchored_world_objects) = load_map(map_path)

    data = None
    for wp in current_graph.waypoints:
        cloud_data = get_point_cloud_data_in_seed_frame(current_waypoints,
                                                        current_waypoint_snapshots, current_anchors,
                                                        wp.id)

        if data is None:
            data = cloud_data
        else:
            data = np.concatenate((data, cloud_data))

    # print(data.shape)

    z_vals = data[:, 2]
    z_min = np.min(z_vals)
    z_max = np.max(z_vals)
    # cmap_vals = z_max - z_vals
    cmap_vals = z_vals
    # turbo was okay but almost too bright.
    # plasma was a nice cool colormap.
    # gnuplot is okay but pretty dark/hard to differentiate.
    color_mapper = MplColorHelper(start_val=z_min, stop_val=z_max, cmap_name="plasma")
    pc_colors = color_mapper.get_rgb(cmap_vals)

    rr.init("data_recorder_playback_map", spawn=True)
    rr.log("map_point_cloud", rr.Points3D(data, colors=pc_colors, radii=0.02))

    # TODO: Default to showing origin?


if __name__ == "__main__":
    main()
