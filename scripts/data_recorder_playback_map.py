import os
from pathlib import Path

import numpy as np
import rerun as rr
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *

from conq.navigation_lib.map.map_anchored import MapAnchored

RR_TIMELINE_NAME = "stable_time"
WAYPOINT_COLOR = np.array((1.0, 1.0, 1.0), dtype=float).reshape(1, 3)
EDGE_COLOR = WAYPOINT_COLOR.copy()

# def visualize_waypoints_in_seed_frame(current_graph):
#     for


def main():
    # Note that the map has to be "anchored", meaning that the map has been ran through the
    # anchoring post-processing routines to relate all waypoints to a single frame. Otherwise, the
    # point cloud can't be extracted as it's not topologically consistent.
    # map_path = Path("/home/dcolli23/DataLocker/spot/spot_graphnav_map_of_lab/")
    map_path = Path("/home/dcolli23/DataLocker/spot/maps/collabspace_jacob/")
    # Boston Dynamics API always saves maps in a "downloaded_graph" sub-directory of where you want
    # to save the map.
    map_path = map_path / "downloaded_graph"

    map_viz = MapAnchored(map_path)

    rr.init("data_recorder_playback_map", spawn=True)
    rr.set_time_seconds(RR_TIMELINE_NAME, 0)
    map_viz.log_rerun()


if __name__ == "__main__":
    main()
