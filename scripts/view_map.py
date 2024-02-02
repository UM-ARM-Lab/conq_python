#!/usr/bin/env python3
from pathlib import Path

import rerun as rr

from conq.navigation_lib.map.map_anchored import MapAnchored


def main(map_path: Path):
    rr.init("map_viewer", spawn=True)
    map_viz = MapAnchored(map_path)
    map_viz.log_rerun()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Demo of viewing a Conq map.")
    parser.add_argument("map_path", type=Path, help="Path to GraphNav map directory")
    args = parser.parse_args()

    main(args.map_path)
