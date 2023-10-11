#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
import rerun as rr
from bosdyn.api.robot_state_pb2 import FootState

from conq.cameras_utils import image_to_opencv, RGB_SOURCES
from conq.rerun_utils import viz_common_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)

    rr.init("viz_dataset")
    rr.connect()

    args = parser.parse_args()

    # Visualize the dataset in rerun
    root = args.dataset
    dts = []

    for mode in ['train', 'val']:
        mode_path = root / mode
        for episode_path in mode_path.glob("episode_*.pkl"):
            with episode_path.open('rb') as f:
                episode = pickle.load(f)

            last_time = None
            for step in episode:
                if last_time is not None:
                    dt = step['time'] - last_time
                    dts.append(dt)

                rr.log_text_entry('time', str(step['time']))
                if step['instruction'] is not None:
                    rr.log_text_entry('instruction', step['instruction'])
                state = step['robot_state']
                snapshot = state.kinematic_state.transforms_snapshot
                gripper_action = state.manipulator_state.gripper_open_percentage / 100

                rr.log_scalar('gripper_action', gripper_action)
                viz_common_frames(snapshot)

                for src in [
                    'hand_color_image',
                    'frontleft_fisheye_image', 'frontright_fisheye_image'
                ]:
                    res = step['images'][src]
                    if res is not None:
                        img_np = image_to_opencv(res, auto_rotate=True)
                        rr.log_image(f'{src}', img_np)

                for src, res in step['images'].items():
                    if res is not None:
                        print(res.source.name, res.source.rows, res.source.cols)
                #         img_np = image_to_opencv(res)
                #         rr.log_image(f'{src}', img_np)

                last_time = step['time']

    mean_dt = np.mean(dts)
    mean_fps = 1 / mean_dt
    print(f"Average dt: {mean_dt:.4f}s, Average FPS: {mean_fps:.1f}")


if __name__ == '__main__':
    main()
