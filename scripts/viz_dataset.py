#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
import rerun as rr

from conq.cameras_utils import image_to_opencv
from conq.rerun_utils import viz_common_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)

    rr.init("viz_dataset")
    rr.connect()

    args = parser.parse_args()

    # Visualize the dataset in rerun
    root = args.dataset
    dataset_pkl_path = root / 'dataset.pkl'

    with dataset_pkl_path.open('rb') as f:
        dataset = pickle.load(f)

    last_time = None
    dts = []
    for step in dataset:
        if last_time is not None:
            dt = step['time'] - last_time
            dts.append(dt)

        rr.log_text_entry('time', str(step['time']))
        if step['instruction'] is not None:
            rr.log_text_entry('instruction', step['instruction'])
        snapshot = step['robot_state'].kinematic_state.transforms_snapshot
        viz_common_frames(snapshot)

        res = step['images']['hand_color_image']
        if res is not None:
            img_np = image_to_opencv(res)
            rr.log_image(f'hand_color_image', img_np)

        # for src, res in step['images'].items():
        #     if res is not None:
        #         img_np = image_to_opencv(res)
        #         rr.log_image(f'{src}', img_np)

        last_time = step['time']

    mean_dt = np.mean(dts)
    mean_fps = 1 / mean_dt
    print(f"Average dt: {mean_dt:.4f}s, Average FPS: {mean_fps:.1f}")


if __name__ == '__main__':
    main()
