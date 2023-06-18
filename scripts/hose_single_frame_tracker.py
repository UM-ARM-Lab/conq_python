#!/usr/bin/env python

from pathlib import Path

import numpy as np
from PIL import Image

from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd, METERS_TO_MILLIMETERS, \
    load_predictions, get_masks_dict
from regrasping_demo.detect_regrasp_point import DetectionError

IMAGE_ROOT = Path("data/1686855846/")


def main():
    img_path_dict = {
        "rgb": "rgb.png",
        "depth": "depth.png",
        "pred": "pred.json"
    }
    for k, filename in img_path_dict.items():
        img_path_dict[k] = IMAGE_ROOT / filename
        assert (img_path_dict[k].exists())

    rgb_pil = Image.open(str(img_path_dict["rgb"]))
    rgb_np = np.array(rgb_pil)

    # Not using the depth image as the depth horizontal FOV is too tight. Using single depth value method instead.
    depth_img = np.ones((rgb_np.shape[0], rgb_np.shape[1]), dtype=float) * METERS_TO_MILLIMETERS

    preds_dict = load_predictions(img_path_dict["pred"])

    masks_dict = get_masks_dict(preds_dict)

    # Run The full pipeline on this one example

    saved_fig_name = IMAGE_ROOT / "cdcpd_output.png"
    vertex_uv_coords = single_frame_planar_cdcpd(rgb_np, depth_img)

    # Run full pipeline on all data
    data_dir = Path("data/")
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue
        img_path_dict = {
            "rgb": "rgb.png",
            "depth": "depth.png",
            "pred": "pred.json"
        }
        all_found = True
        for k, filename in img_path_dict.items():
            img_path_dict[k] = subdir / filename
            if not img_path_dict[k].exists():
                all_found = False
                break
        if not all_found:
            print("skipping ", subdir)
            continue

        rgb_np = np.array(Image.open(img_path_dict["rgb"].as_posix()))

        # Not using the depth image as the depth horizontal FOV is too tight. Using single depth value method instead.
        depth_img = np.ones((rgb_np.shape[0], rgb_np.shape[1]), dtype=float) * METERS_TO_MILLIMETERS

        preds_dict = load_predictions(img_path_dict["pred"])

        try:
            masks_dict = get_masks_dict(preds_dict)
        except DetectionError:
            print("failed on ", subdir)
            continue

        saved_fig_name = subdir / "cdcpd_output.png"
        ordered_hose_points = single_frame_planar_cdcpd(rgb_np, depth_img)


if __name__ == "__main__":
    main()
