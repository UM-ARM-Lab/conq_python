#!/usr/bin/env python

from pathlib import Path

import numpy as np
from PIL import Image

from regrasping_demo.cdcpd_hose_state_predictor import do_single_frame_cdcpd_prediction, METERS_TO_MILLIMETERS, load_predictions, \
    get_masks_and_polygons
from regrasping_demo.detect_regrasp_point import DetectionError

IMAGE_ROOT = Path("data/1686845198/")


def main():
    img_path_dict = {
        "rgb": "rgb.png",
        "depth": "depth.png",
        "pred": "pred.json"
    }
    for k, filename in img_path_dict.items():
        img_path_dict[k] = IMAGE_ROOT / filename
        assert (img_path_dict[k].exists())

    bgr_img = np.array(Image.open(img_path_dict["rgb"].as_posix()))
    print("BGR Image shape:", bgr_img.shape)
    print("BGR Image dtype:", bgr_img.dtype)

    # Not using the depth image as the depth horizontal FOV is too tight. Using single depth value method instead.
    depth_img = np.ones((bgr_img.shape[0], bgr_img.shape[1]), dtype=float) * METERS_TO_MILLIMETERS

    preds_dict = load_predictions(img_path_dict["pred"])

    masks, polygons = get_masks_and_polygons(preds_dict, bgr_img)

    # Run The full pipeline on this one example

    saved_fig_name = IMAGE_ROOT / "cdcpd_output.png"
    vertex_uv_coords = do_single_frame_cdcpd_prediction(bgr_img, depth_img, masks, do_visualization=True,
                                                        saved_fig_name=saved_fig_name)

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

        bgr_img = np.array(Image.open(img_path_dict["rgb"].as_posix()))

        # Not using the depth image as the depth horizontal FOV is too tight. Using single depth value method instead.
        depth_img = np.ones((bgr_img.shape[0], bgr_img.shape[1]), dtype=float) * METERS_TO_MILLIMETERS

        preds_dict = load_predictions(img_path_dict["pred"])

        try:
            masks, polygons = get_masks_and_polygons(preds_dict, bgr_img, verbose=False)
        except DetectionError:
            print("failed on ", subdir)
            continue

        saved_fig_name = subdir / "cdcpd_output.png"
        ordered_hose_points = do_single_frame_cdcpd_prediction(bgr_img, depth_img, masks, do_visualization=True,
                                                               saved_fig_name=saved_fig_name)


if __name__ == "__main__":
    main()
