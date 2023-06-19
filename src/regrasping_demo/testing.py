from pathlib import Path

import numpy as np
from PIL import Image

from arm_segmentation.predictor import Predictor


def get_test_examples():
    np.seterr(all='raise')
    np.set_printoptions(precision=2, suppress=True)

    predictor = Predictor()

    data_dir = Path("homotopy_test_data/")
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue
        img_path_dict = get_filenames(subdir)
        if not img_path_dict:
            print("skipping ", subdir)
            continue

        rgb_pil = Image.open(img_path_dict["rgb"])
        rgb_np = np.asarray(rgb_pil)

        predictions = predictor.predict(rgb_np)

        yield predictor, subdir, rgb_np, predictions


def get_filenames(subdir):
    img_path_dict = {
        "rgb":   "rgb.png",
        "depth": "depth.png",
        "pred":  "pred.json"
    }
    for k, filename in img_path_dict.items():
        img_path_dict[k] = subdir / filename
        if not img_path_dict[k].exists():
            return None
    return img_path_dict
