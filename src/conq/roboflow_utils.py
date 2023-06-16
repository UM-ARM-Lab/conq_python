import base64
import os

import cv2
import numpy as np
import requests
from PIL import Image

MIN_CONFIDENCE = 0.10
MODEL_VERSION = 18


def get_predictions(rgb_np, use_local_inference_server=False):
    img_str = base64.b64encode(cv2.imencode('.png', rgb_np)[1])
    if use_local_inference_server:
        raise NotImplementedError("RoboFlow Pro is required for local inference")
        base_url = f"http://localhost:9001/spot-vaccuming-demo/{MODEL_VERSION}?"
    else:
        base_url = f"https://detect.roboflow.com/spot-vaccuming-demo/{MODEL_VERSION}?"
    query_params = {
        "confidence": int(100 * MIN_CONFIDENCE),
        "api_key": os.environ['ROBOFLOW_API_KEY'],

    }
    resp = requests.post(base_url,
                         params=query_params,
                         data=img_str,
                         headers={
                             "Content-Type": "application/x-www-form-urlencoded"
                         }, stream=True).json()
    predictions = resp['predictions']
    return predictions


def main():
    rgb_pil = Image.open("data/1686842789/rgb.png")
    rgb_np = np.asarray(rgb_pil)

    predictions = get_predictions(rgb_np)

    print(predictions)


if __name__ == "__main__":
    main()
