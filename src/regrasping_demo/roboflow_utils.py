import base64
import os

import cv2
import numpy as np
import requests
from PIL import Image

MIN_CONFIDENCE = 0.10
MODEL_VERSION = 18


def get_roboflow_predictions(rgb_np):
    img_str = base64.b64encode(cv2.imencode('.png', rgb_np)[1])
    base_url = f"https://detect.roboflow.com/spot-vaccuming-demo/{MODEL_VERSION}?"
    query_params = {
        "confidence": int(100 * MIN_CONFIDENCE),
        "api_key":    os.environ['ROBOFLOW_API_KEY'],

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

    predictions = get_roboflow_predictions(rgb_np)

    print(predictions)


if __name__ == "__main__":
    main()
