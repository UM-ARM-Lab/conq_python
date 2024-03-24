"""
To run this script:
$ python3 sam_inference.py --input-image https://your_image_url.jpg
"""

import os
import time
from io import BytesIO
import argparse
import requests
import cv2
import numpy as np
import replicate

from conq.api_keys import REPLICATE_API_KEY

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_KEY.get()


def run_sam_inference(image_url):
    start_time = time.time()
    image_url = replicate.run(
        "pablodawson/segment-anything-automatic:14fbb04535964b3d0c7fad03bb4ed272130f15b956cbedb7b2f20b5b8a2dbaa0",
        input={
            "image": image_url
        }
    )
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution Time: {execution_time} seconds")
    print(image_url)

    return image_url, execution_time

def view_segmentation_mask(image_url):
    response = requests.get(image_url)
    image_data = BytesIO(response.content)

    segmentation_mask = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    cv2.imshow("Segmentation Mask", segmentation_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM Inference with Replicate API")
    parser.add_argument("--input-image", required=True, help="URL of the input image for SAM Inference")

    args = parser.parse_args()

    image_url = args.input_image
    run_sam_inference(image_url)
