
"""Replicate API Token for AgroBots: r8_YXiQPhP3rn6Eh4IAeriv1ZLY7MwCnR30s0rFA

Terminal -> Run below command
$ export REPLICATE_API_TOKEN=r8_YXiQPhP3rn6Eh4IAeriv1ZLY7MwCnR30s0rFA"""

import os
import time
from io import BytesIO

import requests
import cv2
import numpy as np
import replicate


os.environ['REPLICATE_API_TOKEN'] = 'r8_YXiQPhP3rn6Eh4IAeriv1ZLY7MwCnR30s0rFA'

def run_sam_inference():
    start_time = time.time()
    image_url = replicate.run(
        "pablodawson/segment-anything-automatic:14fbb04535964b3d0c7fad03bb4ed272130f15b956cbedb7b2f20b5b8a2dbaa0",
        input={
            "image": "https://upload.wikimedia.org/wikipedia/commons/8/8c/Cristiano_Ronaldo_2018.jpg"
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

