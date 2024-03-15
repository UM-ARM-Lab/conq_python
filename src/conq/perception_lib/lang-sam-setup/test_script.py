"""
Define inputs here: 

- conq_image_source -> Image captured from Conq 
- text_prompt -> Action instruction 

To run script: 
- python3 test_script.py --conq-image-source /Users/saketpradhan/Desktop/lang-segment-anything/lang-segment-anything/image_capture.jpg --text-prompt ball
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM

# Config stuff, add here
color = (0, 255, 0)
thickness = 2


def lang_sam(conq_image_source, text_prompt):
    model = LangSAM()
    image_pil = Image.open(conq_image_source).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

    print(masks, boxes, phrases, logits)
    print(type(masks), type(boxes), type(phrases), type(logits))

    box = boxes[0].cpu().numpy()  # FIXME: Check the boxes[0] and box[0,1,2,3] conditioning here

    # Calculate centroid
    centroid_x = round((box[0] + box[2]) / 2)
    centroid_y = round((box[1] + box[3]) / 2)
    print("Centroid coordinates: ({}, {})".format(centroid_x, centroid_y))

    return centroid_x, centroid_y


def display_bounding_box(conq_image_source, box):
    """Draw the bounding box on the image"""
    image_cv2 = cv2.imread(conq_image_source)
    start_point = (int(box[0]), int(box[1]))
    end_point = (int(box[2]), int(box[3]))

    image_with_box = cv2.rectangle(image_cv2, start_point, end_point, color, thickness)

    cv2.imshow('Image with Bounding Box', image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Get the centroid coordinates')
    parser.add_argument('--conq-image-source', type=str, required=True, help='Path to the Conq image source')
    parser.add_argument('--text-prompt', type=str, required=True, help='Text prompt for LangSAM')
    args = parser.parse_args()

    conq_image_source = args.conq_image_source
    text_prompt = args.text_prompt

    centroid_x, centroid_y = lang_sam(conq_image_source, text_prompt)
    print("Centroid coordinates:", centroid_x, centroid_y)

    # display_bounding_box(conq_image_source, (0, 0, 100, 100))  


if __name__ == "__main__":
    main()