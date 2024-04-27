"""
Define inputs here: 

- conq_image_source -> Image captured from Conq 
- text_prompt -> Action instruction 

To run script: 
- python3 get_mask.py --conq-image-source image_capture.jpg --text-prompt volleyball
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM

# FIXME: fix image source for linux pc
# conq_image_source = "*** ADD A NEW IMAGE SOURCE HERE ***"
# text_prompt = "ball"

# Config stuff, add here
color = (0, 255, 0)
thickness = 2


def lang_sam(conq_image_source, text_prompt):
    print("Getting LangSAM inference")
    model = LangSAM()
    # image_pil = Image.open(conq_image_source).convert("RGB")
    image_pil = Image.fromarray(conq_image_source.astype('uint8'), 'RGB')
    masks, boxes, _, _  = model.predict(image_pil, text_prompt)

    box = boxes[0].cpu().numpy()  # FIXME: Check the boxes[0] and box[0,1,2,3] conditioning here

    centroid_x = round((box[0] + box[2]) / 2)
    centroid_y = round((box[1] + box[3]) / 2)
    print((centroid_x,centroid_y,type(masks)))
    return masks.squeeze().numpy(), centroid_x, centroid_y


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

    masks, centroid_x, centroid_y = lang_sam(conq_image_source, text_prompt)


if __name__ == "__main__":
    main()