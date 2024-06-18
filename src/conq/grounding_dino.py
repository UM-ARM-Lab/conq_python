import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dotenv import load_dotenv

# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np

# grounding dino from hugging face transformers
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class GroundingDino:
    def __init__(self):
        load_dotenv('.env.local')
        self.model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda"

        self.preprocessor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def predict_box_with_text(self, image, text):

        inputs = self.preprocessor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)   

        results = self.preprocessor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        pred_boxes = results[0]['boxes']

        print(f"Predicted box for '{text}' of shape {pred_boxes.shape}")

        return pred_boxes
    def predict_box_score_with_text(self, image_path, text):

        image = Image.open(image_path)

        inputs = self.preprocessor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)   

        results = self.preprocessor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.01,
            text_threshold=0.01,
            target_sizes=[image.size[::-1]]
        )

        pred_boxes = results[0]['boxes']
        scores = results[0]['scores']

        print("Predictions:")
        print(pred_boxes, scores)
        print()

        return pred_boxes,scores
    
    def predict_box_score_with_text_pil(self, image, text):

        inputs = self.preprocessor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)   

        results = self.preprocessor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]]
        )

        pred_boxes = results[0]['boxes']
        scores = results[0]['scores']

        print("Predictions:")
        print(pred_boxes, scores)
        print()

        return pred_boxes,scores

    
    def compute_box_centroid(self, box):
        box = box.squeeze()
        box_center = (box[0] + (box[2]-box[0])/2, box[1] + (box[3]-box[1])/2)
        return box_center

    def plot_image_and_box(self, image, box, centroid=None):

        box = box.squeeze()

        _, ax = plt.subplots(1)

        ax.imshow(image)
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=None)
        ax.add_patch(rect)

        if centroid is not None:
            circ = patches.Circle(centroid, radius=10)
            ax.add_patch(circ)

        plt.show()

    def predict_from_webcam(self):
        cap = cv2.VideoCapture("/home/adibalaji/Desktop/agrobots/conq_python/data/memory_images/IMG_2577.mov")
        while cap.isOpened():
            ret, frame = cap.read()
            image_cv2 = frame
            image = Image.fromarray(image_cv2)
            pred_boxes, scores = self.predict_box_score_with_text_pil(image, "hose attachment")

            box = pred_boxes[0]
            cv2.rectangle(image_cv2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(image_cv2, f'conf:{scores[0]}', (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("GroundingDINO detection from video/webcam", image_cv2)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gd = GroundingDino()

    gd.predict_from_webcam()

    # img = Image.open('/Users/adibalaji/Desktop/agrobots/conq_python/data/memory_images/IMG_2556.jpg')
    # gd = GroundingDino()
    # pred = gd.predict_box_score_with_text(img, "hose nozzle")
    # print(pred)
    # gd.plot_image_and_box(img, pred[0][0])
