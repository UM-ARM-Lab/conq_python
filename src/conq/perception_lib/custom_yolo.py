from ultralytics import YOLO
import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv('.env.local')

class YOLOFarm:
    def __init__(self, model_path, device='cpu'):

        self.model = YOLO(model_path)
        self.device = device

    def load_image(self, image_path):

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")
        return image

    def preprocess_image(self, image, img_size=640):

        img = cv2.resize(image, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        img = np.ascontiguousarray(img)
        return img

    def run_inference(self, image_path):

        image = self.load_image(image_path)
        result = self.model(image, device=self.device)
        return result

    def draw_boxes(self, image, result):

        for i, box in enumerate(result[0].boxes.xyxy):
            x1, y1, x2, y2 = box
            conf = result[0].boxes.conf[i]
            cls = result[0].boxes.cls[i]

            label = f'{self.model.names[int(cls)]}: {conf:.2f}'
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('yolo output', image)
        cv2.waitKey(0)

        return image
    
    def get_object_centroids(self, image_path, results = None):

        image_objects = []

        if results is None:
            result = self.run_inference(image_path=image_path)

        for i, box in enumerate(result[0].boxes.xyxy):

            x1, y1, x2, y2 = box
            cls = result[0].boxes.cls[i]
            conf = result[0].boxes.conf[i]

            if conf < 0.6:
                continue

            centroid = int(x1 + (x2 - x1)/2), int(y1 + (y2 - y1)/2)

            obj_item = {f'{self.model.names[int(cls)]}' : centroid}

            image_objects.append(obj_item)

        return image_objects



if __name__ == "__main__":
    img_path = "/home/adibalaji/Desktop/agrobots/conq_python/data/memory_images/live_hand.jpg"
    yolo_infer = YOLOFarm(model_path=os.getenv('YOLO_FARM'), device='cuda')

    results = yolo_infer.run_inference(img_path)
    image_with_boxes = yolo_infer.draw_boxes(cv2.imread(img_path), results)

    # print(yolo_infer.get_object_centroids(img_path))
