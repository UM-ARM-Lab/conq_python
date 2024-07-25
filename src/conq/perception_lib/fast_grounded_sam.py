import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
import os
from dotenv import load_dotenv
import torch.nn.functional as F
import time
from PIL import Image

from groundingdino.util.inference import Model
from ultralytics import SAM

class FastGroundedSAM:
    
    def __init__(self):
        load_dotenv('.env.local')

        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = os.getenv('GROUNDING_DINO_CONFIG_PATH')
        GROUNDING_DINO_CHECKPOINT_PATH = os.getenv('GROUNDING_DINO_CHECKPOINT_PATH') 
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        self.grounding_dino_model.device = 'cuda'

        self.mobilesam = SAM("/home/adibalaji/Desktop/agrobots/weights_cfgs/mobile_sam.pt").to('cuda')

    def predict_segmentation(self, image_path, text):

        CLASSES = [text]
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8

        # load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f'Beginning {text} prediction...')

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        #sort boxes by confidence to choose only best one
        sorted_indices = np.argsort(-detections.confidence)  # Sort in descending order
        detections.xyxy = detections.xyxy[sorted_indices]
        detections.confidence = detections.confidence[sorted_indices]
        detections.class_id = detections.class_id[sorted_indices]

        # choose best box
        best_box = detections.xyxy[0].reshape(4,)
        print(f'Best box for {text}: {best_box}')
        box = [int(i) for i in best_box.tolist()]

        mask = self.mobilesam.predict(image, bboxes=box)[0].masks.data[0].to(torch.bool).cpu().numpy()

        # Visualize the mask
        color_mask = np.zeros_like(image)
        color_mask[:, :, 2] = mask * 255  # Apply mask to the red channel
        overlay = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

        cv2.imshow('Overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return mask

    def compute_mask_centroid(self, mask):

        mask = torch.tensor(mask)
        
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        
        indices = mask.nonzero(as_tuple=True)
        
        if len(indices[0]) == 0:
            raise ValueError("The mask is empty")
        
        y_mean = indices[0].float().mean().item()
        x_mean = indices[1].float().mean().item()

        print(f'Found mask centroid at {(x_mean, y_mean)}')
        
        return (x_mean, y_mean)
