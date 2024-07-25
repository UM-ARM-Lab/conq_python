import os
import cv2
import time
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import time
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from ultralytics import SAM

#Functions for real time localization of objects
def create_gaussian_kernel(radius, sigma=1):
    """Create a 2D Gaussian kernel."""
    size = 2 * radius + 1
    x, y = np.meshgrid(np.linspace(-radius, radius, size), np.linspace(-radius, radius, size))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return g

def update_heatmap(heatmap, mask, score, kernel):
    kernel_radius = kernel.shape[0] // 2
    kernel_center = kernel_radius, kernel_radius

    for y, x in np.argwhere(mask):
        x_min = max(x - kernel_radius, 0)
        x_max = min(x + kernel_radius + 1, heatmap.shape[1])
        y_min = max(y - kernel_radius, 0)
        y_max = min(y + kernel_radius + 1, heatmap.shape[0])

        k_x_min = kernel_center[0] - (x - x_min)
        k_x_max = kernel_center[0] + (x_max - x)
        k_y_min = kernel_center[1] - (y - y_min)
        k_y_max = kernel_center[1] + (y_max - y)

        heatmap[y_min:y_max, x_min:x_max] += 0.25 * score * kernel[k_y_min:k_y_max, k_x_min:k_x_max]

def normalize_heatmap(heatmap):
    normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap_colored

#Class for fast OWpen Wolrd segmentation
class FastOwlsam:

    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to('cuda')
        self.mobilesam = SAM("/home/adibalaji/Desktop/agrobots/weights_cfgs/mobile_sam.pt").to('cuda')

    def predict_segmentation(self, image_pil, object_name):
        image = image_pil
        texts = [[f"a photo of a {object_name}"]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

        i = 0
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        max_score_index = scores.argmax()
        box = boxes[max_score_index]
        box = [int(i) for i in box.tolist()]

        mask = self.mobilesam.predict(image, bboxes=box)[0].masks.data[0].to(torch.uint8).cpu().numpy()

        return mask, scores[max_score_index]

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
