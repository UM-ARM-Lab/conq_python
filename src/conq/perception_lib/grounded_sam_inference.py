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

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

class GroundedSAM:
    def __init__(self):
        load_dotenv('.env.local')
        self.DEVICE = 'cpu'
        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = os.getenv('GROUNDING_DINO_CONFIG_PATH')
        GROUNDING_DINO_CHECKPOINT_PATH = os.getenv('GROUNDING_DINO_CHECKPOINT_PATH')

        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = os.getenv('SAM_CHECKPOINT_PATH')

        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        # Building SAM Model and SAM Predictor
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam)

    def _segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=False
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def predict_segmentation(self, image_path, text):
        SOURCE_IMAGE_PATH = image_path
        CLASSES = [text]
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8


        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        print(f'Beginning {text} prediction...')

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}" 
        #     for _, _, confidence, class_id, _, _ 
        #     in detections]
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # save the annotated grounding dino image
        # cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

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
        best_box = detections.xyxy[0].reshape(1,4)
        print(f'Best box for {text}: {best_box}')

        detections.mask = self._segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=best_box
        )

        print(f'Found mask for {text} with shape {detections.mask.shape}')

        #Output numpy ndarray mask of shape (1, 480, 640)
        return detections.mask
    
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
    
    def compute_mask_centroid_hough_voting(self, mask):

        mask = torch.tensor(mask)

        # Ensure the mask is of shape (H, W)
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        
        # Compute gradients
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), sobel_x, padding=1).squeeze()
        grad_y = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), sobel_y, padding=1).squeeze()
        
        # Normalize gradients to unit vectors
        magnitude = torch.sqrt(grad_x**2 + grad_y**2) + 1e-8
        unit_grad_x = grad_x / magnitude
        unit_grad_y = grad_y / magnitude
        
        # Create an accumulator for voting
        H, W = mask.shape
        accumulator = torch.zeros((H, W), device=mask.device)
        
        # Iterate over each pixel in the mask
        for y in range(H):
            for x in range(W):
                if mask[y, x] > 0:
                    # Compute the coordinates of the voting point
                    vote_y = int(y - unit_grad_y[y, x].item())
                    vote_x = int(x - unit_grad_x[y, x].item())
                    
                    # Ensure the voting point is within bounds
                    if 0 <= vote_y < H and 0 <= vote_x < W:
                        accumulator[vote_y, vote_x] += 1
        
        # Find the maximum in the accumulator
        _, max_idx = torch.max(accumulator.view(-1), dim=0)
        centroid_y, centroid_x = divmod(max_idx.item(), W)
        
        return (centroid_x, centroid_y)

    
    def plot_image_and_mask(self, image_path, mask, text_prompt, centroid=None):
        image = cv2.imread(image_path)
        c, h, w = mask.shape

        # Reshape to (height, width, 1)
        plottable_mask = mask.reshape(h, w, c)

        _, ax = plt.subplots(1)

        ax.imshow(image)
        ax.imshow(plottable_mask, alpha=0.4)
        ax.set_title(f"'{text_prompt}'")

        if centroid is not None:
             circ = patches.Circle(centroid, radius=10)
             ax.add_patch(circ)

        plt.show()

# gds = GroundedSAM()
# img_path = "/Users/adibalaji/Desktop/agrobots/conq_python/src/conq/manipulation_lib/gpd/data/RGB/live.jpg"
# text = "red clippers"


# pred_mask = gds.predict_segmentation(img_path, text)
# pred_mask_centroid = gds.compute_mask_centroid(pred_mask)
# gds.plot_image_and_mask(image_path=img_path, mask=pred_mask, text_prompt=text, centroid=pred_mask_centroid)








