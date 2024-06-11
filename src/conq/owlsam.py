import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# OwlViT Detection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv


class OwlSam():

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

    def plot_boxes_to_image(image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        return image_pil, mask

    def load_owlvit(self):
        """
        Return: model, processor (for text inputs)
        """
        processor = OwlViTProcessor.from_pretrained(f"google/owlvit-large-patch14", force_download=False)
        model = OwlViTForObjectDetection.from_pretrained(f"google/owlvit-large-patch14", force_download=False)
        model.to('cpu')
        model.eval()

        print('Initialized OWL-ViT and SAM..')
        
        return model, processor
    
    def __init__(self):
        load_dotenv('.env.local')
        self.model, self.processor = self.load_owlvit()
        self.predictor = SamPredictor(build_sam(checkpoint=os.getenv('SAM_PATH')))
        self.box_threshold = 0.0
        self.device = "cpu"
        self.get_top_k = True

        

    def predict_boxes(self, image, texts):
        with torch.no_grad():
            inputs = self.processor(text=texts, images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            # target_sizes = torch.Tensor([image.size[::-1]])
            target_sizes = torch.Tensor([[640,480]])
            
            results = self.processor.post_process_object_detection(outputs=outputs, threshold=self.box_threshold, target_sizes=target_sizes.to(self.device))
            scores = torch.sigmoid(outputs.logits)
            topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)
            
            i = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[i]
            if self.get_top_k:    
                topk_idxs = topk_idxs.squeeze(1).tolist()
                topk_boxes = results[i]['boxes'][topk_idxs]
                topk_scores = topk_scores.view(len(text), -1)
                topk_labels = results[i]["labels"][topk_idxs]
                boxes, scores, labels = topk_boxes, topk_scores, topk_labels
            else:
                boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            

            # Print detected objects and rescaled box coordinates
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

            boxes = boxes.cpu().detach().numpy()
            normalized_boxes = copy.deepcopy(boxes)[0]

            obj_centroid = (normalized_boxes[2]- normalized_boxes[0]) / 2 + normalized_boxes[0], (normalized_boxes[3]- normalized_boxes[1]) / 2 + normalized_boxes[1]
            
            return normalized_boxes, obj_centroid
        
    def predict_masks(self, img_path, boxes):
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        
        # for i in range(boxes.shape[0]):
        #     boxes[i] = torch.Tensor(boxes[i])

        boxes = torch.tensor(boxes, device=self.predictor.device)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        return masks
    
    def mask_centroid(self, segmentation):
        segmentation = torch.tensor(segmentation)
        
        obj_pixel_mask = (segmentation == 1)
        obj_pixels = torch.nonzero(obj_pixel_mask, as_tuple=False)

        cents = obj_pixels.float().mean(dim=0)

        print(f'centroid output {cents}')



        return cents[2], cents[3]
        

owlsam = OwlSam()
img_path = "/home/qianwei/conq_python/data/memory/IMG_1720.jpg"
image = Image.open(img_path)
pred_box, centroid = owlsam.predict_boxes(image, [["hand rake"]])
pred_box = torch.tensor(pred_box)
# print(f'\nObject centroid @ {centroid}')
# print(f'Object bounding box {pred_box}')

owlsam.model.cpu()
del owlsam.model

masks = owlsam.predict_masks(img_path, pred_box)
print(f'Mask found of shape {masks.shape}')

plt.imshow(image)
plt.imshow(masks.squeeze(0).permute(1,2,0).numpy(), alpha=0.5)
plt.show()