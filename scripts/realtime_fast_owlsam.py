from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client.image import build_image_request
import os
import cv2
import time

from conq.navigation.graph_nav.graph_nav_utils import GraphNav
from conq.manipulation_lib.Manipulation import open_gripper, close_gripper, move_gripper
from conq.manipulation_lib.utils import verify_estop
from conq.manipulation_lib.Manipulation import move_gripper, open_gripper, close_gripper
from conq.manipulation_lib.Perception3D import VisualPoseAcquirer, PointCloud, Vision
from conq.manipulation_lib.Grasp import get_grasp_candidates, get_best_grasp_pose

from PIL import Image
import numpy as np
from dotenv import load_dotenv
import time
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from ultralytics import SAM


from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.lease import LeaseClient
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b, VISION_FRAME_NAME

from conq.clients import Clients

ORIENTATION_MAP = {
    'back_fisheye_image':               (-1.00,0.0,0.0, 0.7071,0.,0.7071,0),
    'frontleft_fisheye_image':          (0.75,0.0,0.0, 0.7071,0.,0.7071,0),
    'frontright_fisheye_image':         (0.75,0.0,0.1, 0.7071,0.,0.7071,0),
    'left_fisheye_image':               (0,0.65,0.1, 0.924,0.0,0.0,0.383),
    'right_fisheye_image':              (0,-0.65,0.1, 0.7071,0.,0.7071,0),
    'straight_up':                      (0.5,0,0.85, 1.0,0,0,0),
    'put_down':                         (0.75,0,-0.30, 0.7071,0.,0.7071,0),
    'hand_search':                      (0.55,0.0,0.65, 0.819,0.0,0.574,0.0),
    'hand_search_forward':              (0.8,0.0,0.15, 0.819,0.0,0.574,0.0),
    'find_grasp_front':                 (0.75,0.0,0.25, 0.7071,0.,0.7071,0)
}

ROTATION_ANGLE = {
                                'back_fisheye_image':               0,
                                'frontleft_fisheye_image':          -78,
                                'frontright_fisheye_image':         -102,
                                'frontleft_depth_in_visual_frame':  -78,
                                'frontright_depth_in_visual_frame': -102,
                                'hand_depth_in_hand_color_frame':   -90,
                                'hand_depth':                       0,
                                'hand_color_image':                 0,
                                'left_fisheye_image':               0,
                                'right_fisheye_image':              180
                        }

def _rotate_image(img, angle):
        img = np.asarray(Image.fromarray(img).rotate(angle, expand=True))
        return img

def _image_to_opencv(image, auto_rotate=False):
        """Convert an image proto message to an openCV image."""
        num_channels = 1  # Assume a default of 1 byte encodings.
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                num_channels = 3
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                num_channels = 4
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                num_channels = 1
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                num_channels = 1
                dtype = np.uint16

        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into a RGB rows X cols shape.
                img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
            except ValueError:
                # Unable to reshape the image data, trying a regular decode.
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            img = img[:, :, ::-1]

        if auto_rotate:
            angle = ROTATION_ANGLE[image.source.name]
            if angle != 0:
                img = _rotate_image(img, angle)

        return img

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

sdk = bosdyn.client.create_standard_sdk('RealtimeHandCamTest')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot) 
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_client.take()
robot.time_sync.wait_for_sync()

robot.power_on(timeout_sec=20)
assert robot.is_powered_on(), 'Robot power on failed.'

verify_estop(robot)

lease_client = robot.ensure_client(LeaseClient.default_service_name)
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)
rc_client = robot.ensure_client(RayCastClient.default_service_name)
command_client = robot.ensure_client(RobotCommandClient.default_service_name)

clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client, image=image_client, raycast=rc_client, command=command_client, robot=robot)

blocking_stand(command_client, timeout_sec=10)
time.sleep(1)
move_gripper(clients, ORIENTATION_MAP['hand_search'], blocking = True, duration = 1)
time.sleep(1)
open_gripper(clients)

sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']
vision = Vision(image_client, sources)
pointcloud = PointCloud(vision)

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to('cuda')
mobilesam = SAM("/home/adibalaji/Desktop/agrobots/weights_cfgs/mobile_sam.pt").to('cuda')

texts = [["a photo of a drill"]]
heatmap = np.zeros((480, 640), dtype=np.float32)

# Create a Gaussian kernel
kernel_radius = 30  # Adjust the radius as needed
sigma = 15  # Adjust the sigma as needed
gaussian_kernel = create_gaussian_kernel(kernel_radius, sigma)
decay_factor = 0.70

while True:
    img_rgb, rgb_np = vision.get_latest_RGB(path="", save=False)

    image = Image.fromarray(img_rgb)
    inputs = processor(text=texts, images=image, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

    i = 0
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    try:
        max_score_index = scores.argmax()
        box = boxes[max_score_index]
        box = [int(i) for i in box.tolist()]
        mask = mobilesam.predict(img_rgb, bboxes=box)[0].masks.data[0].to(torch.uint8).squeeze().cpu().numpy()
        
        update_heatmap(heatmap, mask, scores[0].item(), gaussian_kernel)
        heatmap = decay_factor * heatmap

        heatmap_colored = normalize_heatmap(heatmap)

        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

        cv2.imshow("Fast OWL-SAM Heatmap", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    except Exception as e:
        print(f"Problem with OWL-ViT box output: \n{e}")

cv2.destroyAllWindows()
