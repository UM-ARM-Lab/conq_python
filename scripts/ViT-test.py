import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client.image import build_image_request
import cv2
# Boston dynamics
# Estop Imports
import bosdyn.client.estop
from bosdyn.client.estop import EstopClient # This imports a specific class into the script rather
from bosdyn.client.estop import estop_pb2

# Clients
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient

# Conq
from conq.clients import Clients
from conq.manipulation_lib.utils import verify_estop, get_segmask_manual, get_segmask, rotate_quaternion, unstow_arm, stow_arm, stand

# Image utils
from conq.cameras_utils import get_color_img

# Assorted
import cv2
import time
import numpy as np
import json

from conq.navigation.graph_nav.graph_nav_utils import GraphNav

from dotenv import load_dotenv
import os



# Load pre-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224"
model = ViTModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Function to get image embedding from ViT
def get_image_embedding(image_url):
    image = Image.open(image_url)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extracting [CLS] token
    return cls_embedding.squeeze(0).numpy()

# Example usage
image_url1 = '/home/john/Downloads/IMG_2931.jpg'
image_url2 = '/home/john/Downloads/IMG_2930.jpg'
image_url3 = '/home/john/Downloads/IMG_2932.jpg'

embedding1 = get_image_embedding(image_url1)
embedding2 = get_image_embedding(image_url2)
embedding3 = get_image_embedding(image_url3)

# Compute cosine similarity between the two image embeddings
cosine_sim = cosine_similarity([embedding1], [embedding2])
print(f"Cosine similarity between the two images: {cosine_sim[0][0]}")
cosine_sim = cosine_similarity([embedding2], [embedding3])
print(f"Cosine similarity between the two images: {cosine_sim[0][0]}")

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

img_sources = ['right_fisheye_image', 'left_fisheye_image', 'back_fisheye_image']
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

load_dotenv('.env.local')
MEMORY_IMAGE_PATH = os.getenv('MEMORY_IMAGE_PATH')


if __name__ == "__main__":
    # Create an instance of the boston dynamics sdk with a name
    # I think these functions are included in everything that is in the bosdyn.client package
    sdk = bosdyn.client.create_standard_sdk('semantic_memory') 
    robot = sdk.create_robot('192.168.80.3')

    # This is the function that uses the username and password env vars to verify that we should have access to spot
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()


    # This is a homebrewed function ensures a given robot is not currently e-stopped
    verify_estop(robot)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_client.take()

    robot.logger.info('Powering on robot... This may take a several seconds.')
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), 'Robot power on failed.'
    robot.logger.info('Robot powered on.')

    stand(robot, command_client)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Get right_fisheye_image
        rgb_request = build_image_request('right_fisheye_image', pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
        rgb_response= image_client.get_image([rgb_request])[0]
        rgb_np = _image_to_opencv(rgb_response, auto_rotate=True)
        image = np.array(rgb_np,dtype=np.uint8)
        right_fisheye_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(MEMORY_IMAGE_PATH + "curr1.jpg", right_fisheye_image)    
        curr_right_fisheye_image_embedding = get_image_embedding(MEMORY_IMAGE_PATH + "curr1.jpg")

        # Get left_fisheye_image
        rgb_request = build_image_request('left_fisheye_image', pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
        rgb_response= image_client.get_image([rgb_request])[0]
        rgb_np = _image_to_opencv(rgb_response, auto_rotate=True)
        image = np.array(rgb_np,dtype=np.uint8)
        left_fisheye_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(MEMORY_IMAGE_PATH + "curr2.jpg", left_fisheye_image)    
        curr_left_fisheye_image_embedding = get_image_embedding(MEMORY_IMAGE_PATH + "curr2.jpg")


        # Get back_fisheye_image
        rgb_request = build_image_request('back_fisheye_image', pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
        rgb_response= image_client.get_image([rgb_request])[0]
        rgb_np = _image_to_opencv(rgb_response, auto_rotate=True)
        image = np.array(rgb_np,dtype=np.uint8)
        back_fisheye_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(MEMORY_IMAGE_PATH + "curr3.jpg", back_fisheye_image)    
        curr_back_fisheye_image_embedding = get_image_embedding(MEMORY_IMAGE_PATH + "curr3.jpg")

        gn = GraphNav(robot)

        scores = []

        for waypoint_num in range(gn.get_graph_size()):
            print(f'Embedding waypoint_{waypoint_num}')
            # Get right_fisheye_image
            img_path_right_fisheye_image = MEMORY_IMAGE_PATH + 'right_fisheye_image' + f"_waypoint_{waypoint_num}_.jpg"
            mem_right_fisheye_image_embedding = get_image_embedding(img_path_right_fisheye_image)    
            cosine_sim_right_fisheye_image = cosine_similarity([mem_right_fisheye_image_embedding], [curr_right_fisheye_image_embedding])

            # Get left_fisheye_image
            img_path_left_fisheye_image = MEMORY_IMAGE_PATH + 'left_fisheye_image' + f"_waypoint_{waypoint_num}_.jpg"
            mem_left_fisheye_image_embedding = get_image_embedding(img_path_left_fisheye_image)   
            cosine_sim_left_fisheye_image = cosine_similarity([mem_left_fisheye_image_embedding], [curr_left_fisheye_image_embedding])

            # Get back_fisheye_image
            img_path_back_fisheye_image = MEMORY_IMAGE_PATH + 'back_fisheye_image' + f"_waypoint_{waypoint_num}_.jpg"
            mem_back_fisheye_image_embedding = get_image_embedding(img_path_back_fisheye_image)   
            cosine_sim_back_fisheye_image = cosine_similarity([mem_back_fisheye_image_embedding], [curr_back_fisheye_image_embedding])

            print(f'waypoint_{waypoint_num} has a score of {cosine_sim_right_fisheye_image + cosine_sim_left_fisheye_image + cosine_sim_back_fisheye_image}')
            scores.append(cosine_sim_right_fisheye_image + cosine_sim_left_fisheye_image + cosine_sim_back_fisheye_image)


        maxIndex = 0
        maxScore = 0
        for i in range(gn.get_graph_size()):
            if(scores[i] > maxScore):
                maxScore = scores[i]
                maxIndex = i

        gn.localize(maxIndex)

        gn.navigate_to(f'waypoint_{6}')

        print(f"We are estimated to be at waypoint {maxIndex}")

