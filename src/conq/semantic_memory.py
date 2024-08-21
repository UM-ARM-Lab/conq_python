from openai import OpenAI
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import clip
import math
import base64
import time
import json
from decimal import Decimal, getcontext
from dotenv import load_dotenv
import os 
import cv2
import numpy as np

from conq.perception_lib.custom_yolo import YOLOFarm

def update_chatgpt_log(input_tokens, output_tokens):
    
    # file_path = "/Users/adibalaji/Desktop/agrobots/playground/chatgpt_calls.json"
    file_path = "/home/adibalaji/Desktop/agrobots/chatgpt_calls.json"
    
    with open(file_path, 'r') as file:
            data = json.load(file)
    
    # Increment the number of calls and add the token count
    data["calls"] += 1
    data["input_tokens"] += input_tokens
    data["output_tokens"] += output_tokens
    data["spent_dollars"] = (data["input_tokens"]*5.0*1e-6) + (data["output_tokens"]*15.0*1e-6)
    
    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file)

class SemanticMemory:

    def __init__(self):
        
        load_dotenv('.env.local')

        self.images_loc = os.getenv('MEMORY_IMAGE_PATH')
        self.depth_loc = os.getenv('MEMORY_DEPTH_IMAGE_PATH')
        self.memory_loc = os.getenv('OBJECT_MEMORY_JSON_PATH')

        self.memory = {} # stored as a dict with key object_name and value [waypoint, body_x, body_y, body_yaw]
        file_path = "/home/adibalaji/Desktop/agrobots/conq_python/data/json/spot_object_memory.json"
        with open(file_path, 'r') as file:
                self.memory = json.load(file)

        self.handcam_K = np.array([
            [552.02910122, 0.0, 320],
            [0.0, 552.02910122, 240],
            [0.0, 0.0, 1.0]
            ])
        
        self.handcam_K_inv = np.linalg.inv(self.handcam_K)



    def add_object(self, object_name, waypoint_str, body_x, body_y, body_yaw):

        self.text_memory.append(object_name)
        self.memory[object_name] = [waypoint_str, body_x, body_y, body_yaw]

    def get_object_location(self, object):

        return self.memory[object] # returns [waypoint, body_x, body_y, body_yaw] of input object


    def get_average_language_logprob(self, seen_obj, target_obj): # get semantic classification confidence from LLM

        seen_obj_logprob = -100.0

        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('GPT_KEY')}"
            }

        payload = {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": 
                            """
                            You are an expert object location reasoning agent. You will be given some seen objects and a target object. You need to output which which is the best seen object to go to in order to find the target object. You can ONLY use the seen objects for reasoning. Answer in short with a short explanantion and do not exceed 25 words.

                            Example:
                            User prompt: "I see the following: tool box. Where should I go to find box cutter?"
                            Your response: "Go to the tool box to find the box cutter. Tool boxes usually contain tools like box cutter."
                            """},

                            {
                            "role": "user",
                            "content": [
                                {
                                "type": "text",
                                "text": f"I see the following things: {seen_obj}.  Where should I go to find the {target_obj}?"
                                }
                            ]
                            }
                        ],
                        "logprobs": True,
                        "top_logprobs": 0,
                        "max_tokens": 300,
                        "seed": 48103,
                        "temperature": 0.01,
                    }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()

        input_token_count = response["usage"]["prompt_tokens"]
        output_token_count = response["usage"]["completion_tokens"]
        update_chatgpt_log(input_tokens=input_token_count, output_tokens=output_token_count)

        response = response["choices"][0]

        token_logprobs = []

        for chunk in response["logprobs"]["content"]:

            # if chunk["token"] == f" {target_obj}":
            #     seen_obj_logprob = chunk["logprob"]
            #     break

            token_logprobs.append(chunk["logprob"])
        
        seen_obj_logprob = sum(token_logprobs)/len(token_logprobs)

        return seen_obj_logprob
    
    def construct_semantic_search_ranking(self, target_obj):

        lang_score_dict = {}

        for seen, _ in self.memory.items():

            language_prob = Decimal(self.get_average_language_logprob(seen, target_obj)) 

            lang_score_dict[seen] = language_prob

        lang_sorted_dict = dict(sorted(lang_score_dict.items(), key=lambda item: abs(item[1] - Decimal(0.0))))
        lang_sorted_dict = {k : 100*math.exp(v) for k,v in lang_sorted_dict.items()}

        sorted_objects_in_text = [object_text for object_text, prob in list(lang_sorted_dict.items())]

        return sorted_objects_in_text # returns a list of object name strings sorted by confidence high to low for the specified target object
    
    def se2_cam_to_body(self, se2, view):

        def get_2d_rotation_matrix(theta):
            return np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        
        body_x, body_y, body_yaw = None, None, None
        
        # Unpack the SE2 pose
        cam_x, cam_y, cam_yaw = se2
        cam_xy = np.array([cam_x, cam_y])
        
        # Apply transformations based on the view
        if view == "l":
            body_xy = np.matmul(get_2d_rotation_matrix(math.radians(-90)), cam_xy)
            body_xy = body_xy + np.array([0.4, 0.35])

            body_x, body_y = body_xy[0], body_xy[1]
            body_yaw = 90 + cam_yaw
            
        elif view == "cl":
            body_xy = np.matmul(get_2d_rotation_matrix(math.radians(-45)), cam_xy)
            body_xy = body_xy + np.array([0.5, 0.15])

            body_x, body_y = body_xy[0], body_xy[1]
            body_yaw = 45 + cam_yaw

        elif view == "c":
            body_xy = np.matmul(get_2d_rotation_matrix(math.radians(0)), cam_xy) #no rotation
            body_xy = body_xy + np.array([0.55, 0.0])

            body_x, body_y = body_xy[0], body_xy[1]
            body_yaw = 0 + cam_yaw

        elif view == "cr":
            body_xy = np.matmul(get_2d_rotation_matrix(math.radians(45)), cam_xy)
            body_xy = body_xy + np.array([0.5, -0.15])

            body_x, body_y = body_xy[0], body_xy[1]
            body_yaw = -45 + cam_yaw
        elif view == "r":
            body_xy = np.matmul(get_2d_rotation_matrix(math.radians(90)), cam_xy)
            body_xy = body_xy + np.array([0.4, -0.35])

            body_x, body_y = body_xy[0], body_xy[1]
            body_yaw = -90 + cam_yaw

        # Return the transformed SE2 pose
        return [body_x, body_y, body_yaw]


    
    def dream(self):

        yolo_farm = YOLOFarm(model_path=os.getenv('YOLO_FARM'), device='cuda')
        print("Loaded YOLOFarm. Begin dreaming...\n")

        self.images_loc = '/home/adibalaji/Desktop/agrobots/conq_python/data/memory_images/'
        for img_path in os.listdir(self.images_loc):

            start_index = img_path.find("waypoint_")
            end_index = img_path.find("_", start_index + len("waypoint_"))
            waypoint_str = img_path[start_index:end_index]
            viewpoint_str = img_path.split("/")[-1].split("_")[3]

            objects = yolo_farm.get_object_centroids(image_path=f'{self.images_loc}/{img_path}', confidence_thresh=0.2)

            for obj_item in objects:

                obj, cam_x, cam_y = obj_item

                #Calculate object SE2 Pose from centroid, depth and intrinsics
                se2 = [0, 0, 0]
                
                current_depth = np.load(f"{self.depth_loc}/{img_path.split('.')[0]}depth.npy")
                cam_z = current_depth[cam_y, cam_x]

                cam_homogenous = np.array([cam_x, cam_y, 1])
                world_homogenous = np.matmul(self.handcam_K_inv, cam_homogenous)
                world_homogenous[2] = cam_z * 0.001 # set z from depth to make true depth and convert mm to m
                world_pose_cam = world_homogenous

                se2_cam = [
                            world_pose_cam[2], 
                            world_pose_cam[1], 
                            math.degrees(math.atan2((cam_x - self.handcam_K[0,2]), self.handcam_K[0,0]))
                           ]
                
                # !!!!!!!!!!! Needs work !!!!!!!!!!!!!!!!!
                se2 = self.se2_cam_to_body(se2_cam, viewpoint_str)

                waypoint_and_se2 = [waypoint_str, se2[0], se2[1], se2[2]]

                #Add to memory
                if obj not in self.memory:
                    self.memory[obj] = waypoint_and_se2
                    print(f'se2 cam: {se2_cam}')
                    print(f'Added {obj} at body frame pose {waypoint_and_se2}..\n')

        with open(self.memory_loc, 'w') as memory_json_file:
            json.dump(self.memory, memory_json_file, indent=4)

        print(f"Written json memory to {self.memory_loc}. All done!")    

if __name__ == "__main__":

    semantic_memory = SemanticMemory()
    semantic_memory.dream()