from openai import OpenAI
import base64
import requests
import os
import re
import random
from dotenv import load_dotenv

MEMORY_IMAGES_DATAPATH = './data/memory_images/'

# This is a class that memory interfaces while "dreaming"
class Dreaming: 
    # Ctor
    def __init__(self):
        # Load the .env.local file and the corresponding variables
        load_dotenv('.env.local')
        self.GPT_KEY = os.getenv('GPT_KEY')
        self.ORG_KEY = os.getenv('ORG_KEY')
        
        # Create the OpenAI object which is used to interface to the gpt api
        self.client = OpenAI(
            organization=self.ORG_KEY,
            api_key=self.GPT_KEY
        )

    #### Private Member functions

    # Function to encode the image into a format that gpt accepts
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def _parse_gpt_output(self, response):
          # Regular expression to extract objects and areas
            objects_match = re.search(r"Objects: (.*)", response)
            areas_match = re.search(r"Areas: (.*)", response)

            # Extract and split the objects and areas into lists
            if objects_match:
                obj_list = [item.strip() for item in objects_match.group(1).split(',')]
            else:
                obj_list = []

            if areas_match:
                area_list = [item.strip() for item in areas_match.group(1).split(',')]
            else:
                area_list = []

            return obj_list, area_list
    
    def _get_dream_waypoint(self):
        assert self.can_dream()
        images = os.listdir(MEMORY_IMAGES_DATAPATH)
        name  = images[0]
        start = name.find('-')
        end = name.find('.')
        return int(name[start + 1 : end])
    
    def _get_img_waypoint(self, path):
        start = path.find('-')
        end = path.find('.')
        return int(path[start + 1 : end])


    #### Public Member functions
    def can_dream(self):
        return len(os.listdir(MEMORY_IMAGES_DATAPATH)) != 0
    
    # This function populates interfaces with the GPT api to detect all of the objects in the image
    def detect_objects(self):
        # This function must be called under the assumption that there exist images to dream about in MEMORY_IMAGES_DATAPATH
        assert self.can_dream(), "There are no images to dream about"
        path = MEMORY_IMAGES_DATAPATH + os.listdir(MEMORY_IMAGES_DATAPATH)[0]
        waypoint_id = self._get_img_waypoint(path[1:])
        view = self._encode_image(path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.GPT_KEY}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": """
                You are an expert farm/garden tool retrieval robot. Your job is to identify farm/garden objects/tools of interest within an image. You will also identify the static semantic high level 'areas' you see in the image. You will be fed a sequence of 5-7 images. You will print only the names of the objects in this form: obj1, obj2, obj3...., and the names of the areas: area1, area2, area3... You must output something for each input. Bigger immovable objects like tables and doors don't count as objects and you can ignore them. You can also ignore non farm/garden related objects. If you see no objects of interest print None.
                
                For example, your output should look like this:
                
                Objects: sanitizer bottle, ball, screwdriver, coffee mug
                Areas: seating area, conference area, entrance, coffee station
                
                """},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract objects/tools and the static semantic high level areas."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{view}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        obj_list, area_list = self._parse_gpt_output(response.json()['choices'][0]['message']['content'])

        os.remove(path)

        return waypoint_id, obj_list, area_list