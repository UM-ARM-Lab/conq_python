import requests
import base64
from openai import OpenAI
import json

class SceneLabeler:
    def __init__(self):
        self.images_loc = '/Users/adibalaji/Desktop/agrobots/conq_python/data/memory_images/'
        self.json_loc = '/Users/adibalaji/Desktop/agrobots/conq_python/data/json/spot_memory.json'

        # self.MY_API_KEY = blah blah blah
        # self.client = OpenAI(organization='<org_id>', api_key=self.MY_API_KEY)
        self.object_dict = {}

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def extract_objects(self, image_paths):
        images = [self._encode_image(self.images_loc + path) for path in image_paths]

        img_waypoint_ids = [path.split('_')[4] for path in image_paths]

        for idx, image in enumerate(images):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.MY_API_KEY}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": 
                     """
                     You are an expert tool/object identifier robot. You wil be given an image and will extract all the tools/object within that image. You will output the objects as a comma separated list and only that list. If you see no tools/objects in an image, just print None. You can ignore larger/static objects like tables/chairs/people.
                     Example:
                     If you see an image of a table with a drill, shovel, potting soil and rake, you will output: drill, shovel, potting soil, rake
                     """},
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "Identify tools/objects in this image."
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            objs_str = response.json()['choices'][0]['message']['content']
            image_objects = objs_str.split(', ')
            
            for obj_name in image_objects:
                if obj_name != 'None':
                    curr_img_waypoint_id = f"waypoint_{img_waypoint_ids[idx]}"
                    self.object_dict[obj_name] = curr_img_waypoint_id
                    
        
        return self.object_dict
    
    def identify_object_from_bank(self, text, dict):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.MY_API_KEY}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": 
                     f"""
                     Your name is spot, a tool retrieval robot. User will give you a text prompt that will contain a request for an object or task. You will use the prompt to select the best possible object from this object bank: {dict}. You print only one object exactly as it appears in the object bank.
                     Example:
                     For the prompt, Hey spot I need to dig the ground, you will print "shovel".
                     For the prompt, Hey spot fill some water, you will print "bucket".
                     For the prompt, I need to rake the soil, you will print "hand rake".
                     For the prompt, Hey spot bring me the drill, you will print "drill".
                     For the prompt, I need to clean my hands, you will print "hand sanitizer".
                     """},
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": f"User prompt: {text}"
                        },
                    ]
                    }
                ],
                "max_tokens": 300
            }

            raw_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            response = raw_response.json()['choices'][0]['message']['content']

            return response
    
    def save_dict_to_json(self, data):

        with open(self.json_loc, 'w') as json_file:
            json.dump(data, json_file, indent=4)
            print(f'Saved semantic memory to {self.json_loc}')

    def load_dict_from_json(self):

        with open(self.json_loc, 'r') as json_file:
            data = json.load(json_file)
            print(f'Loaded semantic memory from {self.json_loc}')
        
        return data