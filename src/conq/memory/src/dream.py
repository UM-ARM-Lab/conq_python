from openai import OpenAI
import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv('.env.local')

GPT_KEY = os.getenv('GPT_KEY')
ORG_KEY = os.getenv('ORG_KEY')

client = OpenAI(
  organization=ORG_KEY,
  api_key=GPT_KEY
)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
seen_images = []
memory_images_dir = '/home/john/Desktop/college/Research/conq_python/data/memory_images/'
for file in os.listdir(memory_images_dir):
  filepath = os.path.join(memory_images_dir, file)
  b64_img = encode_image(filepath)
  seen_images.append(b64_img)


for view in seen_images:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": """
            You are an expert tool retrieval robot. Your job is to identify objects/tools of interest within an image. You will also identify the static semantic high level 'areas' you see in the image. You will be fed a sequence of 5-7 images. You will print only the names of the objects in this form: obj1, obj2, obj3...., and the names of the areas: area1, area2, area3... You must output something for each input. You may ignore bigger immovable objects like tables and doors.
            
            For example, your output should look like this:
             
            Objects: sanitizer bottle, ball, screewdriver, coffee mug
            Areas: seating area, meeting area, entrance
            
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

    print(response.json()['choices'][0]['message']['content'])
    print()

