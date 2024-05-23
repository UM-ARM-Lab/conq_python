from openai import OpenAI
import base64
import requests
import os

api_key = ''
organization = ''
SECRET_PATH = './memory/secrets/secret.txt'
with open(SECRET_PATH, 'r') as file:
    file_contents = file.read().splitlines()
    organization = file_contents[0]
    api_key = file_contents[1]

client = OpenAI(
  organization=organization,
  api_key=api_key
)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
seen_images = []
memory_images_dir = '/home/john/Desktop/college/Research/conq_python/memory/memory_images/'
for file in os.listdir(memory_images_dir):
  filepath = os.path.join(memory_images_dir, file)
  b64_img = encode_image(filepath)
  seen_images.append(b64_img)


for view in seen_images:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert tool retrieval robot. Your job is to identify objects/tools of interest within an image. You will also identify the static semantic high level 'areas' you see in the image. You will be fed a sequence of 5-7 images. You will print only the names of the objects in this form: obj1, obj2, obj3...., and the names of the areas: area1, area2, area3..."},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What do you see here, be brief."
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

