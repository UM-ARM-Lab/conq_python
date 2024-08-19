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

        self.memory = {} # stored as a dict with key object_name and value [waypoint, body_x, body_y, body_yaw]
        self.text_memory = [] # also directly store the object names as strings for easy access

    def add_object(self, object_name, waypoint_str, body_x, body_y, body_yaw):

        self.text_memory.append(object_name)
        self.memory[object_name] = [waypoint_str, body_x, body_y, body_yaw]

    def get_object_location(self, object):

        return self.memory[object] # returns [waypoint, body_x, body_y, body_yaw] of input object


    def get_average_language_logprob(self, seen_obj, target_obj): # get semantic classification confidence from LLM

        seen_obj_logprob = -100.0

        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('GPY_KEY')}"
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

        for i, seen in enumerate(self.object_language_memory):

            language_prob = Decimal(self.get_average_language_logprob(seen, target_obj)) 

            lang_score_dict[seen] = language_prob

        lang_sorted_dict = dict(sorted(lang_score_dict.items(), key=lambda item: abs(item[1] - Decimal(0.0))))
        lang_sorted_dict = {k : 100*math.exp(v) for k,v in lang_sorted_dict.items()}

        sorted_objects_in_text = [object_text for object_text, prob in list(lang_sorted_dict.items())]

        return sorted_objects_in_text # returns a list of object name strings sorted by confidence high to low for the specified target object
    

if __name__ == "__main__":
    print("SemanticMemory test")