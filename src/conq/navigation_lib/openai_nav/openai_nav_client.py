import ast
import base64
import os
from io import BytesIO
from typing import List, Union

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from ratelimit import limits

from conq.api_keys import OPENAI_API_KEY

# Change this based on which API you are using
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.get()

ONE_MINUTE = 60  # Constant for seconds in a minute


class OpenAINavClient:
    def __init__(self, locations: List[str]):
        # Sets the current working directory to be the same as the file.
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        try:
            if load_dotenv(".env") is False:
                raise TypeError
        except TypeError:
            print("Unable to load .env file.")
            quit()

        self.llm_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )

        # Locations in our environment
        self.locations = locations

        # System context to give to the LLM before asking question
        self.system_context = "Assistant is a large language model trained by OpenAI."

    @limits(calls=3, period=ONE_MINUTE)
    def find_probable_location_for_object(self, object: str):
        # Build the query string
        query_list = [
            "I observe the following structures while exploring a small-scale farm:"
        ]
        query_list.extend(
            [
                f"{index + 1}. {location}"
                for index, location in enumerate(self.locations)
            ]
        )
        query_list.extend(
            [
                f"\nPlease rank the structure based on how likely I am to find a {object} in them. Please "
                "provide the response in a plain text string representing a python dictionary, "
                f'{{"name": probability}} and only include the location with whitespace connected with _ as '
                "the keys and probabilities as the values"
            ]
        )
        query = "\n".join(query_list)

        # Make the query
        response = self.llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_context},
                {"role": "user", "content": query},
            ],
            model=os.environ["model"],
            temperature=0,  # temperature set to 0 to avoid overly verbose responses
        )

        try:
            probabilities_out = ast.literal_eval(response.choices[0].message.content)
        except:
            # Try to send the bad output back through chatGPT
            num_tries = 0
            for i in range(3):
                num_tries += 1
                query = f'This output is not in a plain text string representing a python dictionary, please make it formatted like {{"name": probability}}: {response.choices[0].message.content}'
                response = self.llm_client.chat.completions.create(
                    model=os.environ["model"],
                    messages=[
                        {"role": "system", "content": self.system_context},
                        {"role": "user", "content": query},
                    ],
                    temperature=0,  # temperature set to 0 to avoid overly verbose responses
                )

                # Now try formatting
                try:
                    probabilities_out = ast.literal_eval(
                        response.choices[0].message.content
                    )
                except:
                    probabilities_out = None
                    continue

            if num_tries == 3:
                print(
                    "Unable to get a properly formatted string from ChatGPT, probabilities_out is set to None"
                )

        return probabilities_out

    @limits(calls=6, period=ONE_MINUTE)
    def generate_label_for_image(
        self,
        prompt='Please describe the primary function of the following group of tools in a concise manner using no more than five words and avoid general terms like "essential" or "gardening".',
        image="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT354l_oPc87wCqcXwlLKCRPLfZUP-Cs3Wbzw&usqp=CAU",
    ):
        response = None
        image_url = image

        if type(image) is not str:
            # Create call with actual image rgb data
            base64_image = encode_image(image)
            image_url = f"data:image/png;base64,{base64_image}"

        response = self.llm_client.chat.completions.create(
            model=os.environ["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )
        print(response.choices[0].message.content)


def encode_image_array(image_np):
    image_pil = Image.fromarray(image_np)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_string


def encode_image(image: Union[str, np.ndarray]) -> str:
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, np.ndarray):
        return encode_image_array(image)
    else:
        raise TypeError("Invalid image type. Expected str or np.ndarray.")
