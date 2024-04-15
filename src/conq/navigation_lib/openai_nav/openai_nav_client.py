import ast
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
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
        query_list = ["I observe the following structures while exploring a small-scale farm:"]
        query_list.extend([f"{index + 1}. {location}" for index, location in enumerate(self.locations)])
        query_list.extend([
            f"\nPlease rank the structure based on how likely I am to find a {object} in them. Please "
            "provide the response in a plain text string representing a python dictionary, "
            f"{{\"name\": probability}} and only include the location with whitespace connected with _ as "
            "the keys and probabilities as the values"
        ])
        query = "\n".join(query_list)

        # Make the query
        response = self.llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_context},
                {"role": "user", "content": query},
            ],
            model=os.environ["model"],
            temperature=0, # temperature set to 0 to avoid overly verbose responses
        )

        try:
            probabilities_out = ast.literal_eval(response.choices[0].message.content)
        except:
            # Try to send the bad output back through chatGPT
            num_tries = 0
            for i in range(3):
                num_tries += 1
                query = f"This output is not in a plain text string representing a python dictionary, please make it formatted like {{\"name\": probability}}: {response.choices[0].message.content}"
                response = self.llm_client.chat.completions.create(
                    model=os.environ["model"],
                    messages=[
                        {"role": "system", "content": self.system_context},
                        {"role": "user", "content": query},
                    ],
                    temperature=0, # temperature set to 0 to avoid overly verbose responses
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

    @limits(calls=3, period=ONE_MINUTE)
    def generate_label_for_image(self, prompt="Using only one to two words to answer with no reasoning, please classify what you see in this image into a specific label that is intuitive for humans trying to understand what the tools here are used for relative to gardening and farming", image_url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT354l_oPc87wCqcXwlLKCRPLfZUP-Cs3Wbzw&usqp=CAU"):
        response = self.llm_client.chat.completions.create(
            model="gpt-4-turbo",
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
    
        print(f"The picture is described as: {response.choices[0].message.content}")
