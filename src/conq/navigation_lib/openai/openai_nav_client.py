import ast
import os
from typing import List

from dotenv import load_dotenv
from openai import AzureOpenAI
from ratelimit import limits

from conq.api_keys import UMGPT_API_KEY

# Change this based on which API you are using
os.environ['UMGPT_API_KEY'] = UMGPT_API_KEY.get()

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

        # Create Azure client
        self.llm_client = AzureOpenAI(
            api_key=os.environ["UMGPT_API_KEY"],
            api_version=os.environ["API_VERSION"],
            azure_endpoint=os.environ["openai_api_base"],
            organization=os.environ["OPENAI_organization"],
        )

        # Locations in our environment
        self.locations = locations

        # System context to give to the LLM before asking question
        self.system_context = "Assistant is a large language model trained by OpenAI."

    @limits(calls=3, period=ONE_MINUTE)
    def find_probable_location_for_object(self, object: str):
        # TODO: Test how this does with the locations query with underscores
        # Build the query string
        query = "I observe the following structures while exploring a small-scale farm:"

        for index in range(len(self.locations)):
            query += f"\n{index + 1}. {self.locations[index]}"

        query += f"\n\nPlease rank the structure based on how likely I am to find a {object} in them. Please provide the response in a plain text string representing a dictionary and only include the location with whitespace connected with _ as the keys and probabilities as the values"

        # Make the query
        response = self.llm_client.chat.completions.create(
            model=os.environ["model"],
            messages=[
                {"role": "system", "content": self.system_context},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        probabilities_out = ast.literal_eval(response.choices[0].message.content)

        return probabilities_out
