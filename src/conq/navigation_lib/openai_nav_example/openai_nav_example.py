import ast
import os

from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage


class OpenAINavExample:
    def __init__(self):
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
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["API_VERSION"],
            azure_endpoint=os.environ["openai_api_base"],
            organization=os.environ["OPENAI_organization"],
        )

        # Locations in our environment
        self.locations = ["tool shed", "green house", "farm house"]

        # System context to give to the LLM before asking question
        self.system_context = "Assistant is a large language model trained by OpenAI."

    def find_probable_location_for_object(self, object: str):
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


def main():
    openai_example = OpenAINavExample()

    while True:
        user_input = input("Please input which tool to find (q to quit):")

        if user_input.lower() == "q":
            print("Quitting...")
            break
        probabilities = openai_example.find_probable_location_for_object(user_input)
        print(probabilities)


if __name__ == "__main__":
    main()
