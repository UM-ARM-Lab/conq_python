import argparse
import json
import sys

import numpy as np
from PIL import Image

from conq.navigation_lib.openai_nav.openai_nav_client import OpenAINavClient

# Main function with argument parsing that takes in a relative path to a JSON file and then saves the prompt and image URL array into variables
# Then, it creates an OpenAINavClient object and calls the generate_label_for_image function with the prompt and image URL


def main(argv):
    parser = argparse.ArgumentParser(description="OpenAI Vision API Example")
    parser.add_argument(
        "--json-filepath",
        type=str,
        help="Relative path to JSON file with prompt and image URL",
        required=True,
    )
    args = parser.parse_args(argv)

    with open(args.json_filepath, "r") as file:
        data = file.read()

    data = json.loads(data)
    prompt = data["user_prompt"]
    image_urls = data["image_inputs"]

    client = OpenAINavClient(locations=[])

    while True:
        user_prompt = input(
            "Enter prompt, (j) for input in JSON file, (q) to quit, or another string to be used as the prompt: "
        )
        if user_prompt == "q":
            break
        elif user_prompt == "j":
            for image_url in image_urls:
                print(f"Processing Image URL: {image_url}")
                client.generate_label_for_image(prompt=prompt, image_url=image_url)
        elif user_prompt == "n":
            """ NOTE: THIS IS USED JUST TO TEST THAT AN NDARRAY CAN BE AN INPUT"""
            # Create generic low-res "image" with numpy ndarray to test
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            print(f"Processing Image: {image}")
            client.generate_label_for_image(prompt=prompt, image=image)
        else:
            for image_url in image_urls:
                print(f"Processing Image URL: {image_url}")
                client.generate_label_for_image(prompt=user_prompt, image_url=image_url)


if __name__ == "__main__":
    main(sys.argv[1:])
