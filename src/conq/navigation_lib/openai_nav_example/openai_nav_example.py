from ratelimit import RateLimitException

from conq.navigation_lib.openai.openai_nav_client import OpenAINavClient


def main():
    openai_example = OpenAINavClient()

    while True:
        user_input = input("Please input which tool to find (q to quit):")

        if user_input.lower() == "q":
            print("Quitting...")
            break
        try:
            probabilities = openai_example.find_probable_location_for_object(user_input)
        except RateLimitException as e:
            print(f"RateLimit exceeded for OpenAI Azure calls! RateLimitException: {e}")

        print(probabilities)


if __name__ == "__main__":
    main()
