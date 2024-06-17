from dotenv import load_env
import os


class GPT:
    def __init__(self):
        load_env('.env.local')

        # Get the organization and key
        self.org = os.getenv('ORG_KEY')
        self.key = os.getenv('GPT_KEY')

        # Create additional items for the query
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
        }

    def