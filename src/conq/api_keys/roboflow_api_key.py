from typing import Optional

from conq.api_keys.api_key_storage import ApiKeyStorage


class RoboflowApiKey(ApiKeyStorage):

    def __init__(self, username: Optional[str] = None):
        super().__init__("roboflow", username)


ROBOFLOW_API_KEY = RoboflowApiKey()
