from typing import Optional

from conq.api_keys.api_key_storage import ApiKeyStorage


class OpenaiApiKey(ApiKeyStorage):
    def __init__(self, username: Optional[str] = None):
        super().__init__("openai", username)


OPENAI_API_KEY = OpenaiApiKey()
