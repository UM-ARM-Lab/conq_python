from typing import Optional

from conq.api_keys.api_key_storage import ApiKeyStorage

class UmGptApiKey(ApiKeyStorage):
    def __init__(self, username: Optional[str] = None):
        super().__init__("umgpt", username)

UMGPT_API_KEY = UmGptApiKey()