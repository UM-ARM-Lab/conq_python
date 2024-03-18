from typing import Optional

from conq.api_keys.api_key_storage import ApiKeyStorage


class ReplicateApiKey(ApiKeyStorage):

    def __init__(self, username: Optional[str] = None):
        super().__init__("replicate", username)


REPLICATE_API_KEY = ReplicateApiKey()
