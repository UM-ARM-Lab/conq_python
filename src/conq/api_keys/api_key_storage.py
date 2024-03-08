from typing import Optional
from abc import ABC, abstractmethod
import subprocess

import keyring


class ApiKeyStorage(ABC):
    """Base class for convenient storage and retrieval of API keys

    To use this class, create a subclass and implement the `__init__` method, calling this class's
    `__init__` method with the service name and (optionally) the username.
    """

    @abstractmethod
    def __init__(self, service: str, username: Optional[str] = None):
        """Initializes the API key storage

        Defaults to the current user's username if none is provided.
        """
        self.service: str = service
        self.username: str = self._determine_username(username)
        self._key: Optional[str] = None

    def get(self) -> str:
        """Returns the API key from the keyring

        Prompts the user to input the API key if it is not already set
        """
        key = self._key
        if key is None:
            # Attempt to retrieve the key first as keys persist across sessions.
            key = keyring.get_password(self.service, self.username)

            # If the key is not found, prompt the user to input it.
            if key is None:
                self.set_via_user_input()
                key = self.get()

        return key

    def delete(self):
        keyring.delete_password(self.service, self.username)
        self._key = None

    def set_via_user_input(self):
        """Prompts the user to set the API key"""
        key = input(f"Please enter the API key for {self.service}: ")
        self._set_key(key)

    def _set_key(self, secret: str):
        """Sets the API key in the keyring

        NOTE: This should not be called directly! Use `set` instead.
        """
        keyring.set_password(self.service, self.username, secret)
        self._key = secret

    def _determine_username(self, username: Optional[str] = None):
        """Sets the username for the API key"""
        if username is None:
            username = subprocess.check_output(["whoami"]).decode("utf-8").strip()
        return username
