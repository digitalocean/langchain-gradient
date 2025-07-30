"""Test chat model integration."""

import os
from typing import Type

from dotenv import load_dotenv
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_gradient.chat_models import ChatGradient

load_dotenv()


class TestChatGradientUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatGradient]:
        return ChatGradient

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "llama3.3-70b-instruct",
            "temperature": 0,
            "api_key": os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        }
