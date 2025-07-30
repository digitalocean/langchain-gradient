import os
import pytest
from langchain_core.messages import HumanMessage
from langchain_gradient.chat_models import ChatGradient
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("DIGITALOCEAN_INFERENCE_KEY")
MODEL = "llama3.3-70b-instruct"

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="No Gradient API key set",
)


def test_model_alias():
    llm = ChatGradient(model=MODEL, api_key=API_KEY)
    prompt = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(prompt)
    assert result.content
    assert isinstance(result.content, str)

# def test_stop_sequences_alias():
#     llm = ChatGradient(model=MODEL, api_key=API_KEY, stop_sequences=["cat", "cat,", "cat.", "Cat", "Cat,", "Cat."])
#     prompt = [HumanMessage(content="Say: dog, cat, mouse.")]
#     result = llm.invoke(prompt)
#     assert result.content
#     assert isinstance(result.content, str) 