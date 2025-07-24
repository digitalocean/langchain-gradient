import os
import pytest
from langchain_core.messages import HumanMessage
from langchain_gradientai.chat_models import ChatGradientAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("DIGITALOCEAN_INFERENCE_KEY")

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="No GradientAI API key set",
)

def test_openai_o3_mini_max_completion_tokens():
    llm = ChatGradientAI(model="openai-o3-mini", api_key=API_KEY, max_tokens=256)
    prompt = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(prompt)
    assert result.content
    assert isinstance(result.content, str)

def test_openai_gpt_4o_max_tokens():
    llm = ChatGradientAI(model="openai-gpt-4o", api_key=API_KEY, max_completion_tokens=256)
    prompt = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(prompt)
    assert result.content
    assert isinstance(result.content, str)

def test_openai_gpt_4o_mini_max_tokens():
    llm = ChatGradientAI(model="openai-gpt-4o-mini", api_key=API_KEY, max_completion_tokens=256)
    prompt = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(prompt)
    assert result.content
    assert isinstance(result.content, str) 