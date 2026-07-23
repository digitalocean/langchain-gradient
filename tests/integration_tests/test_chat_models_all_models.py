import os
import pytest
from langchain_core.messages import HumanMessage
from langchain_gradient.chat_models import ChatGradient
from dotenv import load_dotenv

load_dotenv()

MODELS_TO_TEST = [
    "openai-o3-mini",
    "mistral-3-14B",
    "openai-gpt-4o-mini",
    "openai-gpt-4o",
    "openai-gpt-oss-20b",
    "deepseek-r1-distill-llama-70b",
    "llama3.3-70b-instruct"
]

@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_chatgradient_all_models(model_name):
    llm = ChatGradient(
        model=model_name,
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    messages = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(messages)
    assert result.content
    assert isinstance(result.content, str)
    assert hasattr(result, "usage_metadata") or hasattr(result, "response_metadata") 