import os
import pytest
from langchain_core.messages import HumanMessage
from langchain_gradientai.chat_models import ChatGradientAI
from dotenv import load_dotenv

load_dotenv()

MODELS_TO_TEST = [
    "openai-o3-mini",
    "mistral-nemo-instruct-2407",
    "openai-gpt-4o-mini",
    "openai-gpt-4o",
    "llama3-8b-instruct",
    "deepseek-r1-distill-llama-70b",
    "llama3.3-70b-instruct"
]

# "llama3-70b-instruct",
# "anthropic-claude-3.7-sonnet",
# "anthropic-claude-3.5-sonnet",
# "anthropic-claude-3.5-haiku",
# "anthropic-claude-3-opus",

@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_chatgradientai_all_models(model_name):
    llm = ChatGradientAI(
        model=model_name,
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    messages = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(messages)
    assert result.content
    assert isinstance(result.content, str)
    assert hasattr(result, "usage_metadata") or hasattr(result, "response_metadata") 