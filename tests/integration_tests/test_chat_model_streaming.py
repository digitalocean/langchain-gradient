import os
import pytest
from langchain_core.messages import HumanMessage
from langchain_gradientai.chat_models import ChatGradientAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("DIGITALOCEAN_INFERENCE_KEY")
MODEL = "llama3.3-70b-instruct"

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="No GradientAI API key set",
)

def test_stream_first_and_last_chunk():
    llm = ChatGradientAI(model=MODEL, api_key=API_KEY, streaming=True)
    prompt = [HumanMessage(content="Display three cities in the world")]
    stream = llm.stream(prompt)
    found = False
    for chunk in stream:
        assert hasattr(chunk, "content")
        assert chunk.content is not None
        found = True
    assert found, "No streamed completions were received."
    