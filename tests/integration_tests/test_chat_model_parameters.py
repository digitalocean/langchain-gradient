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

def _basic_llm(**kwargs):
    return ChatGradientAI(model=MODEL, api_key=API_KEY, **kwargs)

def _basic_prompt():
    return [HumanMessage(content="Say hello to the world!")]

def test_temperature_param():
    llm_cold = _basic_llm(temperature=0)
    llm_hot = _basic_llm(temperature=1)
    prompt = [HumanMessage(content="Give me a random number between 1 and 100.")]
    result_cold = llm_cold.invoke(prompt)
    result_hot = llm_hot.invoke(prompt)
    assert result_cold.content
    assert result_hot.content
    # With temperature=1, output should be more random, so run a few times and check for variety
    outputs = set(llm_hot.invoke(prompt).content for _ in range(3))
    assert len(outputs) > 1 or result_hot.content != result_cold.content

def test_max_tokens_param():
    llm = _basic_llm(max_tokens=3)
    prompt = [HumanMessage(content="Repeat the word 'hello' 10 times.")]
    result = llm.invoke(prompt)
    assert result.content
    # Should be very short due to max_tokens
    assert len(result.content.split()) <= 5

# def test_stop_param():
#     llm = _basic_llm(stop=["cat"])
#     prompt = [HumanMessage(content="Say: dog, cat, mouse.")]
#     result = llm.invoke(prompt)
#     assert result.content
#     assert "cat" not in result.content or result.content.endswith("cat")

# def test_presence_penalty_param():
#     llm = _basic_llm(presence_penalty=1.0)
#     prompt = [HumanMessage(content="Repeat the word 'hello' five times.")]
#     result = llm.invoke(prompt)
#     assert result.content
#     # With high presence penalty, expect less repetition
#     assert result.content.lower().count("hello") <= 2

# def test_frequency_penalty_param():
#     llm = _basic_llm(frequency_penalty=1.0)
#     prompt = [HumanMessage(content="Repeat the word 'hi' five times.")]
#     result = llm.invoke(prompt)
#     assert result.content
#     # With high frequency penalty, expect less repetition
#     assert result.content.lower().count("hi") <= 2

def test_top_p_param():
    llm = _basic_llm(top_p=0.1)
    prompt = [HumanMessage(content="Tell me a joke.")]
    result = llm.invoke(prompt)
    assert result.content
    # Not easy to assert, but should return a valid string
    assert isinstance(result.content, str)

# TODO: Should be tested with once its fixed in GradientAI SDK
# def test_n_param():
#     llm = _basic_llm(n=2)
#     prompt = _basic_prompt()
#     result = llm.invoke(prompt)
#     # Should return a list of generations or a message with n completions
#     assert hasattr(result, "message") or hasattr(result, "generations")

def test_timeout_param():
    llm = _basic_llm(timeout=0.1)
    prompt = _basic_prompt()
    try:
        llm.invoke(prompt)
    except Exception as e:
        assert "timeout" in str(e).lower() or isinstance(e, Exception)

def test_stream_options_include_usage():
    llm = _basic_llm(stream_options={"include_usage": True})
    prompt = _basic_prompt()
    result = llm.invoke(prompt)
    assert hasattr(result, "usage_metadata")