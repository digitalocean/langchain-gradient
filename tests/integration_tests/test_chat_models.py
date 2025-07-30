"""Test ChatDigitalOcean chat model."""

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gradient.chat_models import ChatGradient

load_dotenv()


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_generate_basic() -> None:
    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    messages = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(messages)
    assert result.content
    assert isinstance(result.content, str)
    assert hasattr(result, "usage_metadata") or hasattr(result, "response_metadata")


# 1. Environment & Setup
def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIGITALOCEAN_INFERENCE_KEY", raising=False)
    with pytest.raises(ValueError):
        ChatGradient(api_key=None).invoke([HumanMessage(content="test")])


# 2. Model Initialization
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_minimal_initialization() -> None:
    llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    assert llm is not None


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_full_initialization() -> None:
    llm = ChatGradient(
        model="llama3-70b-instruct",
        temperature=0.7,
        max_tokens=50,
        timeout=5,
        max_retries=2,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        stop_sequences=["\n"],
        n=1,
        top_p=0.8,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        metadata={"user": "test_user"},
    )
    assert llm is not None


# 3. Basic Functionality
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_invoke_simple() -> None:
    llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    result = llm.invoke([HumanMessage(content="Hello!")])
    assert result.content
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_invoke_multi_message() -> None:
    llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    messages = [
        HumanMessage(content="Translate to French."),
        HumanMessage(content="I love programming."),
    ]
    result = llm.invoke(messages)
    assert result.content
    assert isinstance(result.content, str)


# 5. Error Handling
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_invalid_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ChatGradient(api_key="invalid-key")
    with pytest.raises(Exception):
        llm.invoke([HumanMessage(content="test")])


# 6. Token Usage & Metadata
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_usage_metadata_in_invoke() -> None:
    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        stream_options={"include_usage": True},
    )
    messages = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(messages)
    assert result.content
    assert hasattr(result, "usage_metadata")


# 7. Retries & Timeouts
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_timeout_param() -> None:
    llm = ChatGradient(
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"), timeout=0.0001
    )
    with pytest.raises(Exception):
        llm.invoke([HumanMessage(content="This should timeout.")])


# 8. Edge Cases
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_empty_prompt() -> None:
    llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    with pytest.raises(Exception):
        llm.invoke([HumanMessage(content="")])


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_long_prompt() -> None:
    llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    long_text = " ".join(["longtext"] * 1000)
    result = llm.invoke([HumanMessage(content=long_text)])
    assert result.content is not None


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
def test_unicode_prompt() -> None:
    llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    result = llm.invoke([HumanMessage(content="你好，世界! 🌍")])
    assert result.content is not None

