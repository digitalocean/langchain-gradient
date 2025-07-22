"""Test ChatDigitalOcean chat model."""

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gradientai.chat_models import ChatGradientAI

load_dotenv()


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_generate_basic() -> None:
    llm = ChatGradientAI(
        model="llama3.3-70b-instruct",
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    messages = [HumanMessage(content="Say hello to the world!")]
    result = llm.invoke(messages)
    assert result.content
    assert isinstance(result.content, str)
    assert hasattr(result, "usage_metadata") or hasattr(result, "response_metadata")


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_generate_basic_openai_model() -> None:
    llm = ChatGradientAI(
        model="openai-gpt-4o",
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    messages = [HumanMessage(content="What is the capital of France?")]
    result = llm.invoke(messages)
    print("result:", result)
    assert result.content
    assert isinstance(result.content, str)
    assert hasattr(result, "usage_metadata") or hasattr(result, "response_metadata")


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_generate_basic_openai_model_all_params() -> None:
    llm = ChatGradientAI(
        model="openai-gpt-4o",
        temperature=0.5,
        max_tokens=100,
        timeout=10,
        max_retries=3,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        stop_sequences=["\n"],
        n=2,
        top_p=0.9,
        stream=False,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    messages = [HumanMessage(content="Write a short story about a Dinosaur?")]
    result = llm.invoke(messages)
    print("result:", result)
    assert result.content
    assert isinstance(result.content, str)
    assert hasattr(result, "usage_metadata") or hasattr(result, "response_metadata")


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_generate_basic_openai_model_all_params_streaming() -> None:
    llm = ChatGradientAI(
        model="openai-gpt-4o",
        temperature=0.5,
        max_tokens=100,
        timeout=10,
        max_retries=3,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        stop_sequences=["\n"],
        n=2,
        top_p=0.9,
        stream=True,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    messages = [HumanMessage(content="Write a short story about a Dinosaur?")]
    stream = llm.stream(messages)
    found_content = False
    found_usage = False
    for completion in stream:
        # Handle both ChatGenerationChunk and AIMessageChunk
        if hasattr(completion, "message"):
            msg = completion.message
        else:
            msg = completion  # It's already an AIMessageChunk
        assert msg is not None
        if hasattr(msg, "content") and msg.content:
            found_content = True
        if hasattr(msg, "usage_metadata") and msg.usage_metadata is not None:
            found_usage = True
    assert found_content, "No streamed completions were received."
    assert not found_usage, (
        "Usage metadata was included in the stream when it should not have been."
    )


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_streaming_basic() -> None:
    llm = ChatGradientAI(
        model="llama3.3-70b-instruct",
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    stream = llm.stream("count to three, one token at a time.")
    found = False
    for chunk in stream:
        assert hasattr(chunk, "content")
        assert chunk.content is not None
        found = True
    assert found, "No streamed completions were received."


# 1. Environment & Setup


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIGITALOCEAN_INFERENCE_KEY", raising=False)
    with pytest.raises(ValueError):
        ChatGradientAI(api_key=None).invoke([HumanMessage(content="test")])


# 2. Model Initialization
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_minimal_initialization() -> None:
    llm = ChatGradientAI(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    assert llm is not None


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_full_initialization() -> None:
    llm = ChatGradientAI(
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
    reason="No GradientAI API key set",
)
def test_invoke_simple() -> None:
    llm = ChatGradientAI(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    result = llm.invoke([HumanMessage(content="Hello!")])
    assert result.content
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_invoke_multi_message() -> None:
    llm = ChatGradientAI(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    messages = [
        HumanMessage(content="Translate to French."),
        HumanMessage(content="I love programming."),
    ]
    result = llm.invoke(messages)
    assert result.content
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_stream_final_chunk_usage_metadata() -> None:
    llm = ChatGradientAI(
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        stream_options={"include_usage": True},
    )
    stream = llm.stream([HumanMessage(content="Say hello!")])
    last_chunk = None
    for chunk in stream:
        print(type(chunk))
        last_chunk = chunk

    if isinstance(last_chunk, ChatGenerationChunk):
        assert hasattr(last_chunk.message, "usage_metadata")
        assert last_chunk.message.usage_metadata is not None
    elif isinstance(last_chunk, AIMessageChunk):
        assert last_chunk.usage_metadata is not None
    else:
        raise AssertionError(f"Unexpected chunk type: {type(last_chunk)}")


# 4. Parameter Handling
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
@pytest.mark.parametrize("temperature", [0, 0.5, 1])
def test_temperature_param(temperature: float) -> None:
    llm = ChatGradientAI(
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"), temperature=temperature
    )
    result = llm.invoke([HumanMessage(content="Say something random.")])
    assert result.content


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_max_tokens_param() -> None:
    llm = ChatGradientAI(
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"), max_tokens=5
    )
    result = llm.invoke([HumanMessage(content="Say a long sentence.")])
    assert result.content


# 5. Error Handling
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_invalid_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ChatGradientAI(api_key="invalid-key")
    with pytest.raises(Exception):
        llm.invoke([HumanMessage(content="test")])


# 6. Token Usage & Metadata
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_usage_metadata_in_invoke() -> None:
    llm = ChatGradientAI(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    result = llm.invoke([HumanMessage(content="Say hello!")])
    assert hasattr(result, "usage_metadata")
    assert result.usage_metadata is not None


# 7. Retries & Timeouts
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_timeout_param() -> None:
    llm = ChatGradientAI(
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"), timeout=0.0001
    )
    with pytest.raises(Exception):
        llm.invoke([HumanMessage(content="This should timeout.")])


# 8. Edge Cases
@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_empty_prompt() -> None:
    llm = ChatGradientAI(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    with pytest.raises(Exception):
        llm.invoke([HumanMessage(content="")])


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_long_prompt() -> None:
    llm = ChatGradientAI(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    long_text = " ".join(["longtext"] * 1000)
    result = llm.invoke([HumanMessage(content=long_text)])
    assert result.content is not None


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_unicode_prompt() -> None:
    llm = ChatGradientAI(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
    result = llm.invoke([HumanMessage(content="你好，世界! 🌍")])
    assert result.content is not None


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No GradientAI API key set",
)
def test_stream_options_include_usage() -> None:
    llm = ChatGradientAI(
        model="openai-gpt-4o",
        temperature=0.5,
        max_tokens=20,
        streaming=True,
        stream_options={"include_usage": True},
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )

    messages = [HumanMessage(content="Tell me a fun fact about dinosaurs.")]
    stream = llm.stream(messages)
    found_content = False
    found_usage = False
    for chunk in stream:
        # Handle both ChatGenerationChunk and AIMessageChunk
        if hasattr(chunk, "message"):
            msg = chunk.message
        else:
            msg = chunk
        if hasattr(msg, "content") and msg.content:
            found_content = True
        if hasattr(msg, "usage_metadata") and msg.usage_metadata is not None:
            found_usage = True
    assert found_content, "No streamed completions were received."
    assert found_usage, "No usage metadata was received in the final chunk."
