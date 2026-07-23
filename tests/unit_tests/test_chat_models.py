"""Test chat model integration."""

import json
import os
from types import SimpleNamespace
from typing import Type

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel

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


class TestToolCalling:
    """Unit tests for tool calling implementation."""

    def test_bind_tools_converts_to_openai_format(self) -> None:
        """Test that bind_tools converts LangChain tools to OpenAI format."""

        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: sunny"

        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")
        bound = llm.bind_tools([get_weather])

        # Check that tools are in OpenAI format
        assert "tools" in bound.kwargs
        tools = bound.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"
        assert "parameters" in tools[0]["function"]

    def test_bind_tools_defaults_tool_choice_to_auto(self) -> None:
        """Test that bind_tools defaults tool_choice to 'auto'."""

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")
        bound = llm.bind_tools([dummy_tool])

        assert bound.kwargs.get("tool_choice") == "auto"

    def test_bind_tools_respects_explicit_tool_choice(self) -> None:
        """Test that explicit tool_choice is preserved."""

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")

        # Test "none"
        bound = llm.bind_tools([dummy_tool], tool_choice="none")
        assert bound.kwargs.get("tool_choice") == "none"

        # Test "required"
        bound = llm.bind_tools([dummy_tool], tool_choice="required")
        assert bound.kwargs.get("tool_choice") == "required"

    def test_bind_tools_converts_any_to_required(self) -> None:
        """Test that tool_choice='any' is converted to 'required'."""

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")
        bound = llm.bind_tools([dummy_tool], tool_choice="any")

        assert bound.kwargs.get("tool_choice") == "required"

    def test_bind_tools_with_pydantic_model(self) -> None:
        """Test that bind_tools works with Pydantic models."""

        class WeatherQuery(BaseModel):
            """Query for weather information."""

            location: str
            unit: str = "celsius"

        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")
        bound = llm.bind_tools([WeatherQuery])

        tools = bound.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "WeatherQuery"

    def test_convert_message_with_tool_message(self) -> None:
        """Test that ToolMessage is converted with tool_call_id."""
        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")

        tool_msg = ToolMessage(content="Result: 42", tool_call_id="call_123")
        converted = llm._convert_message(tool_msg)

        assert converted["role"] == "tool"
        assert converted["content"] == "Result: 42"
        assert converted["tool_call_id"] == "call_123"

    def test_convert_message_with_ai_tool_calls(self) -> None:
        """Test that AIMessage with tool_calls is converted correctly."""
        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"location": "SF"}, "id": "call_456"}
            ],
        )
        converted = llm._convert_message(ai_msg)

        assert converted["role"] == "assistant"
        assert "tool_calls" in converted
        assert len(converted["tool_calls"]) == 1
        tc = converted["tool_calls"][0]
        assert tc["id"] == "call_456"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "SF"}

    def test_convert_message_human_message(self) -> None:
        """Test that HumanMessage is converted correctly."""
        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")

        human_msg = HumanMessage(content="Hello!")
        converted = llm._convert_message(human_msg)

        assert converted["role"] == "user"
        assert converted["content"] == "Hello!"

    def test_parse_tool_calls_from_dict(self) -> None:
        """Test parsing tool calls from dict format."""
        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")

        raw_tool_calls = [
            {
                "id": "call_789",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ]
        parsed = llm._parse_tool_calls(raw_tool_calls)

        assert len(parsed) == 1
        assert parsed[0]["name"] == "get_weather"
        assert parsed[0]["args"] == {"location": "NYC"}
        assert parsed[0]["id"] == "call_789"

    def test_parse_tool_calls_handles_invalid_json(self) -> None:
        """Test that invalid JSON in arguments is handled gracefully."""
        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")

        raw_tool_calls = [
            {
                "id": "call_bad",
                "function": {"name": "broken", "arguments": "not valid json"},
            }
        ]
        # Should not raise, just skip the invalid tool call
        parsed = llm._parse_tool_calls(raw_tool_calls)
        assert len(parsed) == 0


class TestStreaming:
    """Unit tests for streaming behavior."""

    def test_stream_skips_empty_choices_chunks(self, monkeypatch) -> None:
        """Test that chunks with empty choices are ignored during streaming."""

        class FakeGradient:
            def __init__(self, **kwargs) -> None:
                del kwargs
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                del kwargs
                return iter(
                    [
                        SimpleNamespace(choices=[]),
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(
                                        content="hello", tool_calls=None
                                    )
                                )
                            ]
                        ),
                    ]
                )

        monkeypatch.setattr("langchain_gradient.chat_models.Gradient", FakeGradient)

        llm = ChatGradient(model="llama3.3-70b-instruct", api_key="test-key")
        chunks = list(llm._stream([HumanMessage(content="hi")]))

        assert len(chunks) == 1
        assert chunks[0].message.content == "hello"


class TestStreamOptions:
    """Unit tests for stream_options request parameter handling."""

    @staticmethod
    def _fake_gradient(captured: dict):
        class FakeGradient:
            def __init__(self, **kwargs) -> None:
                del kwargs
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                captured.update(kwargs)
                if kwargs.get("stream"):
                    return iter(
                        [
                            SimpleNamespace(
                                choices=[
                                    SimpleNamespace(
                                        delta=SimpleNamespace(
                                            content="hello", tool_calls=None
                                        )
                                    )
                                ],
                                usage=SimpleNamespace(
                                    prompt_tokens=1,
                                    completion_tokens=2,
                                    total_tokens=3,
                                ),
                            ),
                        ]
                    )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="hello", tool_calls=None, refusal=None
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=1, completion_tokens=2, total_tokens=3
                    ),
                    model="llama3.3-70b-instruct",
                    id="cmpl-123",
                )

        return FakeGradient

    def test_stream_options_not_sent_on_invoke(self, monkeypatch) -> None:
        """The API rejects stream_options when stream=True is not set, so it
        must be stripped from non-streaming requests."""
        captured: dict = {}
        monkeypatch.setattr(
            "langchain_gradient.chat_models.Gradient", self._fake_gradient(captured)
        )

        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key="test-key",
            stream_options={"include_usage": True},
        )
        result = llm.invoke([HumanMessage(content="hi")])

        assert "stream_options" not in captured
        assert result.content == "hello"
        assert result.usage_metadata == {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }

    def test_stream_options_sent_when_streaming(self, monkeypatch) -> None:
        """stream_options is a valid parameter for streaming requests."""
        captured: dict = {}
        monkeypatch.setattr(
            "langchain_gradient.chat_models.Gradient", self._fake_gradient(captured)
        )

        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key="test-key",
            stream_options={"include_usage": True},
        )
        list(llm._stream([HumanMessage(content="hi")]))

        assert captured.get("stream") is True
        assert captured.get("stream_options") == {"include_usage": True}
