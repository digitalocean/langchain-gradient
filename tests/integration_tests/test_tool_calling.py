"""Integration tests for tool calling functionality."""

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_gradient.chat_models import ChatGradient

load_dotenv()

API_KEY = os.environ.get("DIGITALOCEAN_INFERENCE_KEY")


@pytest.mark.skipif(not API_KEY, reason="No Gradient API key set")
class TestToolCallingIntegration:
    """Integration tests for tool calling with the actual API."""

    def test_bind_tools_returns_tool_calls(self) -> None:
        """Test that bind_tools + invoke returns tool_calls in response."""

        @tool
        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Weather in {location}: sunny"

        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key=API_KEY,
            temperature=0,
        )
        llm_with_tools = llm.bind_tools([get_weather])

        response = llm_with_tools.invoke("What's the weather in San Francisco?")

        assert isinstance(response, AIMessage)
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0]["name"] == "get_weather"
        assert "location" in response.tool_calls[0]["args"]

    def test_tool_message_in_conversation(self) -> None:
        """Test that ToolMessage can be included in follow-up conversation."""

        @tool
        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Weather in {location}: sunny"

        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key=API_KEY,
            temperature=0,
        )
        llm_with_tools = llm.bind_tools([get_weather])

        # First turn: get tool call
        response1 = llm_with_tools.invoke("What's the weather in Tokyo?")
        assert len(response1.tool_calls) > 0

        # Second turn: provide tool result
        tool_call = response1.tool_calls[0]
        messages = [
            HumanMessage(content="What's the weather in Tokyo?"),
            response1,
            ToolMessage(content="Weather in Tokyo: 72°F and cloudy", tool_call_id=tool_call["id"]),
        ]
        response2 = llm_with_tools.invoke(messages)

        # Model should respond with the weather info
        assert isinstance(response2, AIMessage)
        assert response2.content  # Should have text content now

    def test_tool_choice_required(self) -> None:
        """Test that tool_choice='required' forces a tool call."""

        @tool
        def say_hello(name: str) -> str:
            """Say hello to someone."""
            return f"Hello, {name}!"

        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key=API_KEY,
            temperature=0,
        )
        llm_with_tools = llm.bind_tools([say_hello], tool_choice="required")

        # Even a simple question should result in a tool call
        response = llm_with_tools.invoke("Hi there!")

        assert isinstance(response, AIMessage)
        assert len(response.tool_calls) > 0

    def test_multiple_tools(self) -> None:
        """Test bind_tools with multiple tools."""

        @tool
        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Weather in {location}: sunny"

        @tool
        def get_time(timezone: str) -> str:
            """Get the current time in a timezone."""
            return f"Time in {timezone}: 3:00 PM"

        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key=API_KEY,
            temperature=0,
        )
        llm_with_tools = llm.bind_tools([get_weather, get_time])

        # Ask about weather - should use get_weather
        response = llm_with_tools.invoke("What's the weather like in Paris?")
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0]["name"] == "get_weather"

