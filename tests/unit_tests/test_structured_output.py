"""Tests for structured output functionality."""

import pytest
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage
from unittest.mock import Mock, patch

from langchain_gradient import ChatGradient


class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    email: str = Field(description="The person's email")


class TestStructuredOutputPydantic:
    """Test structured output with Pydantic models."""
    
    @patch.object(ChatGradient, '_generate')
    def test_simple_pydantic_output(self, mock_generate):
        """Test basic Pydantic structured output."""
        mock_response = Mock()
        mock_response.content = """```
{
    "name": "John Doe",
    "age": 30,
    "email": "john@example.com"
}
```"""
        mock_generate.return_value = Mock(generations=[Mock(message=mock_response)])
        
        llm = ChatGradient(model="llama3.3-70b-instruct")
        structured_llm = llm.with_structured_output(Person)
        
        result = structured_llm.invoke("Create a person named John Doe")
        
        assert isinstance(result, Person)
        assert result.name == "John Doe"
        assert result.age == 30
        assert result.email == "john@example.com"
    
    @patch.object(ChatGradient, '_generate')
    def test_validation_error(self, mock_generate):
        """Test that invalid data raises ValidationError."""
        mock_response = Mock()
        mock_response.content = """```
{
    "name": "John",
    "age": "not_a_number",
    "email": "john@example.com"
}
```"""
        mock_generate.return_value = Mock(generations=[Mock(message=mock_response)])
        
        llm = ChatGradient(model="llama3.3-70b-instruct")
        structured_llm = llm.with_structured_output(Person)
        
        with pytest.raises(ValueError) as exc_info:
            structured_llm.invoke("Create a person")
        
        assert "Failed to validate output" in str(exc_info.value)


class TestStructuredOutputRawMode:
    """Test include_raw parameter."""
    
    @patch.object(ChatGradient, '_generate')
    def test_include_raw_success(self, mock_generate):
        """Test include_raw returns structured response."""
        mock_response = Mock()
        mock_response.content = """```
{"name": "Alice", "age": 25, "email": "alice@example.com"}
```"""
        mock_generate.return_value = Mock(generations=[Mock(message=mock_response)])
        
        llm = ChatGradient(model="llama3.3-70b-instruct")
        structured_llm = llm.with_structured_output(Person, include_raw=True)
        
        result = structured_llm.invoke("Create Alice")
        
        assert "raw" in result
        assert "parsed" in result
        assert "parsing_error" in result
        assert isinstance(result["parsed"], Person)
        assert result["parsing_error"] is None
