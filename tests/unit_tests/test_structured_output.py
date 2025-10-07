"""Unit tests for structured output functionality."""

import json
import pytest
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_gradient.chat_models import (
    ChatGradient,
    StructuredOutputError,
    StructuredOutputRunnable,
)


# Test Pydantic models
class Person(BaseModel):
    """A simple person model."""
    name: str
    age: int
    email: str


class PersonWithOptional(BaseModel):
    """A person model with optional fields."""
    name: str
    age: int
    email: Optional[str] = None
    phone: Optional[str] = None


class PersonList(BaseModel):
    """A model containing a list of persons."""
    people: List[Person]
    total_count: int


class ComplexModel(BaseModel):
    """A more complex model with nested structures."""
    id: int
    title: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class TestStructuredOutputRunnable:
    """Test the StructuredOutputRunnable class."""

    def test_init(self):
        """Test initialization of StructuredOutputRunnable."""
        llm = Mock(spec=ChatGradient)
        runnable = StructuredOutputRunnable(
            llm=llm,
            schema=Person,
            method="json_mode",
            include_raw=False,
        )
        
        assert runnable.llm is llm
        assert runnable.schema is Person
        assert runnable.method == "json_mode"
        assert runnable.include_raw is False

    def test_get_schema_description(self):
        """Test schema description generation."""
        llm = Mock(spec=ChatGradient)
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        
        description = runnable._get_schema_description()
        assert isinstance(description, str)
        assert "name" in description
        assert "age" in description
        assert "email" in description

    def test_extract_json_from_code_block(self):
        """Test JSON extraction from code blocks."""
        llm = Mock(spec=ChatGradient)
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        
        # Test with ```json code block
        content = '```json\n{"name": "John", "age": 30, "email": "john@example.com"}\n```'
        result = runnable._extract_json(content)
        assert result == '{"name": "John", "age": 30, "email": "john@example.com"}'
        
        # Test with ``` code block (no language)
        content = '```\n{"name": "John", "age": 30, "email": "john@example.com"}\n```'
        result = runnable._extract_json(content)
        assert result == '{"name": "John", "age": 30, "email": "john@example.com"}'

    def test_extract_json_from_plain_text(self):
        """Test JSON extraction from plain text."""
        llm = Mock(spec=ChatGradient)
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        
        # Test with JSON in the middle of text
        content = 'Here is the person data: {"name": "John", "age": 30, "email": "john@example.com"} as requested.'
        result = runnable._extract_json(content)
        assert result == '{"name": "John", "age": 30, "email": "john@example.com"}'

    def test_parse_response_success(self):
        """Test successful response parsing."""
        llm = Mock(spec=ChatGradient)
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        
        content = '{"name": "John", "age": 30, "email": "john@example.com"}'
        result = runnable._parse_response(content)
        
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_parse_response_invalid_json(self):
        """Test parsing with invalid JSON."""
        llm = Mock(spec=ChatGradient)
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        
        content = '{"name": "John", "age": 30, "email": "john@example.com"'  # Missing closing brace
        
        with pytest.raises(StructuredOutputError) as exc_info:
            runnable._parse_response(content)
        
        assert "Invalid JSON" in str(exc_info.value)

    def test_parse_response_validation_error(self):
        """Test parsing with Pydantic validation error."""
        llm = Mock(spec=ChatGradient)
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        
        content = '{"name": "John", "age": "thirty", "email": "john@example.com"}'  # age should be int
        
        with pytest.raises(StructuredOutputError) as exc_info:
            runnable._parse_response(content)
        
        assert "Pydantic validation failed" in str(exc_info.value)

    def test_invoke_with_string_input(self):
        """Test invoke with string input."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='{"name": "John", "age": 30, "email": "john@example.com"}')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        result = runnable.invoke("Create a person named John")
        
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_invoke_with_messages_input(self):
        """Test invoke with messages input."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='{"name": "Jane", "age": 25, "email": "jane@example.com"}')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        messages = [HumanMessage(content="Create a person named Jane")]
        result = runnable.invoke(messages)
        
        assert isinstance(result, Person)
        assert result.name == "Jane"
        assert result.age == 25
        assert result.email == "jane@example.com"

    def test_invoke_with_include_raw(self):
        """Test invoke with include_raw=True."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='{"name": "Bob", "age": 35, "email": "bob@example.com"}')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=Person, include_raw=True)
        result = runnable.invoke("Create a person named Bob")
        
        assert isinstance(result, dict)
        assert "parsed" in result
        assert "raw" in result
        assert "parsing_error" in result
        
        assert isinstance(result["parsed"], Person)
        assert result["parsed"].name == "Bob"
        assert result["raw"] is mock_response
        assert result["parsing_error"] is None

    def test_invoke_with_parsing_error_and_include_raw(self):
        """Test invoke with parsing error and include_raw=True."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='invalid json')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=Person, include_raw=True)
        result = runnable.invoke("Create a person")
        
        assert isinstance(result, dict)
        assert result["parsed"] is None
        assert result["raw"] is mock_response
        assert result["parsing_error"] is not None
        assert "Invalid JSON" in result["parsing_error"] or "Failed to parse structured output" in result["parsing_error"]

    def test_invoke_with_parsing_error_without_include_raw(self):
        """Test invoke with parsing error and include_raw=False."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='invalid json')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=Person, include_raw=False)
        
        with pytest.raises(StructuredOutputError):
            runnable.invoke("Create a person")


class TestChatGradientStructuredOutput:
    """Test the with_structured_output method of ChatGradient."""

    def test_with_structured_output_returns_runnable(self):
        """Test that with_structured_output returns a StructuredOutputRunnable."""
        llm = ChatGradient(api_key="test-key")
        structured_llm = llm.with_structured_output(Person)
        
        assert isinstance(structured_llm, StructuredOutputRunnable)
        assert structured_llm.llm is llm
        assert structured_llm.schema is Person
        assert structured_llm.method == "json_mode"
        assert structured_llm.include_raw is False

    def test_with_structured_output_custom_parameters(self):
        """Test with_structured_output with custom parameters."""
        llm = ChatGradient(api_key="test-key")
        structured_llm = llm.with_structured_output(
            Person,
            method="json_mode",
            include_raw=True,
        )
        
        assert isinstance(structured_llm, StructuredOutputRunnable)
        assert structured_llm.method == "json_mode"
        assert structured_llm.include_raw is True

    def test_with_structured_output_different_models(self):
        """Test with_structured_output with different Pydantic models."""
        llm = ChatGradient(api_key="test-key")
        
        # Test with simple model
        simple_structured = llm.with_structured_output(Person)
        assert simple_structured.schema is Person
        
        # Test with model with optional fields
        optional_structured = llm.with_structured_output(PersonWithOptional)
        assert optional_structured.schema is PersonWithOptional
        
        # Test with complex model
        complex_structured = llm.with_structured_output(ComplexModel)
        assert complex_structured.schema is ComplexModel
        
        # Test with list model
        list_structured = llm.with_structured_output(PersonList)
        assert list_structured.schema is PersonList


class TestComplexStructuredOutputScenarios:
    """Test complex scenarios with structured output."""

    def test_nested_model_parsing(self):
        """Test parsing of nested models."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='''
        {
            "id": 1,
            "title": "Test Article",
            "description": "A test article",
            "tags": ["test", "article"],
            "metadata": {"author": "John Doe", "category": "tech"}
        }
        ''')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=ComplexModel)
        result = runnable.invoke("Create a complex model")
        
        assert isinstance(result, ComplexModel)
        assert result.id == 1
        assert result.title == "Test Article"
        assert result.description == "A test article"
        assert result.tags == ["test", "article"]
        assert result.metadata == {"author": "John Doe", "category": "tech"}

    def test_list_model_parsing(self):
        """Test parsing of models containing lists."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='''
        {
            "people": [
                {"name": "John", "age": 30, "email": "john@example.com"},
                {"name": "Jane", "age": 25, "email": "jane@example.com"}
            ],
            "total_count": 2
        }
        ''')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=PersonList)
        result = runnable.invoke("Create a list of people")
        
        assert isinstance(result, PersonList)
        assert len(result.people) == 2
        assert result.total_count == 2
        assert result.people[0].name == "John"
        assert result.people[1].name == "Jane"

    def test_optional_fields_handling(self):
        """Test handling of optional fields."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='{"name": "John", "age": 30}')  # Missing optional fields
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=PersonWithOptional)
        result = runnable.invoke("Create a person")
        
        assert isinstance(result, PersonWithOptional)
        assert result.name == "John"
        assert result.age == 30
        assert result.email is None
        assert result.phone is None

    def test_json_in_markdown_code_block(self):
        """Test JSON extraction from markdown code blocks."""
        llm = Mock(spec=ChatGradient)
        mock_response = AIMessage(content='''
        Here's the person data you requested:
        
        ```json
        {
            "name": "Alice",
            "age": 28,
            "email": "alice@example.com"
        }
        ```
        
        This person meets all your requirements.
        ''')
        llm.invoke.return_value = mock_response
        
        runnable = StructuredOutputRunnable(llm=llm, schema=Person)
        result = runnable.invoke("Create a person")
        
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 28
        assert result.email == "alice@example.com"
