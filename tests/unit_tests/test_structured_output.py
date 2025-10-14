"""Unit tests for structured output parsing and validation."""

from typing import List

import pytest
from pydantic import BaseModel

from langchain_gradient.chat_models import ChatGradient


class Person(BaseModel):
    name: str
    age: int
    email: str


class DummyLLM(ChatGradient):
    """A tiny fake LLM that returns a pre-canned response for testing."""

    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self._content = content

    def invoke(self, messages: List, **kwargs):
        # mimic the object returned by ChatGradient.invoke used in code
        class R:
            def __init__(self, content):
                self.content = content

        return R(self._content)


def test_single_structured_output_success():
    json_str = '{"name": "John", "age": 30, "email": "john@example.com"}'
    llm = DummyLLM(content=json_str)
    structured = llm.with_structured_output(Person)
    person = structured.invoke(messages=["prompt"])
    assert isinstance(person, Person)
    assert person.name == "John"


def test_multiple_structured_output_success():
    json_str = '[{"name": "Alice", "age": 25, "email": "a@example.com"}, {"name": "Bob", "age": 28, "email": "b@example.com"}]'
    llm = DummyLLM(content=json_str)
    structured = llm.with_structured_output(Person, multiple=True)
    people = structured.invoke(messages=["prompt"])
    assert isinstance(people, list)
    assert all(isinstance(p, Person) for p in people)
    assert people[0].name == "Alice"


def test_invalid_json_raises():
    llm = DummyLLM(content="not a json")
    structured = llm.with_structured_output(Person)
    with pytest.raises(ValueError):
        structured.invoke(messages=["prompt"])


def test_validation_error_raises():
    # Missing age field
    json_str = '{"name": "John", "email": "john@example.com"}'
    llm = DummyLLM(content=json_str)
    structured = llm.with_structured_output(Person)
    with pytest.raises(ValueError) as e:
        structured.invoke(messages=["prompt"])
    assert "Validation error" in str(e.value)
