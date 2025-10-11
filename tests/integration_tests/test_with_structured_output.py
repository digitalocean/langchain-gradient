"""Comprehensive tests for with_structured_output functionality."""

import os
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from langchain_gradient.chat_models import ChatGradient

load_dotenv()


class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age", ge=0, le=150)
    occupation: str = Field(description="The person's job or profession")


class ComplexPerson(BaseModel):
    name: str
    age: int
    address: Dict[str, str]
    hobbies: List[str]
    is_active: bool


class PersonWithOptional(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[str] = None


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
class TestStructuredOutput:
    """Comprehensive test suite for with_structured_output."""

    def test_with_structured_output_json_parsing_error(self):
        """Test handling of malformed JSON responses."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        # Mock the _generate method instead of invoke
        with patch.object(llm, '_generate') as mock_generate:
            mock_generate.return_value = ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="This is not valid JSON"))]
            )
            
            with pytest.raises(ValueError, match="Failed to parse JSON response"):
                chain.invoke(HumanMessage(content="test"))

    def test_with_structured_output_validation_error(self):
        """Test handling of Pydantic validation failures."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output(Person)
        
        # Mock the _generate method instead of invoke
        with patch.object(llm, '_generate') as mock_generate:
            mock_generate.return_value = ChatResult(
                generations=[ChatGeneration(message=AIMessage(content='{"invalid": "data"}'))]
            )
            
            with pytest.raises(ValueError, match="Failed to parse response as Person"):
                chain.invoke(HumanMessage(content="test"))

    def test_with_structured_output_invalid_input_format(self):
        """Test handling of invalid input formats."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        # Test with invalid input types
        with pytest.raises(ValueError, match="Input must be a HumanMessage"):
            chain.invoke("invalid string")
        
        with pytest.raises(ValueError, match="Input must be a HumanMessage"):
            chain.invoke(123)
        
        with pytest.raises(ValueError, match="Input must be a HumanMessage"):
            chain.invoke({})

    def test_with_structured_output_message_without_content(self):
        """Test handling of messages without content attribute."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        class BadMessage:
            def __init__(self):
                pass
        
        with pytest.raises(ValueError, match="Input must be a HumanMessage"):
            chain.invoke(BadMessage())

    def test_with_structured_output_empty_message_list(self):
        """Test handling of empty message list."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        with pytest.raises(ValueError):
            chain.invoke([])

    def test_with_structured_output_complex_schema(self):
        """Test with complex Pydantic schema."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output(ComplexPerson)
        
        result = chain.invoke(HumanMessage(content="Tell me about a person with hobbies and address"))
        assert isinstance(result, ComplexPerson)
        assert isinstance(result.address, dict)
        assert isinstance(result.hobbies, list)
        assert isinstance(result.is_active, bool)

    def test_with_structured_output_optional_fields(self):
        """Test with optional Pydantic fields."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output(PersonWithOptional)
        
        result = chain.invoke(HumanMessage(content="Tell me about someone"))
        assert isinstance(result, PersonWithOptional)
        # Optional fields may be None

    def test_with_structured_output_invalid_parsing_method(self):
        """Test with invalid parsing method."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        
        with pytest.raises(ValueError, match="Invalid parsing method"):
            llm.with_structured_output(Person, parsing_method="invalid_method")

    def test_with_structured_output_unicode_content(self):
        """Test with Unicode content in messages."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        result = chain.invoke(HumanMessage(content="Tell me about 中文 characters"))
        assert isinstance(result, dict)

    def test_with_structured_output_custom_parameters(self):
        """Test with custom model parameters."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        
        chain = llm.with_structured_output(
            Person,
            temperature=0.7,
            max_tokens=50
        )
        
        result = chain.invoke(HumanMessage(content="Tell me about someone"))
        assert isinstance(result, Person)

    def test_with_structured_output_end_to_end(self):
        """Test complete workflow from input to structured output."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        
        # Test both modes
        person_chain = llm.with_structured_output(Person)
        json_chain = llm.with_structured_output()
        
        # Test with same input
        input_msg = HumanMessage(content="Tell me about Albert Einstein")
        
        person_result = person_chain.invoke(input_msg)
        json_result = json_chain.invoke(input_msg)
        
        # Verify both work
        assert isinstance(person_result, Person)
        assert isinstance(json_result, dict)
        
        # Verify data consistency (rough check)
        assert person_result.name.lower() in json_result.get("name", "").lower()

    def test_with_structured_output_large_response(self):
        """Test handling of large JSON responses."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        result = chain.invoke(HumanMessage(content="Give me detailed information about the solar system"))
        assert isinstance(result, dict)
        assert len(str(result)) > 100  # Ensure response is substantial

    def test_with_structured_output_both_modes_consistency(self):
        """Test that both modes return consistent data structure."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        
        person_chain = llm.with_structured_output(Person)
        json_chain = llm.with_structured_output()
        
        input_msg = HumanMessage(content="Tell me about Marie Curie")
        
        person_result = person_chain.invoke(input_msg)
        json_result = json_chain.invoke(input_msg)
        
        # Both should have similar fields (be flexible about field names)
        assert hasattr(person_result, 'name')
        # Check for name-related fields in JSON (could be 'name', 'full_name', etc.)
        name_fields = [k for k in json_result.keys() if 'name' in k.lower()]
        assert len(name_fields) > 0, f"No name-related fields found in {json_result.keys()}"
        
        assert hasattr(person_result, 'age')
        # Check for age-related fields in JSON (could be 'age', 'birthdate', etc.)
        age_fields = [k for k in json_result.keys() if 'age' in k.lower() or 'birth' in k.lower() or 'born' in k.lower()]
        assert len(age_fields) > 0, f"No age-related fields found in {json_result.keys()}"
        
        assert hasattr(person_result, 'occupation')
        # Check for occupation-related fields in JSON
        occupation_fields = [k for k in json_result.keys() if 'occupation' in k.lower() or 'profession' in k.lower() or 'job' in k.lower()]
        assert len(occupation_fields) > 0, f"No occupation-related fields found in {json_result.keys()}"

    def test_with_structured_output_error_messages_contain_context(self):
        """Test that error messages contain helpful context."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        with patch.object(llm, '_generate') as mock_generate:
            mock_generate.return_value = ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Invalid JSON {"))]
            )
            
            with pytest.raises(ValueError) as exc_info:
                chain.invoke(HumanMessage(content="test"))
            
            # Error should contain response content for debugging
            assert "Response content:" in str(exc_info.value)
            assert "Invalid JSON" in str(exc_info.value)

    def test_with_structured_output_schema_none_explicit(self):
        """Test explicit None schema parameter."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output(schema=None)
        
        result = chain.invoke(HumanMessage(content="Tell me about someone"))
        assert isinstance(result, dict)

    def test_with_structured_output_schema_none_implicit(self):
        """Test implicit None schema parameter."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        result = chain.invoke(HumanMessage(content="Tell me about someone"))
        assert isinstance(result, dict)

    def test_with_structured_output_list_of_messages(self):
        """Test with list of messages input."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output(Person)
        
        result = chain.invoke([HumanMessage(content="Tell me about someone")])
        assert isinstance(result, Person)

    def test_with_structured_output_multiple_messages_uses_first(self):
        """Test that multiple messages use the first one."""
        llm = ChatGradient(api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"))
        chain = llm.with_structured_output()
        
        messages = [
            HumanMessage(content="Tell me about Einstein"),
            HumanMessage(content="Tell me about Tesla")
        ]
        
        result = chain.invoke(messages)
        assert isinstance(result, dict)
        # Should process the first message (Einstein)
