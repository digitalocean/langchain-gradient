"""Integration tests for structured output functionality."""

import os
import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_gradient.chat_models import ChatGradient, StructuredOutputError


# Test Pydantic models for integration tests
class Person(BaseModel):
    """A person model for testing."""
    name: str
    age: int
    email: str


class PersonWithOptional(BaseModel):
    """A person model with optional fields."""
    name: str
    age: int
    email: Optional[str] = None
    occupation: Optional[str] = None


class Company(BaseModel):
    """A company model for testing."""
    name: str
    industry: str
    founded_year: int
    employees: int
    headquarters: str


class PersonList(BaseModel):
    """A model containing multiple persons."""
    people: List[Person]
    total_count: int


class ProductReview(BaseModel):
    """A product review model."""
    product_name: str
    rating: int = Field(..., ge=1, le=5)
    review_text: str
    reviewer_name: str
    would_recommend: bool


@pytest.mark.skipif(
    not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    reason="No Gradient API key set",
)
class TestStructuredOutputIntegration:
    """Integration tests for structured output with real API calls."""

    def test_basic_person_creation(self):
        """Test basic person creation with structured output."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(Person)
        
        result = structured_llm.invoke(
            "Create a person named John Smith, age 30, email john.smith@example.com"
        )
        
        assert isinstance(result, Person)
        assert result.name == "John Smith"
        assert result.age == 30
        assert result.email == "john.smith@example.com"

    def test_person_with_optional_fields(self):
        """Test person creation with optional fields."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(PersonWithOptional)
        
        result = structured_llm.invoke(
            "Create a person named Alice Johnson, age 28, email alice@example.com, occupation Software Engineer"
        )
        
        assert isinstance(result, PersonWithOptional)
        assert result.name == "Alice Johnson"
        assert result.age == 28
        assert result.email == "alice@example.com"
        assert result.occupation == "Software Engineer"

    def test_company_creation(self):
        """Test company creation with structured output."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(Company)
        
        result = structured_llm.invoke(
            "Create a tech company called TechCorp, founded in 2010, with 500 employees, headquartered in San Francisco"
        )
        
        assert isinstance(result, Company)
        assert result.name == "TechCorp"
        assert result.industry == "tech" or "technology" in result.industry.lower()
        assert result.founded_year == 2010
        assert result.employees == 500
        assert "San Francisco" in result.headquarters

    def test_product_review_creation(self):
        """Test product review creation with field validation."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(ProductReview)
        
        result = structured_llm.invoke(
            "Create a product review for iPhone 15, rating 4 out of 5, "
            "review text 'Great phone with excellent camera', "
            "reviewer name Sarah Wilson, would recommend true"
        )
        
        assert isinstance(result, ProductReview)
        assert result.product_name == "iPhone 15"
        assert result.rating == 4
        assert 1 <= result.rating <= 5  # Validate field constraint
        assert "Great phone" in result.review_text or "excellent camera" in result.review_text
        assert result.reviewer_name == "Sarah Wilson"
        assert result.would_recommend is True

    def test_multiple_people_creation(self):
        """Test creation of multiple people in a list."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(PersonList)
        
        result = structured_llm.invoke(
            "Create a list of 3 people: "
            "1. John Doe, age 30, email john@example.com "
            "2. Jane Smith, age 25, email jane@example.com "
            "3. Bob Johnson, age 35, email bob@example.com "
            "Set total_count to 3"
        )
        
        assert isinstance(result, PersonList)
        assert result.total_count == 3
        assert len(result.people) == 3
        
        # Check first person
        assert result.people[0].name == "John Doe"
        assert result.people[0].age == 30
        assert result.people[0].email == "john@example.com"
        
        # Check second person
        assert result.people[1].name == "Jane Smith"
        assert result.people[1].age == 25
        assert result.people[1].email == "jane@example.com"
        
        # Check third person
        assert result.people[2].name == "Bob Johnson"
        assert result.people[2].age == 35
        assert result.people[2].email == "bob@example.com"

    def test_with_include_raw_success(self):
        """Test structured output with include_raw=True for successful parsing."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(Person, include_raw=True)
        
        result = structured_llm.invoke(
            "Create a person named Emma Brown, age 27, email emma@example.com"
        )
        
        assert isinstance(result, dict)
        assert "parsed" in result
        assert "raw" in result
        assert "parsing_error" in result
        
        # Check parsed result
        assert isinstance(result["parsed"], Person)
        assert result["parsed"].name == "Emma Brown"
        assert result["parsed"].age == 27
        assert result["parsed"].email == "emma@example.com"
        
        # Check raw result
        assert result["raw"] is not None
        assert hasattr(result["raw"], "content")
        
        # Check no parsing error
        assert result["parsing_error"] is None

    def test_creative_prompt_handling(self):
        """Test handling of creative prompts that might not follow exact format."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0.3,  # Slightly higher temperature for creativity
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(Person)
        
        result = structured_llm.invoke(
            "Imagine a fictional character who is a software developer. "
            "Give them a realistic name, age between 25-35, and a professional email address."
        )
        
        assert isinstance(result, Person)
        assert isinstance(result.name, str) and len(result.name) > 0
        assert 25 <= result.age <= 35
        assert "@" in result.email and "." in result.email

    def test_complex_nested_data(self):
        """Test with more complex nested data structures."""
        
        class Address(BaseModel):
            street: str
            city: str
            state: str
            zip_code: str
        
        class PersonWithAddress(BaseModel):
            name: str
            age: int
            email: str
            address: Address
        
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(PersonWithAddress)
        
        result = structured_llm.invoke(
            "Create a person named Michael Davis, age 32, email michael@example.com, "
            "living at 123 Main Street, New York, NY, 10001"
        )
        
        assert isinstance(result, PersonWithAddress)
        assert result.name == "Michael Davis"
        assert result.age == 32
        assert result.email == "michael@example.com"
        
        assert isinstance(result.address, Address)
        assert result.address.street == "123 Main Street"
        assert result.address.city == "New York"
        assert result.address.state == "NY"
        assert result.address.zip_code == "10001"

    def test_error_handling_with_impossible_constraints(self):
        """Test error handling when LLM cannot satisfy model constraints."""
        
        class StrictPerson(BaseModel):
            name: str = Field(..., min_length=1, max_length=5)  # Very short name constraint
            age: int = Field(..., ge=150, le=200)  # Impossible age range
            email: str
        
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(StrictPerson, include_raw=True)
        
        result = structured_llm.invoke(
            "Create a normal person with a regular name and age"
        )
        
        # Should return error information when include_raw=True
        assert isinstance(result, dict)
        assert "parsing_error" in result
        # Either parsing succeeds (unlikely) or fails gracefully
        assert result["raw"] is not None

    def test_message_list_input(self):
        """Test structured output with message list input."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            temperature=0,
            api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        )
        
        structured_llm = llm.with_structured_output(Person)
        
        messages = [
            HumanMessage(content="I need you to create a person profile."),
            HumanMessage(content="Name: Lisa Chen, Age: 29, Email: lisa.chen@example.com"),
        ]
        
        result = structured_llm.invoke(messages)
        
        assert isinstance(result, Person)
        assert result.name == "Lisa Chen"
        assert result.age == 29
        assert result.email == "lisa.chen@example.com"
