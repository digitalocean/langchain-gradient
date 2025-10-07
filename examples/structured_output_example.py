"""
Example demonstrating structured output functionality with ChatGradient.

This example shows how to use the with_structured_output() method to get
structured responses from the ChatGradient model using Pydantic models.
"""

import os
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_gradient import ChatGradient


# Define Pydantic models for structured output
class Person(BaseModel):
    """A person model."""
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
    """A company model."""
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
    """A product review model with validation."""
    product_name: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    review_text: str
    reviewer_name: str
    would_recommend: bool


def main():
    """Run structured output examples."""
    # Initialize ChatGradient
    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        temperature=0,
        api_key=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
    )
    
    print("🚀 ChatGradient Structured Output Examples\n")
    
    # Example 1: Basic Person Creation
    print("📝 Example 1: Basic Person Creation")
    print("-" * 40)
    
    structured_llm = llm.with_structured_output(Person)
    person = structured_llm.invoke(
        "Create a person named John Smith, age 30, email john.smith@example.com"
    )
    
    print(f"Created person: {person}")
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Email: {person.email}\n")
    
    # Example 2: Person with Optional Fields
    print("📝 Example 2: Person with Optional Fields")
    print("-" * 40)
    
    structured_llm_optional = llm.with_structured_output(PersonWithOptional)
    person_optional = structured_llm_optional.invoke(
        "Create a person named Alice Johnson, age 28, occupation Software Engineer"
    )
    
    print(f"Created person: {person_optional}")
    print(f"Name: {person_optional.name}")
    print(f"Age: {person_optional.age}")
    print(f"Email: {person_optional.email}")
    print(f"Occupation: {person_optional.occupation}\n")
    
    # Example 3: Company Creation
    print("📝 Example 3: Company Creation")
    print("-" * 40)
    
    structured_llm_company = llm.with_structured_output(Company)
    company = structured_llm_company.invoke(
        "Create a tech company called InnovateTech, founded in 2015, "
        "with 250 employees, headquartered in Austin, Texas"
    )
    
    print(f"Created company: {company}")
    print(f"Name: {company.name}")
    print(f"Industry: {company.industry}")
    print(f"Founded: {company.founded_year}")
    print(f"Employees: {company.employees}")
    print(f"Headquarters: {company.headquarters}\n")
    
    # Example 4: Multiple People (List)
    print("📝 Example 4: Multiple People Creation")
    print("-" * 40)
    
    structured_llm_list = llm.with_structured_output(PersonList)
    people_list = structured_llm_list.invoke(
        "Create a list of 2 people: "
        "1. Emma Davis, age 26, email emma@example.com "
        "2. Michael Brown, age 31, email michael@example.com "
        "Set total_count to 2"
    )
    
    print(f"Created people list: {people_list}")
    print(f"Total count: {people_list.total_count}")
    for i, person in enumerate(people_list.people, 1):
        print(f"Person {i}: {person.name}, {person.age}, {person.email}")
    print()
    
    # Example 5: Product Review with Validation
    print("📝 Example 5: Product Review with Validation")
    print("-" * 40)
    
    structured_llm_review = llm.with_structured_output(ProductReview)
    review = structured_llm_review.invoke(
        "Create a product review for MacBook Pro, rating 4 out of 5, "
        "review text 'Excellent performance and build quality', "
        "reviewer name Sarah Wilson, would recommend true"
    )
    
    print(f"Created review: {review}")
    print(f"Product: {review.product_name}")
    print(f"Rating: {review.rating}/5")
    print(f"Review: {review.review_text}")
    print(f"Reviewer: {review.reviewer_name}")
    print(f"Would recommend: {review.would_recommend}\n")
    
    # Example 6: Using include_raw=True
    print("📝 Example 6: Including Raw Response")
    print("-" * 40)
    
    structured_llm_raw = llm.with_structured_output(Person, include_raw=True)
    result_with_raw = structured_llm_raw.invoke(
        "Create a person named David Lee, age 33, email david@example.com"
    )
    
    print("Result with raw response:")
    print(f"Parsed: {result_with_raw['parsed']}")
    print(f"Raw response content: {result_with_raw['raw'].content[:100]}...")
    print(f"Parsing error: {result_with_raw['parsing_error']}\n")
    
    # Example 7: Error Handling
    print("📝 Example 7: Error Handling")
    print("-" * 40)
    
    try:
        # This might fail if the LLM doesn't provide valid JSON
        structured_llm_error = llm.with_structured_output(Person, include_raw=False)
        error_result = structured_llm_error.invoke(
            "Just say hello, don't create any person data"
        )
        print(f"Unexpected success: {error_result}")
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")
    
    # Same example but with include_raw=True to see the error gracefully
    structured_llm_error_raw = llm.with_structured_output(Person, include_raw=True)
    error_result_raw = structured_llm_error_raw.invoke(
        "Just say hello, don't create any person data"
    )
    
    print("Error handling with include_raw=True:")
    print(f"Parsed: {error_result_raw['parsed']}")
    print(f"Parsing error: {error_result_raw['parsing_error']}")
    print(f"Raw response: {error_result_raw['raw'].content[:100]}...")
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    if not os.environ.get("DIGITALOCEAN_INFERENCE_KEY"):
        print("❌ Please set the DIGITALOCEAN_INFERENCE_KEY environment variable")
        exit(1)
    
    main()
