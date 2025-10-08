# Structured Output with ChatGradient

## Overview
ChatGradient supports structured outputs using Pydantic models, similar to OpenAI's API.

## Basic Usage

from pydantic import BaseModel, Field
from langchain_gradient import ChatGradient

class Person(BaseModel):
name: str = Field(description="The person's name")
age: int = Field(description="The person's age")
email: str = Field(description="Email address")

llm = ChatGradient(model="llama3.3-70b-instruct")
structured_llm = llm.with_structured_output(Person)

response = structured_llm.invoke(
"Create a person named John, age 30, email john@example.com"
)
print(response)

Output: Person(name='John', age=30, email='john@example.com')

## Multiple Objects

from typing import List

class PersonList(BaseModel):
people: List[Person]

structured_llm = llm.with_structured_output(PersonList)
response = structured_llm.invoke("Create 3 people with different names")
print(response.people) # List of Person objects

## Error Handling

Get raw output and errors
structured_llm = llm.with_structured_output(Person, include_raw=True)

result = structured_llm.invoke("Create a person")

if result["parsing_error"]:
print(f"Error: {result['parsing_error']}")
print(f"Raw: {result['raw'].content}")
else:
print(f"Parsed: {result['parsed']}")

undefined
