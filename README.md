# langchain-gradient  
[![PyPI Downloads](https://static.pepy.tech/badge/langchain-gradient)](https://pepy.tech/projects/langchain-gradient)

This package contains the LangChain integration with DigitalOcean

## Installation

```bash
pip install -U langchain-gradient
```

And you should configure credentials by setting the `DIGITALOCEAN_INFERENCE_KEY` environment variable:

1. Log in to the DigitalOcean Cloud console
2. Go to the **Gradient Platform** and navigate to **Serverless Inference**.
2. Click on **Create model access key**, enter a name, and create the key.
3. Use the generated key as your `DIGITALOCEAN_INFERENCE_KEY`:   


Create .env file with your access key:  
```DIGITALOCEAN_INFERENCE_KEY=your_access_key_here```

## Chat Models

`ChatGradient` class exposes chat models from langchain-gradient.

### Invoke

```python
import os
from dotenv import load_dotenv
from langchain_gradient import ChatGradient

load_dotenv()

llm = ChatGradient(
    model="llama3.3-70b-instruct",
    api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY")
)

result = llm.invoke("What is the capital of France?.")
print(result)
```

### Stream

```python
import os
from dotenv import load_dotenv
from langchain_gradient import ChatGradient

load_dotenv()

llm = ChatGradient(
    model="llama3.3-70b-instruct",
    api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY")
)

for chunk in llm.stream("Tell me what happened to the Dinosaurs?"):
    print(chunk.content, end="", flush=True)
```

### Structured Output

`ChatGradient` supports structured output using Pydantic models, similar to OpenAI's `with_structured_output()` method. This feature automatically parses LLM responses into validated Pydantic model instances.

```python
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_gradient import ChatGradient

load_dotenv()

# Define your Pydantic model
class Person(BaseModel):
    name: str
    age: int
    email: str

llm = ChatGradient(
    model="llama3.3-70b-instruct",
    api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY")
)

# Create structured output runnable
structured_llm = llm.with_structured_output(Person)

# Get structured response
response = structured_llm.invoke("Create a person named John, age 30, email john@example.com")
print(response)
# Returns: Person(name="John", age=30, email="john@example.com")

# Access structured data
print(f"Name: {response.name}")
print(f"Age: {response.age}")
print(f"Email: {response.email}")
```

#### Advanced Structured Output Features

**Include Raw Response:**
```python
# Get both parsed and raw response
structured_llm = llm.with_structured_output(Person, include_raw=True)
result = structured_llm.invoke("Create a person named Alice, age 25")

print(result["parsed"])  # Person instance
print(result["raw"])     # Original AIMessage
print(result["parsing_error"])  # None if successful
```

**Complex Models:**
```python
from typing import List, Optional
from pydantic import Field

class Company(BaseModel):
    name: str
    industry: str
    founded_year: int
    employees: int = Field(..., gt=0)
    headquarters: str

class PersonList(BaseModel):
    people: List[Person]
    total_count: int

# Works with complex nested structures
structured_llm = llm.with_structured_output(Company)
company = structured_llm.invoke("Create a tech company founded in 2010 with 500 employees")
```

**Error Handling:**
```python
from langchain_gradient import StructuredOutputError

try:
    result = structured_llm.invoke("Just say hello")  # Won't match Person schema
except StructuredOutputError as e:
    print(f"Parsing failed: {e}")

# Or use include_raw=True for graceful error handling
structured_llm = llm.with_structured_output(Person, include_raw=True)
result = structured_llm.invoke("Just say hello")
if result["parsing_error"]:
    print(f"Error: {result['parsing_error']}")
    print(f"Raw response: {result['raw'].content}")
```

#### Supported Features

- ✅ Pydantic model validation
- ✅ JSON parsing from various formats (code blocks, plain text)
- ✅ Nested models and complex data structures
- ✅ Optional fields and default values
- ✅ Field validation and constraints
- ✅ List and array handling
- ✅ Graceful error handling
- ✅ Raw response access
- ✅ Type safety

More features coming soon.
