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

More features coming soon.
