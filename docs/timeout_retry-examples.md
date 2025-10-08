# Timeout and Retry Configuration

## Basic Usage

from langchain_gradient import ChatGradient
from langchain_gradient.exceptions import APITimeoutError, APIRetryError

Initialize with timeout and retry configuration
llm = ChatGradient(
model="llama3.3-70b-instruct",
timeout=30.0,
max_retries=3
)

try:
response = llm.invoke("Hello world!")
print(response)
except APITimeoutError as e:
print(f"Request timed out: {e}")
except APIRetryError as e:
print(f"Max retries exceeded: {e}")


## Advanced Configuration

from langchain_gradient import ChatGradient
from langchain_gradient.retry_config import RetryConfig

Custom retry configuration with exponential backoff
custom_retry = RetryConfig(
max_retries=5,
initial_delay=1.0,
max_delay=60.0,
exponential_base=2.0,
jitter=True
)

llm = ChatGradient(
model="llama3.3-70b-instruct",
timeout=60.0,
retry_config=custom_retry
)


## Exponential Backoff

The retry mechanism uses exponential backoff with jitter to prevent retry storms:
- Delay = initial_delay × (base^attempt)
- Random jitter (±20%) added by default
- Maximum delay capped at max_delay
