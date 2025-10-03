"""LangchainDigitalocean chat models."""
import os
import time
import random
from typing import Any, Dict, Iterator, List, Optional, Union
from gradient import Gradient
from gradient.exceptions import APITimeoutError, APIConnectionError, APIError
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, model_validator
from typing_extensions import TypedDict
from .constants import ALLOWED_MODEL_FIELDS


class StreamOptions(TypedDict, total=False):
    include_usage: bool


class ChatGradient(BaseChatModel):
    """
    ChatGradient model for DigitalOcean Gradient.
    
    This class provides an interface to the DigitalOcean Gradient chat completion API,
    compatible with LangChain's chat model interface. It supports both standard and streaming
    chat completions, and exposes model parameters for fine-tuning requests.
    
    Enhanced with configurable timeout and retry handling using exponential backoff.
    
    Parameters
    ----------
    api_key : str, optional
        DigitalOcean Gradient API key. If not provided, will use DIGITALOCEAN_INFERENCE_KEY
        environment variable.
    model_name : str
        Name of the model to use (e.g., "llama3.3-70b-instruct").
    timeout : float, optional
        Request timeout in seconds. Defaults to 60.0 seconds.
    max_retries : int, optional
        Maximum number of retry attempts for failed requests. Defaults to 3.
    stream_options : StreamOptions, optional
        Streaming configuration options.
    **kwargs : Any
        Additional model parameters from ALLOWED_MODEL_FIELDS.
    
    Examples
    --------
    Basic usage:
    >>> from langchain_gradient import ChatGradient
    >>> chat = ChatGradient(
    ...     api_key="your_api_key",
    ...     model_name="llama3.3-70b-instruct",
    ...     timeout=30.0,
    ...     max_retries=5
    ... )
    
    With custom timeout configuration:
    >>> import httpx
    >>> chat = ChatGradient(
    ...     api_key="your_api_key", 
    ...     model_name="llama3.3-70b-instruct",
    ...     timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0)
    ... )
    """
    
    api_key: Optional[str] = Field(default=None, description="DigitalOcean Gradient API key")
    model_name: str = Field(description="Model name to use")
    timeout: Union[float, Any] = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    stream_options: Optional[StreamOptions] = Field(default=None)
    
    # Add all allowed model fields as optional parameters
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    
    _client: Optional[Gradient] = None
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"
        arbitrary_types_allowed = True
    
    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that api key exists in environment."""
        api_key = values.get("api_key") or os.getenv("DIGITALOCEAN_INFERENCE_KEY")
        if not api_key:
            raise ValueError(
                "DigitalOcean Gradient API key not found. "
                "Please provide api_key parameter or set DIGITALOCEAN_INFERENCE_KEY environment variable."
            )
        values["api_key"] = api_key
        return values
    
    @property
    def client(self) -> Gradient:
        """Get or create the Gradient client with timeout and retry configuration."""
        if self._client is None:
            self._client = Gradient(
                inference_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
        return self._client
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model used by this chat model."""
        return "gradient"
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for API calls."""
        params = {"model": self.model_name}
        
        # Add optional parameters if they are set
        for field in ALLOWED_MODEL_FIELDS:
            value = getattr(self, field, None)
            if value is not None:
                params[field] = value
        
        if self.stream_options:
            params["stream_options"] = self.stream_options
        
        return params
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format LangChain messages for Gradient API."""
        formatted_messages = []
        for message in messages:
            if hasattr(message, 'role'):
                role = message.role
            elif hasattr(message, 'type'):
                # Map LangChain message types to chat roles
                role_mapping = {
                    'human': 'user',
                    'ai': 'assistant', 
                    'system': 'system'
                }
                role = role_mapping.get(message.type, 'user')
            else:
                role = 'user'
            
            formatted_messages.append({
                "role": role,
                "content": message.content
            })
        return formatted_messages
    
    def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except APITimeoutError as e:
                last_exception = e
                if attempt == self.max_retries:
                    raise APITimeoutError(f"Request timed out after {self.max_retries} retries: {str(e)}") from e
                
                # Exponential backoff: 2^attempt + random jitter
                delay = min(60, (2 ** attempt) + random.uniform(0, 1))
                time.sleep(delay)
                
            except (APIConnectionError, APIError) as e:
                last_exception = e
                # Check if this is a retryable error
                if self._is_retryable_error(e):
                    if attempt == self.max_retries:
                        raise type(e)(f"Request failed after {self.max_retries} retries: {str(e)}") from e
                    
                    # Exponential backoff
                    delay = min(60, (2 ** attempt) + random.uniform(0, 1))
                    time.sleep(delay)
                else:
                    # Non-retryable error, raise immediately
                    raise
            
            except Exception as e:
                # For other exceptions, don't retry
                raise
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable based on Gradient SDK patterns."""
        if isinstance(error, APITimeoutError):
            return True
        
        if isinstance(error, APIConnectionError):
            return True
            
        if hasattr(error, 'status_code'):
            # Retry on 408 Request Timeout, 409 Conflict, 429 Rate Limit, >=500 Internal errors
            retryable_codes = {408, 409, 429} | set(range(500, 600))
            return error.status_code in retryable_codes
        
        return False
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion response."""
        formatted_messages = self._format_messages(messages)
        params = self._get_model_params()
        
        if stop:
            params["stop"] = stop
        
        # Override with any additional kwargs
        params.update(kwargs)
        
        def _make_request():
            return self.client.chat.completions.create(
                messages=formatted_messages,
                **params
            )
        
        try:
            completion = self._execute_with_retry(_make_request)
            
            # Extract the response
            if completion.choices:
                message_content = completion.choices[0].message.content
                ai_message = AIMessage(content=message_content)
                generation = ChatGeneration(message=ai_message)
                
                # Add usage metadata if available
                llm_output = {}
                if hasattr(completion, 'usage') and completion.usage:
                    llm_output["usage"] = {
                        "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                        "total_tokens": getattr(completion.usage, "total_tokens", None),
                    }
                
                return ChatResult(generations=[generation], llm_output=llm_output)
            else:
                # No choices returned
                ai_message = AIMessage(content="")
                generation = ChatGeneration(message=ai_message)
                return ChatResult(generations=[generation])
                
        except Exception as e:
            # Handle any remaining errors
            error_message = f"Gradient API error: {str(e)}"
            ai_message = AIMessage(content=f"[ERROR] {error_message}")
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion responses."""
        formatted_messages = self._format_messages(messages)
        params = self._get_model_params()
        params["stream"] = True
        
        if stop:
            params["stop"] = stop
        
        # Override with any additional kwargs
        params.update(kwargs)
        
        def _make_stream_request():
            return self.client.chat.completions.create(
                messages=formatted_messages,
                **params
            )
        
        try:
            stream = self._execute_with_retry(_make_stream_request)
            
            for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and choice.delta:
                        content = getattr(choice.delta, 'content', '') or ''
                        if content:
                            ai_chunk = AIMessageChunk(content=content)
                            yield ChatGenerationChunk(message=ai_chunk)
            
            # Optionally yield usage metadata at the end if available
            if self.stream_options and self.stream_options.get("include_usage"):
                # This would be handled by the stream completion itself in real usage
                pass
                        
        except Exception as e:
            # Yield an error chunk if possible
            error_chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=f"[ERROR] {str(e)}", 
                    response_metadata={"error": str(e)}
                )
            )
            yield error_chunk
    
    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        # env_vars, model_params, expected_attrs
        # Map DIGITALOCEAN_INFERENCE_KEY -> api_key, and require model param
        return (
            {"DIGITALOCEAN_INFERENCE_KEY": "test-env-key"},
            {"model": "bird-brain-001", "buffer_length": 50},
            {"api_key": "test-env-key", "model_name": "bird-brain-001"},
        )
    
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True
    
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Exclude sensitive credentials from serialization
        state.pop("api_key", None)
        return state
    
    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
