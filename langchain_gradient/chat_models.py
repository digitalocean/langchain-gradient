"""LangchainDigitalocean chat models."""

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from gradient import Gradient
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
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import TypedDict

from .constants import ALLOWED_MODEL_FIELDS


class StreamOptions(TypedDict, total=False):
    include_usage: bool


class StructuredOutputError(Exception):
    """Exception raised when structured output parsing fails."""
    pass


class StructuredOutputRunnable(Runnable):
    """A runnable that wraps ChatGradient to provide structured output functionality."""
    
    def __init__(
        self,
        llm: "ChatGradient",
        schema: Type[BaseModel],
        method: str = "json_mode",
        include_raw: bool = False,
    ):
        self.llm = llm
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
    
    def invoke(
        self,
        input: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Invoke the LLM and parse the output into the structured format."""
        # Convert input to messages if it's a string
        if isinstance(input, str):
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
        else:
            messages = input
        
        # Add JSON format instruction to the last message
        if messages and self.method == "json_mode":
            last_message = messages[-1]
            schema_description = self._get_schema_description()
            enhanced_content = f"{last_message.content}\n\nPlease respond with a valid JSON object that matches this schema:\n{schema_description}"
            
            # Create a new message with enhanced content
            messages = messages[:-1] + [HumanMessage(content=enhanced_content)]
        
        # Get the raw response from the LLM
        response = self.llm.invoke(messages, config, **kwargs)
        
        # Parse the response content
        try:
            parsed_output = self._parse_response(response.content)
            
            if self.include_raw:
                return {
                    "parsed": parsed_output,
                    "raw": response,
                    "parsing_error": None,
                }
            else:
                return parsed_output
                
        except Exception as e:
            if self.include_raw:
                return {
                    "parsed": None,
                    "raw": response,
                    "parsing_error": str(e),
                }
            else:
                raise StructuredOutputError(f"Failed to parse structured output: {e}") from e
    
    def _get_schema_description(self) -> str:
        """Get a description of the Pydantic schema."""
        try:
            # Get the JSON schema
            schema_dict = self.schema.model_json_schema()
            return json.dumps(schema_dict, indent=2)
        except Exception:
            # Fallback to a simple description
            fields = []
            for field_name, field_info in self.schema.model_fields.items():
                field_type = field_info.annotation if hasattr(field_info, 'annotation') else 'Any'
                fields.append(f'"{field_name}": {field_type}')
            return "{\n  " + ",\n  ".join(fields) + "\n}"
    
    def _parse_response(self, content: str) -> BaseModel:
        """Parse the response content into the structured format."""
        # Try to extract JSON from the response
        json_str = self._extract_json(content)
        
        try:
            # Parse JSON
            json_data = json.loads(json_str)
            
            # Validate with Pydantic model
            return self.schema(**json_data)
            
        except json.JSONDecodeError as e:
            raise StructuredOutputError(f"Invalid JSON in response: {e}")
        except ValidationError as e:
            raise StructuredOutputError(f"Pydantic validation failed: {e}")
        except Exception as e:
            raise StructuredOutputError(f"Unexpected error during parsing: {e}")
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from the response content."""
        content = content.strip()
        
        # Look for JSON wrapped in code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        
        # Look for JSON wrapped in code blocks without language specification
        if "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                potential_json = content[start:end].strip()
                # Check if it looks like JSON
                if potential_json.startswith(("{", "[")):
                    return potential_json
        
        # Look for JSON-like content (starts with { or [)
        for i, char in enumerate(content):
            if char in "{[":
                # Find the matching closing bracket
                bracket_count = 0
                start_char = char
                end_char = "}" if char == "{" else "]"
                
                for j in range(i, len(content)):
                    if content[j] == start_char:
                        bracket_count += 1
                    elif content[j] == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            return content[i:j+1]
        
        # If no JSON structure found, try to parse the entire content
        return content


class ChatGradient(BaseChatModel):
    """
    ChatGradient model for DigitalOcean Gradient.

    This class provides an interface to the DigitalOcean Gradient chat completion API,
    compatible with LangChain's chat model interface. It supports both standard and streaming
    chat completions, and exposes model parameters for fine-tuning requests.

    Parameters
    ----------
    api_key : Optional[str]
        Gradient API key. If not provided, will use the DIGITALOCEAN_INFERENCE_KEY environment variable.
    model_name : str
        Model name to use. Defaults to "llama3.3-70b-instruct".
    frequency_penalty : Optional[float]
        Penalizes repeated tokens according to frequency.
    logit_bias : Optional[Dict[str, float]]
        Bias for logits.
    logprobs : Optional[bool]
        Whether to return logprobs.
    max_completion_tokens : Optional[int]
        Maximum number of tokens to generate.
    max_tokens : Optional[int]
        Maximum number of tokens to generate.
    metadata : Optional[Dict[str, str]]
        Metadata to include in the request.
    n : Optional[int]
        Number of chat completions to generate for each prompt.
    presence_penalty : Optional[float]
        Penalizes repeated tokens.
    stop : Union[Optional[str], List[str], None]
        Default stop sequences.
    streaming : Optional[bool]
        Whether to stream the results or not.
    stream_options : Optional[StreamOptions]
        Stream options. If include_usage is True, token usage metadata will be included in responses.
    temperature : Optional[float]
        What sampling temperature to use.
    top_logprobs : Optional[int]
        The number of top logprobs to return.
    top_p : Optional[float]
        Total probability mass of tokens to consider at each step.
    user : str
        A unique identifier representing the user. Defaults to "langchain-gradient".
    timeout : Optional[float]
        Timeout for requests.
    max_retries : int
        Max number of retries. Defaults to 2.

    Example
    -------
    ```python
    from langchain_core.messages import HumanMessage
    from langchain_gradient import ChatGradient

    chat = ChatGradient(model_name="llama3.3-70b-instruct")
    response = chat.invoke([
        HumanMessage(content="What is the capital of France?")
    ])
    print(response)
    ```

    Output:
    ```python
    AIMessage(content="The capital of France is Paris.")
    ```

    Methods
    -------
    _generate(messages, ...)
        Generate a chat completion for the given messages.
    _stream(messages, ...)
        Stream chat completions for the given messages.
    """
    api_key: Optional[str] = Field(
        default=os.environ.get("DIGITALOCEAN_INFERENCE_KEY"),
        exclude=True,
    )
    """Gradient model access key sourced from DIGITALOCEAN_INFERENCE_KEY."""
    model_name: str = Field(default="llama3.3-70b-instruct", alias="model")
    """Model name to use."""
    frequency_penalty: Optional[float] = None
    """Penalizes repeated tokens according to frequency."""
    logit_bias: Optional[Dict[str, float]] = None
    """Bias for logits."""
    logprobs: Optional[bool] = None
    """Whether to return logprobs."""
    max_completion_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    max_tokens: Optional[int] = Field(default=None)
    """Maximum number of tokens to generate."""
    metadata: Optional[Dict[str, str]] = None
    """Metadata to include in the request."""
    n: Optional[int] = None
    """Number of chat completions to generate for each prompt."""
    presence_penalty: Optional[float] = None
    """Penalizes repeated tokens."""
    stop: Union[Optional[str], List[str], None] = Field(
        default=None, alias="stop_sequences"
    )
    """Default stop sequences."""
    streaming: Optional[bool] = Field(default=False, alias="stream")
    """Whether to stream the results or not."""
    stream_options: Optional[StreamOptions] = None
    """Stream options."""
    temperature: Optional[float] = None
    """What sampling temperature to use."""
    top_logprobs: Optional[int] = None
    """The number of top logprobs to return."""
    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""
    user: str = "langchain-gradient"
    """A unique identifier representing the user."""
    timeout: Optional[float] = None
    """Timeout for requests."""
    max_retries: int = 2
    """Max number of retries."""

    @model_validator(mode="before")
    @classmethod
    def validate_temperature(cls, values: dict[str, Any]) -> Any:
        """Currently o1 models only allow temperature=1."""
        model = values.get("model_name") or values.get("model") or ""
        if model.startswith("o1") and "temperature" not in values:
            values["temperature"] = 1
        return values

    @property
    def user_agent_package(self) -> str:
        return f"LangChain"

    @property
    def user_agent_version(self) -> str:
        return "0.1.22"
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-gradient"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    def _update_parameters_with_model_fields(self, parameters: dict) -> None:
        # Only add explicitly supported model fields
        for field in ALLOWED_MODEL_FIELDS:
            value = getattr(self, field, None)
            # Use the alias if defined (e.g., model_name -> model)
            model_field = self.__class__.model_fields.get(field)
            key = (
                model_field.alias
                if model_field and getattr(model_field, "alias", None)
                else field
            )
            if key not in parameters and value is not None:
                parameters[key] = value

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.api_key:
            raise ValueError(
                "Gradient model access key not provided. Set DIGITALOCEAN_INFERENCE_KEY env var or pass api_key param."
            )

        inference_client = Gradient(
            model_access_key=self.api_key,
            base_url="https://inference.do-ai.run/v1",
            max_retries=self.max_retries,
            user_agent_package=self.user_agent_package,
            user_agent_version=self.user_agent_version,
        )

        def convert_message(msg: BaseMessage) -> Dict[str, Any]:
            if hasattr(msg, "type"):
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(
                    msg.type, msg.type
                )
            else:
                role = getattr(msg, "role", "user")
            return {"role": role, "content": msg.content}

        parameters: Dict[str, Any] = {
            "messages": [convert_message(m) for m in messages],
            "model": self.model_name,
        }

        self._update_parameters_with_model_fields(parameters)

        if "stop_sequences" in parameters:
            parameters["stop"] = parameters.pop("stop_sequences")

        # Only pass expected keyword arguments to create()
        completion = inference_client.chat.completions.create(**parameters)
        choice = completion.choices[0]
        content = (
            choice.message.content
            if hasattr(choice.message, "content")
            else choice.message
        )
        usage = getattr(completion, "usage", {})
        response_metadata = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "token_usage": {
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            "model_name": getattr(completion, "model", None),
            "id": getattr(completion, "id", None),
        }
        message_kwargs = {
            "content": content,
            "additional_kwargs": {"refusal": getattr(choice.message, "refusal", None)},
            "response_metadata": response_metadata,
        }
        if self.stream_options and self.stream_options.get("include_usage"):
            message_kwargs["usage_metadata"] = {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        message = AIMessage(**message_kwargs)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if not self.api_key:
            raise ValueError(
                "Gradient model access key not provided. Set DIGITALOCEAN_INFERENCE_KEY env var or pass api_key param."
            )

        inference_client = Gradient(
            model_access_key=self.api_key,
            base_url="https://inference.do-ai.run/v1",
            user_agent_package=self.user_agent_package,
            user_agent_version=self.user_agent_version, 
        )

        def convert_message(msg: BaseMessage) -> Dict[str, Any]:
            if hasattr(msg, "type"):
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(
                    msg.type, msg.type
                )
            else:
                role = getattr(msg, "role", "user")
            return {"role": role, "content": msg.content}

        parameters: Dict[str, Any] = {
            "messages": [convert_message(m) for m in messages],
            "stream": True,  # Enable streaming
            "model": self.model_name,
        }
        
        self._update_parameters_with_model_fields(parameters)

        try:
            stream = inference_client.chat.completions.create(**parameters)
            for completion in stream:
                # Extract the streamed content
                content = completion.choices[0].delta.content
                if not content:
                    continue  # skip empty chunks

                chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

            # Optionally yield usage metadata at the end if available
            if self.stream_options and self.stream_options.get("include_usage"):
                usage = getattr(completion, "usage", {})
                usage_metadata = {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                if any(v is not None for v in usage_metadata.values()):
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            usage_metadata=usage_metadata,  # type: ignore
                        )
                    )
        except Exception as e:
            # Yield an error chunk if possible
            error_chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=f"[ERROR] {str(e)}", response_metadata={"error": str(e)}
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

    def with_structured_output(
        self,
        schema: Type[BaseModel],
        *,
        method: str = "json_mode",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> StructuredOutputRunnable:
        """
        Create a runnable that returns structured output using a Pydantic model.
        
        This method creates a wrapper around the ChatGradient model that automatically
        parses the LLM's response into a structured Pydantic model.
        
        Parameters
        ----------
        schema : Type[BaseModel]
            The Pydantic model class to use for structured output validation.
        method : str, optional
            The method to use for structured output. Currently supports "json_mode".
            Defaults to "json_mode".
        include_raw : bool, optional
            Whether to include the raw LLM response along with the parsed output.
            If True, returns a dict with "parsed", "raw", and "parsing_error" keys.
            If False, returns only the parsed Pydantic model instance.
            Defaults to False.
        **kwargs : Any
            Additional keyword arguments (reserved for future use).
            
        Returns
        -------
        StructuredOutputRunnable
            A runnable that can be invoked to get structured output.
            
        Example
        -------
        ```python
        from pydantic import BaseModel
        from langchain_gradient import ChatGradient
        
        class Person(BaseModel):
            name: str
            age: int
            email: str
        
        llm = ChatGradient(model="llama3.3-70b-instruct")
        structured_llm = llm.with_structured_output(Person)
        
        response = structured_llm.invoke("Create a person named John, age 30, email john@example.com")
        # Returns: Person(name="John", age=30, email="john@example.com")
        ```
        
        Raises
        ------
        StructuredOutputError
            If the LLM response cannot be parsed into the specified schema.
        ValidationError
            If the parsed data doesn't match the Pydantic model requirements.
        """
        return StructuredOutputRunnable(
            llm=self,
            schema=schema,
            method=method,
            include_raw=include_raw,
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
