"""LangchainDigitalocean chat models."""

import json
import logging
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from gradient import Gradient
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall, tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, model_validator
from typing_extensions import TypedDict

from .constants import ALLOWED_MODEL_FIELDS

logger = logging.getLogger(__name__)


class StreamOptions(TypedDict, total=False):
    include_usage: bool


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
    tools: Optional[List[Dict[str, Any]]] = None
    """Tools available to the model (in OpenAI format)."""
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    """Tool choice strategy: 'auto', 'none', 'required', or specific tool."""

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
        return "0.1.24"
    
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

    def _convert_message(self, msg: BaseMessage) -> Dict[str, Any]:
        """Convert a LangChain message to OpenAI API format."""
        if hasattr(msg, "type"):
            role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(
                msg.type, msg.type
            )
        else:
            role = getattr(msg, "role", "user")
        
        result: Dict[str, Any] = {"role": role, "content": msg.content}
        
        if isinstance(msg, ToolMessage):
            result["tool_call_id"] = msg.tool_call_id
        
        if isinstance(msg, AIMessage) and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in msg.tool_calls
            ]
        
        return result

    def _parse_tool_calls(self, raw_tool_calls: List[Any]) -> List[ToolCall]:
        """Parse raw tool calls from API response into LangChain format."""
        tool_calls = []
        for tc in raw_tool_calls:
            try:
                # Handle both object and dict responses
                if hasattr(tc, "function"):
                    func = tc.function
                    name = func.name if hasattr(func, "name") else func.get("name")
                    arguments = func.arguments if hasattr(func, "arguments") else func.get("arguments")
                    tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
                else:
                    func = tc.get("function", {})
                    name = func.get("name")
                    arguments = func.get("arguments")
                    tc_id = tc.get("id")
                
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
                tool_calls.append(create_tool_call(name=name, args=args, id=tc_id))
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.warning(
                    f"Failed to parse tool call, skipping. Tool call data: {tc}, Error: {e}"
                )
        return tool_calls

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

        parameters: Dict[str, Any] = {
            "messages": [self._convert_message(m) for m in messages],
            "model": self.model_name,
        }

        self._update_parameters_with_model_fields(parameters)

        if "stop_sequences" in parameters:
            parameters["stop"] = parameters.pop("stop_sequences")

        tools = kwargs.get("tools") or self.tools
        tool_choice = kwargs.get("tool_choice") or self.tool_choice
        
        if tools:
            parameters["tools"] = tools
        if tool_choice:
            parameters["tool_choice"] = tool_choice

        completion = inference_client.chat.completions.create(**parameters)
        choice = completion.choices[0]
        content = (
            choice.message.content
            if hasattr(choice.message, "content")
            else choice.message
        ) or ""  # Ensure content is never None
        
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
        
        tool_calls = []
        raw_tool_calls = getattr(choice.message, "tool_calls", None)
        if raw_tool_calls:
            tool_calls = self._parse_tool_calls(raw_tool_calls)
        
        message_kwargs: Dict[str, Any] = {
            "content": content,
            "additional_kwargs": {"refusal": getattr(choice.message, "refusal", None)},
            "response_metadata": response_metadata,
        }
        
        if tool_calls:
            message_kwargs["tool_calls"] = tool_calls
        
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

        parameters: Dict[str, Any] = {
            "messages": [self._convert_message(m) for m in messages],
            "stream": True,  # Enable streaming
            "model": self.model_name,
        }
        
        self._update_parameters_with_model_fields(parameters)

        # Handle tools from kwargs (e.g., from bind_tools)
        tools = kwargs.get("tools") or self.tools
        tool_choice = kwargs.get("tool_choice") or self.tool_choice
        
        if tools:
            parameters["tools"] = tools
        if tool_choice:
            parameters["tool_choice"] = tool_choice

        try:
            stream = inference_client.chat.completions.create(**parameters)
            for completion in stream:
                delta = completion.choices[0].delta
                
                content = getattr(delta, "content", None) or ""
                
                tool_call_chunks = []
                raw_tool_calls = getattr(delta, "tool_calls", None)
                if raw_tool_calls:
                    for tc in raw_tool_calls:
                        if hasattr(tc, "function"):
                            func = tc.function
                            name = getattr(func, "name", None)
                            arguments = getattr(func, "arguments", None)
                            tc_id = getattr(tc, "id", None)
                            index = getattr(tc, "index", None)
                        else:
                            func = tc.get("function", {})
                            name = func.get("name")
                            arguments = func.get("arguments")
                            tc_id = tc.get("id")
                            index = tc.get("index")
                        
                        tool_call_chunks.append(
                            create_tool_call_chunk(
                                name=name,
                                args=arguments,
                                id=tc_id,
                                index=index,
                            )
                        )
                
                if not content and not tool_call_chunks:
                    continue
                
                chunk_kwargs: Dict[str, Any] = {"content": content}
                if tool_call_chunks:
                    chunk_kwargs["tool_call_chunks"] = tool_call_chunks
                
                chunk = ChatGenerationChunk(message=AIMessageChunk(**chunk_kwargs))
                if run_manager and content:
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

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model. Can be:
                - OpenAI-format tool dicts
                - Pydantic models
                - Python functions
                - LangChain BaseTool instances
            tool_choice: Tool choice strategy:
                - "auto": Let the model decide (default)
                - "none": Don't use tools
                - "required": Must use a tool
                - {"type": "function", "function": {"name": "tool_name"}}: Use specific tool

        Returns:
            A Runnable that returns an AIMessage, potentially with tool_calls.

        Example:
            >>> from langchain_core.tools import tool
            >>> @tool
            ... def get_weather(location: str) -> str:
            ...     '''Get weather for a location.'''
            ...     return f"Weather in {location}: sunny"
            >>> 
            >>> llm = ChatGradient(model="llama3.3-70b-instruct")
            >>> llm_with_tools = llm.bind_tools([get_weather])
            >>> response = llm_with_tools.invoke("What's the weather in SF?")
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        formatted_tool_choice = tool_choice
        if tool_choice == "any":
            formatted_tool_choice = "required"
        elif tool_choice is None:
            formatted_tool_choice = "auto"
        
        return self.bind(
            tools=formatted_tools,
            tool_choice=formatted_tool_choice,
            **kwargs,
        )
