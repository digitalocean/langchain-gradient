# Constants for langchain_gradient

ALLOWED_MODEL_FIELDS = {
    "model_name",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "n",
    "presence_penalty",
    "stop",
    "streaming",
    "temperature",
    "top_logprobs",
    "top_p",
    "user",
    # timeout is a client transport setting only — do not send it in the JSON body
    "stream_options",
    "tools",
    "tool_choice",
    # Add new fields here as needed
}
