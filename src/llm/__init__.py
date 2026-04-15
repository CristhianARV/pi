import os
from langchain.chat_models import init_chat_model


def get_llm(model_name: str = "claude-sonnet-4-6"):
    """Initialize and return a chat model."""
    return init_chat_model(model_name)
