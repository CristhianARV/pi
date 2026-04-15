from langchain_ollama import ChatOllama


class LLMManager:
    """Manages a remote Ollama LLM connection."""

    DEFAULT_BASE_URL = "http://157.26.83.15/ollama/"
    DEFAULT_MODEL = "qwen3:4b"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url
        self._llm: ChatOllama | None = None

    def get_llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = ChatOllama(model=self.model, base_url=self.base_url)
        return self._llm
