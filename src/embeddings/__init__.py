from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingsManager:
    """Manages a HuggingFace embeddings model (lazy-loaded)."""

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B" #"BAAI/bge-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._embeddings: HuggingFaceEmbeddings | None = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._embeddings
