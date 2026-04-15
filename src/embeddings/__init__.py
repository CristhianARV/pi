from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> HuggingFaceEmbeddings:
    """Initialize and return HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name=model_name)
