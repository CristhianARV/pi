from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings


def get_vector_store(
    embeddings: Embeddings,
    collection_name: str = "test",
    location: str = ":memory:",
) -> QdrantVectorStore:
    """Create a Qdrant vector store with the given embeddings."""
    client = QdrantClient(location)
    vector_size = len(embeddings.embed_query("sample text"))

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


