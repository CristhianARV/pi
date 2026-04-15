from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """Manages a Qdrant vector store collection."""

    DEFAULT_URL = "http://localhost:6333"
    DEFAULT_COLLECTION = "docs"

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = DEFAULT_COLLECTION,
        url: str = DEFAULT_URL,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self._client = QdrantClient(
            url=url,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )
        self._store: QdrantVectorStore | None = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self.collection_name):
            vector_size = len(self.embeddings.embed_query("sample text"))
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def get_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore(
                client=self._client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        return self._store

    def add_documents(self, docs: list[Document]) -> list[str]:
        """Index documents and return their ids."""
        return self.get_store().add_documents(docs)

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        return self.get_store().similarity_search(query, k=k)
