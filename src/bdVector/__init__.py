from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, SparseVectorParams, VectorParams
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode


class VectorStoreManager:
    DEFAULT_URL = "http://localhost:6333"
    DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
    DEFAULT_VECTOR_SIZE = 1024

    SEARCH_MODE_ALIASES = {
        "semantic": RetrievalMode.DENSE,
        "dense": RetrievalMode.DENSE,
        "text": RetrievalMode.SPARSE,
        "sparse": RetrievalMode.SPARSE,
        "hybrid": RetrievalMode.HYBRID,
}

    def __init__(self, embeddings, collection_name, url=DEFAULT_URL,
                 sparse_embeddings=None, vector_size=DEFAULT_VECTOR_SIZE,
                 force_recreate=False):
        self.embeddings = embeddings
        # IMPORTANT: injection externe pour partager entre collections
        self.sparse_embeddings = sparse_embeddings or FastEmbedSparse(
            model_name=self.DEFAULT_SPARSE_MODEL
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = QdrantClient(url=url)
        self._ensure_collection(force_recreate)

    @classmethod
    def normalize_search_mode(cls, mode):
        if isinstance(mode, RetrievalMode):
            return mode
        key = mode.lower() if isinstance(mode, str) else mode
        if key in cls.SEARCH_MODE_ALIASES:
            return cls.SEARCH_MODE_ALIASES[key]
        try:
            return RetrievalMode(key)
        except ValueError:
            raise ValueError(
                f"Unknown search mode: {mode!r}. "
                f"Expected one of: {list(cls.SEARCH_MODE_ALIASES)}"
            )

    def _ensure_collection(self, force_recreate):
        if self._client.collection_exists(self.collection_name):
            if not force_recreate:
                return
            self._client.delete_collection(self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": VectorParams(size=self.vector_size,
                                                  distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    def get_store(self, mode=RetrievalMode.HYBRID):
        mode = self.normalize_search_mode(mode)
        return QdrantVectorStore(
            client=self._client,
            collection_name=self.collection_name,
            retrieval_mode=mode,
            vector_name="dense",
            sparse_vector_name="sparse",
            embedding=self.embeddings if mode != RetrievalMode.SPARSE else None,
            sparse_embedding=self.sparse_embeddings if mode != RetrievalMode.DENSE else None,
        )

    def add_documents(self, docs):
        return self.get_store(RetrievalMode.HYBRID).add_documents(docs)

    def search(self, query, mode=RetrievalMode.HYBRID, k=10):
        return self.get_store(mode).similarity_search(query, k=k)

    @classmethod
    def list_collections(cls, url=DEFAULT_URL):
        return [c.name for c in QdrantClient(url=url).get_collections().collections]
