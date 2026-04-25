import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from llm import LLMManager
from parser import DocumentParser
from rag import RAGPipeline


@dataclass
class LocalRAGResponse:
    question: str
    answer: str
    retrieved_contexts: List[str]
    retrieved_context_ids: List[str]
    latency_ms: float
    raw_payload: Dict[str, Any]


class LocalRAGClient:
    """
    Adapter direct vers le RAG local du repo `pi`.
    Cette version appelle directement le pipeline Python local.
    """

    def __init__(self) -> None:
        self.pipeline: Optional[RAGPipeline] = None

    def setup(self) -> None:
        embeddings = EmbeddingsManager().get_embeddings()
        vsm = VectorStoreManager(
            embeddings=embeddings,
            collection_name="docs",
        )
        llm = LLMManager().get_llm()
        parser = DocumentParser()

        self.pipeline = RAGPipeline(
            vector_store_manager=vsm,
            llm=llm,
        )

        file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "Manuals",
            "mds_axis_compensation_en.pdf",
        )

        collection_info = vsm._client.get_collection("docs")
        if collection_info.points_count == 0:
            self.pipeline.index_file(file_path, parser)

        self.pipeline.build_agent()

    async def __aenter__(self) -> "LocalRAGClient":
        self.setup()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def aquery(
        self,
        question: str,
        top_k: int = 5,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> LocalRAGResponse:
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline not initialized. Use LocalRAGClient inside 'async with'."
            )

        result = self.pipeline.ask_with_context(question, top_k=top_k)

        return LocalRAGResponse(
            question=question,
            answer=result.get("response", ""),
            retrieved_contexts=result.get("retrieved_contexts", []),
            retrieved_context_ids=[
                str(x) for x in result.get("retrieved_context_ids", [])
            ],
            latency_ms=float(result.get("latency_ms", 0.0)),
            raw_payload=result,
        )