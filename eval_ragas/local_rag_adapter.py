import os
from pathlib import Path
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
    Adapter direct vers le RAG local.
    Appelle directement le pipeline Python local.
    """
    def __init__(self) -> None:
        self.pipeline: Optional[RAGPipeline] = None

    def setup(self) -> None:
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        default_manual_path = project_root / "data" / "Manuals" / "mds_axis_compensation_en.pdf"
        manual_path = Path(
            os.getenv("LOCAL_RAG_MANUAL_PATH", str(default_manual_path))
        ).expanduser().resolve()

        collection_name = os.getenv("LOCAL_RAG_COLLECTION_NAME", "docs_eval_one")
        max_index_docs = int(os.getenv("LOCAL_RAG_MAX_INDEX_DOCS", "5"))

        print(f"[DEBUG] manual_path={manual_path}")
        print(f"[DEBUG] collection_name={collection_name}")
        print(f"[DEBUG] max_index_docs={max_index_docs}")

        embeddings = EmbeddingsManager().get_embeddings()
        vsm = VectorStoreManager(
            embeddings=embeddings,
            collection_name=collection_name,
        )
        llm = LLMManager().get_llm()
        parser = DocumentParser()

        self.pipeline = RAGPipeline(
            vector_store_manager=vsm,
            llm=llm,
        )

        try:
            collection_info = vsm._client.get_collection(collection_name)
            points_count = collection_info.points_count
        except Exception:
            points_count = 0

        print(f"[DEBUG] existing points_count={points_count}")

        if points_count == 0:
            if not manual_path.exists():
                raise FileNotFoundError(
                    f"Manual not found: {manual_path}\n"
                    "Set LOCAL_RAG_MANUAL_PATH to a valid PDF path."
                )

            print("[DEBUG] loading docs...")
            docs = parser.load(str(manual_path))
            print(f"[DEBUG] parser returned {len(docs)} docs")

            docs = docs[:max_index_docs]
            print(f"[DEBUG] indexing {len(docs)} docs")

            if not docs:
                raise RuntimeError("Parser returned no documents to index.")

            vsm.add_documents(docs)
            print("[DEBUG] indexing done.")

        self.pipeline.build_agent()
        print("[DEBUG] agent ready.")

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

        print(f"[DEBUG] querying: {question}")
        result = self.pipeline.ask_with_context(question, top_k=top_k)
        print("[DEBUG] query done.")

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