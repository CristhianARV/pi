import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from llm import LLMManager
from parser import DocumentParser
from rag import RAGPipeline
from themes import normalize_theme


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
        project_root = Path(__file__).resolve().parents[1]
        default_data_dir = project_root / "data"
        data_dir = Path(
            os.getenv("LOCAL_RAG_DATA_DIR", str(default_data_dir))
        ).expanduser().resolve()

        collection_name = os.getenv("LOCAL_RAG_COLLECTION_NAME", "docs_thematic")
        max_index_docs = int(os.getenv("LOCAL_RAG_MAX_INDEX_DOCS", "0"))

        print(f"[DEBUG] data_dir={data_dir}")
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
            if not data_dir.exists():
                raise FileNotFoundError(
                    f"Data directory not found: {data_dir}\n"
                    "Set LOCAL_RAG_DATA_DIR to a valid directory containing PDFs."
                )

            pdf_paths = sorted(data_dir.rglob("*.pdf"))
            if not pdf_paths:
                raise FileNotFoundError(
                    f"No PDF files found under: {data_dir}\n"
                    "Set LOCAL_RAG_DATA_DIR to a valid directory containing PDFs."
                )

            if max_index_docs > 0:
                pdf_paths = pdf_paths[:max_index_docs]

            total_chunks = 0
            for pdf_path in pdf_paths:
                print(f"[DEBUG] loading docs from {pdf_path}...")
                docs = parser.load(str(pdf_path))
                print(f"[DEBUG] parser returned {len(docs)} docs")

                if not docs:
                    continue

                vsm.add_documents(docs)
                total_chunks += len(docs)

            if total_chunks == 0:
                raise RuntimeError("Parser returned no documents to index.")

            print(f"[DEBUG] indexing done. total_chunks={total_chunks}")

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
        theme: str | None = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> LocalRAGResponse:
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline not initialized. Use LocalRAGClient inside 'async with'."
            )

        print(f"[DEBUG] querying: {question}")
        selected_theme = normalize_theme(theme) if theme is not None else None
        if selected_theme == "Unknown":
            selected_theme = None

        result = self.pipeline.ask_with_context(
            question,
            top_k=top_k,
            theme=selected_theme,
        )
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