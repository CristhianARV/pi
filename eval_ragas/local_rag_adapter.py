import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from bdVector import VectorStoreManager
from models import make_embeddings, make_sparse_embeddings, make_llm
from parser import DocumentParser
from rag import RAGPipeline


def normalize_theme(theme: str | None) -> str | None:
    if theme is None:
        return None

    value = str(theme).strip()
    if not value:
        return None

    if value.lower() in {"unknown", "none", "nan", "null"}:
        return None

    canonical_themes = {
        "commissioning": "Commissioning",
        "common_documents": "Common_documents",
        "common documents": "Common_documents",
        "cycles": "Cycles",
        "functional_description": "Functional_description",
        "functional description": "Functional_description",
        "interfaces": "Interfaces",
        "manuals": "Manuals",
        "plc-libraries": "PLC-libraries",
        "plc_libraries": "PLC-libraries",
        "plc libraries": "PLC-libraries",
        "troubleshooting": "troubleshooting",
        "operator_support": "operator_support",
        "operator support": "operator_support",
        "safety_guardrail": "safety_guardrail",
        "safety guardrail": "safety_guardrail",
        "unanswerable": "unanswerable",
    }

    return canonical_themes.get(value.lower(), value)


@dataclass
class LocalRAGResponse:
    question: str
    answer: str
    retrieved_contexts: List[str]
    retrieved_context_ids: List[str]
    retrieved_context_metadata: List[Dict[str, Any]]
    latency_ms: float
    raw_payload: Dict[str, Any]


class LocalRAGClient:
    """
    Adapter direct vers le RAG local V2.
    Utilisé par le module RAGAS pour appeler le pipeline Python local.
    """

    def __init__(self) -> None:
        self.pipeline: Optional[RAGPipeline] = None

    def setup(self) -> None:
        default_data_dir = PROJECT_ROOT / "data"

        data_dir = Path(
            os.getenv("LOCAL_RAG_DATA_DIR", str(default_data_dir))
        ).expanduser().resolve()

        collection_name = os.getenv(
            "LOCAL_RAG_COLLECTION_NAME",
            "docs_thematic",
        )

        max_index_docs = int(os.getenv("LOCAL_RAG_MAX_INDEX_DOCS", "0"))
        search_mode = os.getenv("LOCAL_RAG_SEARCH_MODE", "semantic")
        top_k = int(os.getenv("LOCAL_RAG_TOP_K", "5"))
        fetch_k = int(os.getenv("LOCAL_RAG_FETCH_K", "50"))
        use_reranker = os.getenv("LOCAL_RAG_USE_RERANKER", "0") == "1"

        print(f"[DEBUG] data_dir={data_dir}")
        print(f"[DEBUG] collection_name={collection_name}")
        print(f"[DEBUG] max_index_docs={max_index_docs}")
        print(f"[DEBUG] search_mode={search_mode}")
        print(f"[DEBUG] top_k={top_k}")

        embeddings = make_embeddings()
        sparse_embeddings = make_sparse_embeddings()

        vsm = VectorStoreManager(
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            collection_name=collection_name,
        )

        llm = make_llm()
        parser = DocumentParser()

        self.pipeline = RAGPipeline(
            vector_stores={collection_name: vsm},
            llm=llm,
            default_collection=collection_name,
            fetch_k=fetch_k,
            top_k=top_k,
            use_reranker=use_reranker,
            search_mode=search_mode,
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

                ids = vsm.add_documents(docs)
                total_chunks += len(ids)

            if total_chunks == 0:
                raise RuntimeError("Parser returned no documents to index.")

            print(f"[DEBUG] indexing done. total_chunks={total_chunks}")

        print("[DEBUG] RAG V2 pipeline ready.")

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

        selected_theme = normalize_theme(theme)

        print(f"[DEBUG] querying: {question}")
        print(f"[DEBUG] selected_theme={selected_theme}")

        result = self.pipeline.ask_with_context(
            query=question,
            top_k=top_k,
        )

        print("[DEBUG] query done.")

        return LocalRAGResponse(
            question=question,
            answer=result.get("response", ""),
            retrieved_contexts=result.get("retrieved_contexts", []),
            retrieved_context_ids=[
                str(x) for x in result.get("retrieved_context_ids", [])
            ],
            retrieved_context_metadata=result.get("retrieved_context_metadata", []),
            latency_ms=float(result.get("latency_ms", 0.0)),
            raw_payload=result,
        )