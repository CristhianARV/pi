from typing import Sequence
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from copy import copy


class Reranker:
    """CrossEncoder reranker. Scores (query, doc) pairs and keeps top-k."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: Sequence[Document],
        top_k: int = 10,
    ) -> list[Document]:
        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        out = []
        for doc, score in top:
            new_doc = copy(doc)
            new_doc.metadata = {**doc.metadata, "rerank_score": float(score)}
            out.append(doc)
        return out