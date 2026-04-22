import re

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from langchain_core.documents import Document


def _remove_toc_and_index(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[\s*}\s*\d+\s*\]", "", text)
    text = re.sub(
        r"## Table of contents.*?(?=\n## Overview of compensation parameters\b)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"## Keyword index.*?(?=\n## 4 Appendix\b)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"<!-- image -->", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)


def _safe_meta(obj):
    """Recursively sanitize metadata so Qdrant gRPC can serialize it."""
    if isinstance(obj, dict):
        return {k: _safe_meta(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_meta(v) for v in obj]
    if isinstance(obj, int) and not isinstance(obj, bool):
        if obj > _INT64_MAX or obj < _INT64_MIN:
            return str(obj)
    return obj


class _MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class DocumentParser:
    """Loads and chunks documents using Docling with HybridChunker.

    Pipeline: PDF → markdown → clean (TOC/index removal) → re-parse → chunk.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_MAX_TOKENS = 1500

    def __init__(
        self,
        tokenizer: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._converter = DocumentConverter()

    def load(self, file_path: str) -> list[Document]:
        """Load and chunk a document file (PDF, DOCX, …) into LangChain Documents."""
        raw_md = self._converter.convert(file_path).document.export_to_markdown()
        cleaned_md = _remove_toc_and_index(raw_md)
        cleaned_doc = self._converter.convert_string(
            cleaned_md,
            format=InputFormat.MD,
        ).document

        chunker = HybridChunker(
            tokenizer=self.tokenizer,
            serializer_provider=_MDTableSerializerProvider(),
            max_tokens=self.max_tokens,
            merge_peers=True,
            repeat_table_header=True,
            omit_header_on_overflow=True,
        )

        raw_chunks = list(chunker.chunk(dl_doc=cleaned_doc))
        docs = []
        for chunk in raw_chunks:
            text = chunker.contextualize(chunk=chunk).strip()
            if not text:
                continue
            meta = _safe_meta(chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else {})
            meta["source"] = file_path
            docs.append(Document(page_content=text, metadata=meta))
        return docs
