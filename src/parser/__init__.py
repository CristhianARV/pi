import re
import warnings
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from enum import Enum

from themes import infer_theme_from_path


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
    if isinstance(obj, Enum):
        return obj.value
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
    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_MAX_TOKENS = 1500
    _EMPTY_DOCS_ERROR = (
        "No text could be extracted from the PDF with Docling OCR or fallback extraction. "
        "The file is likely scanned/image-only or uses unsupported PDF encoding."
    )

    def __init__(
        self,
        tokenizer: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        pdf_pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            force_full_page_ocr=True,
            ocr_options=RapidOcrOptions(),
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options
                )
            }
        )

    def _load_with_docling(self, file_path: str) -> list[Document]:
        raw_md = self._converter.convert(file_path).document.export_to_markdown()
        cleaned_md = _remove_toc_and_index(raw_md)
        if not cleaned_md.strip():
            return []

        document_name = Path(file_path).name
        theme = infer_theme_from_path(file_path)

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
            meta = _safe_meta(
                chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else {}
            )
            meta["source"] = document_name
            meta["file_path"] = file_path
            meta["document_name"] = document_name
            meta["theme"] = theme
            meta["parser"] = "docling_ocr"
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def _load_with_fallback(self, file_path: str) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        document_name = Path(file_path).name
        theme = infer_theme_from_path(file_path)

        docs = []
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue

                page_metadata = _safe_meta(
                    {
                        "source": document_name,
                        "file_path": file_path,
                        "document_name": document_name,
                        "theme": theme,
                        "page": page_number,
                        "parser": "pdfplumber_fallback",
                    }
                )

                for chunk_index, chunk_text in enumerate(splitter.split_text(text)):
                    if not chunk_text.strip():
                        continue
                    metadata = dict(page_metadata)
                    metadata["chunk"] = chunk_index
                    docs.append(
                        Document(
                            page_content=chunk_text.strip(),
                            metadata=metadata,
                        )
                    )
        return docs

    def load(self, file_path: str) -> list[Document]:
        docling_error = None
        docs = []

        try:
            docs = self._load_with_docling(file_path)
        except Exception as exc:
            docling_error = exc
            warnings.warn(
                f"Docling OCR failed for '{file_path}' with {exc!r}; using fallback loader.",
                RuntimeWarning,
                stacklevel=2,
            )

        if docs:
            return docs

        warnings.warn(
            f"Docling OCR returned no chunks for '{file_path}'; using fallback loader.",
            RuntimeWarning,
            stacklevel=2,
        )

        docs = self._load_with_fallback(file_path)
        if docs:
            return docs

        message = f"{self._EMPTY_DOCS_ERROR} File: {file_path}"
        if docling_error is not None:
            raise RuntimeError(message) from docling_error
        raise RuntimeError(message)