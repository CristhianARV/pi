import re
from pathlib import Path

from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from transformers import AutoTokenizer

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Headings qui marquent un chunk comme inutile pour le RAG
USELESS_HEADINGS = {
    "table of contents",
    "contents",
    "index",
    "keyword",
    "keywords",
    "overview",
    "appendix",
    "list of figures",
    "list of tables",
    "preface",
    "general and safety instructions",
}


def _is_useless_chunk(doc) -> bool:
    text = doc.page_content.strip()
    dl_meta = doc.metadata.get("dl_meta", {})

    headings = [h.strip().lower() for h in dl_meta.get("headings", [])]

    # 1) Un des headings du chunk est dans la liste noire
    if any(h in USELESS_HEADINGS for h in headings):
        return True

    # 2) Docling a explicitement étiqueté un item comme index
    doc_items = dl_meta.get("doc_items", [])
    if any(item.get("label") == "document_index" for item in doc_items):
        return True

    # 3) Sécurité : le texte lui-même commence par un de ces titres
    text_lower = text.lower()
    if any(text_lower.startswith(h) for h in USELESS_HEADINGS):
        return True

    if text.startswith("P\n"):
        return True

    return False


def _clean_docling_text(text: str) -> str:
    """Strip parser artifacts from Docling-extracted text."""
    text = re.sub(r"\[\s*}\s*\d+\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def clean_metadata(doc):
    dl_meta = doc.metadata.get("dl_meta", {})
    doc_items = dl_meta.get("doc_items", [])
    page_no = (doc_items[0].get("prov", [{}])[0].get("page_no")
               if doc_items else None)
    return {
        "filename": dl_meta.get("origin", {}).get("filename"),
        "page_no": page_no,
        "heading": dl_meta.get("headings"),
    }

class DocumentParser:
    """Loads and chunks documents using Docling via langchain-docling.

    Pipeline: file → DoclingLoader (HybridChunker) → drop TOC/index chunks → clean text.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_MAX_TOKENS = 700
    SUPPORTED_EXTENSIONS = frozenset({
        ".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt",
    })

    def __init__(
        self,
        tokenizer: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.tokenizer_id = tokenizer
        self.max_tokens = max_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = False

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    def load(self, file_path: str | Path) -> list[Document]:
        """Load and chunk a single document file into LangChain Documents."""
        file_path = str(file_path)
        chunker = HybridChunker(
            tokenizer=self._tokenizer,
            max_tokens=self.max_tokens,
            merge_peers=False,
            repeat_table_header=True,
            omit_header_on_overflow=True,
        )

        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=chunker,
            converter=self._converter,
        )
        raw_docs = loader.load()

        docs: list[Document] = []
        for raw in raw_docs:
            if _is_useless_chunk(raw):
                continue
            cleaned = _clean_docling_text(raw.page_content)
            if not cleaned:
                continue

            meta = clean_metadata(raw)
            docs.append(
                Document(
                    page_content=cleaned, 
                    metadata=meta
                )
            )
        return docs

    def load_folder(
        self,
        folder_path: str | Path,
        recursive: bool = False,
    ) -> list[Document]:
        """Load and chunk every supported file in a folder into one flat list.

        Typical usage: one folder == one Qdrant collection.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        files = folder.rglob("*") if recursive else folder.iterdir()
        all_docs: list[Document] = []
        for file in sorted(files):
            if not file.is_file():
                continue
            if file.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            try:
                all_docs.extend(self.load(file))
            except Exception as exc:
                print(f"[DocumentParser] skipped {file.name}: {exc}")
        return all_docs

    def load_subfolders(
        self,
        parent_folder: str | Path,
        recursive: bool = False,
    ) -> dict[str, list[Document]]:
        """For a parent folder containing subfolders, returns {subfolder_name: docs}.

        Each entry is meant to feed one Qdrant collection.
        """
        parent = Path(parent_folder)
        if not parent.is_dir():
            raise ValueError(f"Not a directory: {parent}")

        return {
            sub.name: self.load_folder(sub, recursive=recursive)
            for sub in sorted(parent.iterdir())
            if sub.is_dir()
        }