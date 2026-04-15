from langchain_docling.loader import DoclingLoader, ExportType
from langchain_core.documents import Document
from docling.chunking import HybridChunker


class DocumentParser:
    """Loads and chunks documents using Docling with HybridChunker."""

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_MAX_TOKENS = 300

    def __init__(
        self,
        tokenizer: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def load(self, file_path: str) -> list[Document]:
        """Load and chunk a document file (PDF, DOCX, …) into LangChain Documents."""
        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer=self.tokenizer,
                max_tokens=self.max_tokens,
                merge_peers=True,
                repeat_table_header=True,
                omit_header_on_overflow=True,
            ),
        )
        return loader.load()
