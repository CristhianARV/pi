import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console

from llm import LLMManager
from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from parser import DocumentParser
from rag import RAGPipeline
from interface_graphique import ChatApp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COLLECTION = os.getenv("LOCAL_RAG_COLLECTION_NAME", "docs_thematic")


def run():
    console = Console()

    with console.status("[cyan]Loading embeddings model…[/cyan]"):
        embeddings = EmbeddingsManager().get_embeddings()

    with console.status("[cyan]Connecting to Qdrant…[/cyan]"):
        vsm = VectorStoreManager(embeddings=embeddings, collection_name=COLLECTION)

    with console.status("[cyan]Initialising LLM…[/cyan]"):
        llm = LLMManager().get_llm()

    parser = DocumentParser()
    pipeline = RAGPipeline(vector_store_manager=vsm, llm=llm)

    collection_info = vsm._client.get_collection(COLLECTION)
    if collection_info.points_count == 0:
        pdf_paths = sorted(Path(DATA_DIR).rglob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found under: {DATA_DIR}")

        total_chunks = 0
        with console.status("[cyan]Indexing documents…[/cyan]"):
            for pdf_path in pdf_paths:
                total_chunks += pipeline.index_file(str(pdf_path), parser)
        console.print(f"[green]Indexed {total_chunks} chunks from {len(pdf_paths)} files.[/green]")
    else:
        console.print(
            f"[dim]Collection '{COLLECTION}' already has "
            f"{collection_info.points_count} chunks — skipping indexing.[/dim]"
        )

    pipeline.build_agent()

    app = ChatApp(pipeline)
    app.mainloop()


if __name__ == "__main__":
    run()
