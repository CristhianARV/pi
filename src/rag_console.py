import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console

from llm import LLMManager
from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from parser import DocumentParser
from rag import RAGPipeline
from interface import CLI

FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Manuals", "mds_axis_compensation_en.pdf")
COLLECTION = "docs"


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
        with console.status(f"[cyan]Indexing document…[/cyan]"):
            n = pipeline.index_file(FILE_PATH, parser)
        console.print(f"[green]Indexed {n} chunks.[/green]\n")
    else:
        console.print(
            f"[dim]Collection '{COLLECTION}' already has "
            f"{collection_info.points_count} chunks — skipping indexing.[/dim]\n"
        )

    pipeline.build_agent()
    CLI(pipeline).run()


if __name__ == "__main__":
    run()
