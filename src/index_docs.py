import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console

from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from parser import DocumentParser

# === À modifier selon ce que tu veux indexer ===
COLLECTION = "Functional_description"
FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Functional_description")
FORCE = False  # True = delete et create again
# ===============================================


def run():
    console = Console()

    if not os.path.isdir(FOLDER_PATH):
        console.print(f"[red]Folder not found: {FOLDER_PATH}[/red]")
        return

    with console.status("[cyan]Loading embeddings…[/cyan]"):
        embeddings = EmbeddingsManager().get_embeddings()

    with console.status(f"[cyan]Connecting to Qdrant ('{COLLECTION}')…[/cyan]"):
        vsm = VectorStoreManager(
            embeddings=embeddings,
            collection_name=COLLECTION,
            force_recreate=FORCE,
        )

    points_count = vsm._client.get_collection(COLLECTION).points_count

    if points_count and not FORCE:
        console.print(
            f"[yellow]'{COLLECTION}' already has {points_count} chunks "
            f"— set FORCE = True to reindex.[/yellow]"
        )
        return

    # Indexation
    doc_parser = DocumentParser()
    with console.status(f"[cyan]Indexing {COLLECTION}…[/cyan]"):
        docs = doc_parser.load_folder(FOLDER_PATH)
        ids = vsm.add_documents(docs)
    console.print(f"[green]Indexed {len(ids)} chunks into '{COLLECTION}'.[/green]")


if __name__ == "__main__":
    run()