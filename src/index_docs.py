import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console

from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from parser import DocumentParser

# === À modifier selon ce que tu veux indexer ===
COLLECTION = "Manuals"
FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Manuals")
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
        vsm = VectorStoreManager(embeddings=embeddings, collection_name=COLLECTION)

    info = vsm._client.get_collection(COLLECTION)

    # Backfill sparse si besoin
    if vsm.sparse_backfill_required() and info.points_count:
        with console.status("[cyan]Adding sparse vectors…[/cyan]"):
            n = vsm.ensure_sparse_vectors()
        console.print(f"[green]Updated {n} chunks with sparse vectors.[/green]")

    # Skip ou purge
    if info.points_count:
        if not FORCE:
            console.print(
                f"[yellow]'{COLLECTION}' already has {info.points_count} chunks "
                f"— set FORCE = True to reindex.[/yellow]"
            )
            return
        with console.status(f"[yellow]Clearing '{COLLECTION}'…[/yellow]"):
            vsm._client.delete_collection(COLLECTION)
            vsm = VectorStoreManager(embeddings=embeddings, collection_name=COLLECTION)

    # Indexation
    parser = DocumentParser()
    with console.status(f"[cyan]Indexing {COLLECTION}…[/cyan]"):
        docs = parser.load_folder(FOLDER_PATH)
        ids = vsm.add_documents(docs)
    console.print(f"[green]Indexed {len(ids)} chunks into '{COLLECTION}'.[/green]")


if __name__ == "__main__":
    run()