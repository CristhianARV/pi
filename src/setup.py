from rich.console import Console

from llm import LLMManager
from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from reranker import Reranker
from rag import RAGPipeline


QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "Manuals"

FETCH_K = 50
TOP_K = 10
USE_RERANKER = True              # default state of the UI checkbox
DEFAULT_SEARCH_MODE = "semantic"


def build_pipeline(console: Console | None = None) -> RAGPipeline:
    console = console or Console()

    with console.status("[cyan]Loading embeddings model…[/cyan]"):
        embeddings = EmbeddingsManager().get_embeddings()

    with console.status("[cyan]Initialising LLM…[/cyan]"):
        llm = LLMManager().get_llm()

    with console.status("[cyan]Loading reranker model…[/cyan]"):
        reranker = Reranker()

    collection_names = (
        VectorStoreManager.list_collections(QDRANT_URL) or [DEFAULT_COLLECTION]
    )

    with console.status("[cyan]Connecting to Qdrant collections…[/cyan]"):
        vsms = {
            name: VectorStoreManager(
                embeddings=embeddings,
                collection_name=name,
                url=QDRANT_URL,
            )
            for name in collection_names
        }

    default_coll = (
        DEFAULT_COLLECTION if DEFAULT_COLLECTION in vsms else collection_names[0]
    )

    return RAGPipeline(
        vector_stores=vsms,
        llm=llm,
        default_collection=default_coll,
        reranker=reranker,
        fetch_k=FETCH_K,
        top_k=TOP_K,
        use_reranker=USE_RERANKER,
        search_mode=DEFAULT_SEARCH_MODE,
    )