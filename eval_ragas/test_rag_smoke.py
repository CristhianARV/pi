import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from parser import DocumentParser
from bdVector import VectorStoreManager
from models import make_embeddings, make_sparse_embeddings, make_llm
from rag import RAGPipeline


PDF_PATH = "/home/florian/Projet_PI/pi/data/Manuals/PLC-libraries/mcp_appl_examples_en.pdf"
QUESTION = "What is the purpose of the Frame_PLCopenP1.pro example?"
COLLECTION_NAME = "docs_smoke_v2"


print("[1] parsing...")
parser = DocumentParser()
docs = parser.load(PDF_PATH)
print("N_DOCS =", len(docs))

docs = docs[:5]
print("USING_DOCS =", len(docs))


print("[2] embeddings + vector store...")
embeddings = make_embeddings()
sparse_embeddings = make_sparse_embeddings()

vsm = VectorStoreManager(
    embeddings=embeddings,
    sparse_embeddings=sparse_embeddings,
    collection_name=COLLECTION_NAME,
    force_recreate=True,
)


print("[3] indexing...")
ids = vsm.add_documents(docs)
print("INDEXED =", len(ids))


print("[4] retrieval...")
retrieved = vsm.search(QUESTION, k=5, mode="semantic")
print("RETRIEVED =", len(retrieved))

for i, d in enumerate(retrieved, start=1):
    print(f"\n--- RETRIEVED {i} ---")
    print("METADATA =", getattr(d, "metadata", {}))
    print((d.page_content or "")[:400])


print("[5] RAG pipeline...")
llm = make_llm()

pipeline = RAGPipeline(
    vector_stores={COLLECTION_NAME: vsm},
    llm=llm,
    default_collection=COLLECTION_NAME,
    top_k=5,
    use_reranker=False,
    search_mode="semantic",
)

result = pipeline.ask_with_context(
    query=QUESTION,
    top_k=5,
)

print("\n[6] answer:")
print(result["response"])

print("\n[7] retrieved_context_ids:")
print(result["retrieved_context_ids"])

print("\n[8] retrieved_context_metadata:")
print(result["retrieved_context_metadata"])