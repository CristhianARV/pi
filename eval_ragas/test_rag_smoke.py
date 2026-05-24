from parser import DocumentParser
from embeddings import EmbeddingsManager
from bdVector import VectorStoreManager
from llm import LLMManager

PDF_PATH = "data/Manuals/mcp_appl_examples_en_PLC-libraries.pdf"
QUESTION = "What is the purpose of the Frame_PLCopenP1.pro example?"

print("[1] parsing...")
parser = DocumentParser()
docs = parser.load(PDF_PATH)
print("N_DOCS =", len(docs))

docs = docs[:5]  # limit to first 5 chunks for smoke test
print("USING_DOCS =", len(docs))

print("[2] embeddings + vector store...")
embeddings = EmbeddingsManager().get_embeddings()
vsm = VectorStoreManager(
    embeddings=embeddings,
    collection_name="docs_smoke_v2",
)

try:
    info = vsm._client.get_collection("docs_smoke_v2")
    print("EXISTING_POINTS =", info.points_count)
except Exception:
    pass

print("[3] indexing...")
ids = vsm.add_documents(docs)
print("INDEXED =", len(ids))

print("[4] retrieval...")
retrieved = vsm.similarity_search(QUESTION, k=5)
print("RETRIEVED =", len(retrieved))
for i, d in enumerate(retrieved, start=1):
    print(f"\n--- RETRIEVED {i} ---")
    print((d.page_content or "")[:400])

print("[5] llm...")
llm = LLMManager().get_llm()
context = "\n\n".join(
    f"Source: {getattr(d, 'metadata', {})}\nContent: {getattr(d, 'page_content', '')}"
    for d in retrieved
)

prompt = (
    "You are a helpful assistant.\n"
    "Answer the question only using the retrieved context.\n"
    "If the answer is not in the context, say so clearly.\n\n"
    f"Question:\n{QUESTION}\n\n"
    f"Retrieved context:\n{context}"
)

resp = llm.invoke(prompt)
print("\n[6] answer:")
print(resp)