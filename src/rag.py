import os
from langchain.tools import tool
from langchain.agents import create_agent

from llm import get_llm
from embeddings import get_embeddings
from bdVector import get_vector_store
from parser import load_pdf, split_documents

# --- Setup ---


model = get_llm()
embeddings_model = get_embeddings()
vector_store = get_vector_store(embeddings_model)

# --- Load and index documents ---

docs = load_pdf("../data/Manuals/mds_axis_compensation_en.pdf")
splits = split_documents(docs)
document_ids = vector_store.add_documents(splits)
print(f"Indexed {len(document_ids)} chunks.")

# --- RAG tool ---

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# --- Agent with retrieval tool ---

tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)


# --- Test ---

if __name__ == "__main__":
    query = (
        "What is achs_nr meaning and how is it used in the context of the document?"
    )
    print("\n=== Agent with retrieval tool ===")
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
