import time
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage


from bdVector import VectorStoreManager


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Wires a VectorStoreManager and an LLM into a LangChain agent that
    uses a retrieval tool to answer user queries.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm,
        k: int = 10,
        system_prompt: str = (
            "You are a helpful assistant. "
            "Use the retrieve_context tool to look up relevant information "
            "before answering the user's question."
        ),
    ):
        self.vsm = vector_store_manager
        self.llm = llm
        self.k = k
        self.system_prompt = system_prompt
        self._agent = None
        self._last_retrieved_docs = []
        self._last_retrieved_query = None
        self._active_theme = None

    def set_active_theme(self, theme: str | None) -> None:
        self._active_theme = theme

    def _build_retrieve_tool(self):
        vsm = self.vsm
        k = self.k

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve relevant document chunks to help answer a query."""
            docs = vsm.similarity_search(query, k=k, theme=self._active_theme)

            self._last_retrieved_docs = docs
            self._last_retrieved_query = query

            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in docs
            )
            return serialized, docs

        return retrieve_context

    def build_agent(self):
        retrieve_tool = self._build_retrieve_tool()
        self._agent = create_agent(
            self.llm,
            [retrieve_tool],
            system_prompt=self.system_prompt,
        )
        return self._agent

    def index_file(self, file_path: str, parser) -> int:
        """Parse and index a document. Returns the number of chunks indexed."""
        docs = parser.load(file_path)
        ids = self.vsm.add_documents(docs)
        return len(ids)

    def stream(self, query: str):
        """Stream agent responses for a query, yielding messages."""
        if self._agent is None:
            self.build_agent()
        for event in self._agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            yield event["messages"][-1]

    def ask(self, query: str) -> BaseMessage:
        """Run the agent and return the final message."""
        last = None
        for msg in self.stream(query):
            last = msg
        return last

    def _message_to_text(self, msg: BaseMessage) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            return "\n".join(str(x) for x in content)
        return str(content)

    def ask_with_context(
        self,
        query: str,
        top_k: int | None = None,
        theme: str | None = None,
    ) -> dict:
        """
        Evaluation-friendly path:
        - retrieves contexts directly from the vector store
        - calls the LLM directly without agent/tool orchestration
        - returns response + retrieved contexts + latency
        """
        k = top_k if top_k is not None else self.k
        effective_theme = theme if theme is not None else self._active_theme

        start = time.perf_counter()

        docs = self.vsm.similarity_search(query, k=k, theme=effective_theme)
        self._last_retrieved_docs = docs
        self._last_retrieved_query = query

        retrieved_contexts = [getattr(doc, "page_content", "") for doc in docs]
        retrieved_context_ids = []

        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            doc_id = (
                metadata.get("id")
                or metadata.get("chunk_id")
                or metadata.get("document_id")
                or metadata.get("source")
                or ""
            )
            retrieved_context_ids.append(str(doc_id))

        context_text = "\n\n".join(
            f"Source: {getattr(doc, 'metadata', {})}\nContent: {getattr(doc, 'page_content', '')}"
            for doc in docs
        )

        prompt = (
            "You are a helpful assistant.\n"
            "Answer the user's question only using the retrieved context.\n"
            "If the answer is not in the context, say so clearly.\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{context_text}"
        )

        msg = self.llm.invoke(prompt)
        
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "question": query,
            "theme": effective_theme,
            "response": self._message_to_text(msg) if msg is not None else "",
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_ids": retrieved_context_ids,
            "latency_ms": round(latency_ms, 2),
        }