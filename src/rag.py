import time

from langchain_core.messages import HumanMessage, SystemMessage

from bdVector import VectorStoreManager


SEARCH_MODES = ("semantic", "text", "hybrid")
DEFAULT_SEARCH_MODE = "semantic"


DEFAULT_SYSTEM_PROMPT = """You are a careful technical assistant for CNC manuals.
Answer questions using ONLY the retrieved context.
If the context does not contain the answer, say so explicitly — do not invent.
Cite source filename and page when possible."""


ANSWER_PROMPT = """Answer the user's question using ONLY the context below.
If the context is insufficient, say so explicitly — do not invent.
Cite source filename and page when possible.

=== CONTEXT ({mode}) ===
{context}

=== QUESTION ===
{question}
"""


SYNTHESIS_PROMPT = """You received two independent answers to the same question.
Each was generated from a different retrieval strategy:
- ANSWER A comes from semantic (embedding-based) retrieval.
- ANSWER B comes from textual (BM25 keyword) retrieval.

Your job:
1. If both answers agree on the facts, give a single consolidated answer.
2. If they cover different aspects, merge them.
3. If they contradict, pick the one with more specific citations (filename + page)
   and explain briefly why you trust it more.
4. If both say they don't know, say so.

Do not invent information that is in neither answer.

=== QUESTION ===
{question}

=== ANSWER A (semantic) ===
{answer_sem}

=== ANSWER B (textual) ===
{answer_txt}

=== FINAL ANSWER ==="""


MODE_LABELS = {
    "semantic": "Sémantique",
    "text": "Texte (BM25)",
}


def _format_docs_for_prompt(docs) -> str:
    if not docs:
        return "(no documents retrieved)"
    return "\n\n".join(
        f"[{i+1}] Source: {d.metadata}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def _format_docs_full(docs, max_snippet: int = 220) -> str:
    if not docs:
        return "_Aucun document._"
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        filename = meta.get("filename", "?")
        page = meta.get("page_no", "?")
        score = meta.get("rerank_score")
        score_str = f" — score: {score:.3f}" if score is not None else ""
        snippet = " ".join(d.page_content.split())
        if len(snippet) > max_snippet:
            snippet = snippet[:max_snippet] + "…"
        lines.append(f"**{i}.** `{filename}` p.{page}{score_str}  \n> {snippet}")
    return "\n\n".join(lines)


class RAGPipeline:
    """Retrieval pipeline with selectable mode and optional reranker."""

    def __init__(
        self,
        vector_stores: dict,
        llm,
        default_collection: str,
        reranker=None,
        fetch_k: int = 50,
        top_k: int = 10,
        use_reranker: bool = True,
        search_mode: str = DEFAULT_SEARCH_MODE,
        system_prompt: str | None = None,
    ):
        self.vector_stores = vector_stores
        self.llm = llm
        self.collection_name = default_collection
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.reranker = reranker
        self.use_reranker = bool(use_reranker) and reranker is not None
        self.search_mode = (
            search_mode if search_mode in SEARCH_MODES else DEFAULT_SEARCH_MODE
        )
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # ---------------- accessors ----------------

    @property
    def vsm(self) -> VectorStoreManager:
        return self.vector_stores[self.collection_name]

    def available_collections(self) -> list[str]:
        return list(self.vector_stores.keys())

    def set_collection(self, name: str) -> None:
        if name not in self.vector_stores:
            raise ValueError(f"Unknown collection: {name}")
        self.collection_name = name

    def set_search_mode(self, mode: str) -> None:
        if mode in SEARCH_MODES:
            self.search_mode = mode

    def set_use_reranker(self, on: bool) -> None:
        self.use_reranker = bool(on) and self.reranker is not None

    # ---------------- retrieval ----------------

    def _retrieve(self, query: str, mode: str, theme: str | None = None) -> list:
        """fetch_k → optional rerank → top_k."""
        k = self.fetch_k if self.use_reranker else self.top_k
        candidates = self.vsm.search(query, k=k, mode=mode)

        if self.use_reranker and self.reranker and candidates:
            return self.reranker.rerank(query, candidates, top_k=self.top_k)

        return candidates[: self.top_k]

    # ---------------- streaming ----------------

    def stream(self, query: str):
        if self.search_mode == "hybrid":
            yield from self._stream_hybrid(query)
        else:
            yield from self._stream_single(query, self.search_mode)

    def _stream_single(self, query: str, mode: str):
        label = MODE_LABELS[mode]
        rerank_note = " · reranké" if self.use_reranker else ""

        step_search = f"### 🔎 Recherche — {label}\n_En cours…_"
        step_docs = ""
        step_answer = ""

        def render() -> str:
            return "\n\n".join(s for s in (step_search, step_docs, step_answer) if s)

        yield render()

        docs = self._retrieve(query, mode)
        step_search = f"### 🔎 Recherche — {label} ({len(docs)} chunks{rerank_note})"
        step_docs = "**Documents :**\n\n" + _format_docs_full(docs)
        step_answer = "### 💬 Réponse\n"
        yield render()

        yield from self._stream_answer(
            query, docs, label.upper(), step_search, step_docs
        )

    def _stream_hybrid(self, query: str):
        rerank_note = " · reranké" if self.use_reranker else ""
        step1 = "### 🔎 Étape 1 — Recherche sémantique\n_En cours…_"
        step2 = ""
        step3 = ""

        def render() -> str:
            return "\n\n".join(s for s in (step1, step2, step3) if s)

        yield render()

        docs_sem = self._retrieve(query, "semantic")
        answer_sem = self._answer_from_docs(query, docs_sem, "SEMANTIC")
        step1 = (
            f"### 🔎 Étape 1 — Recherche sémantique "
            f"({len(docs_sem)} chunks{rerank_note})\n\n"
            f"**Documents :**\n\n{_format_docs_full(docs_sem)}\n\n"
            f"**Réponse intermédiaire :**\n\n{answer_sem}"
        )
        step2 = "### 🔎 Étape 2 — Recherche textuelle (BM25)\n_En cours…_"
        yield render()

        docs_txt = self._retrieve(query, "text")
        answer_txt = self._answer_from_docs(query, docs_txt, "TEXTUAL")
        step2 = (
            f"### 🔎 Étape 2 — Recherche textuelle "
            f"({len(docs_txt)} chunks{rerank_note})\n\n"
            f"**Documents :**\n\n{_format_docs_full(docs_txt)}\n\n"
            f"**Réponse intermédiaire :**\n\n{answer_txt}"
        )
        step3 = "### 🧠 Étape 3 — Synthèse finale\n"
        yield render()

        synth_prompt = SYNTHESIS_PROMPT.format(
            question=query, answer_sem=answer_sem, answer_txt=answer_txt,
        )
        synth = ""
        for chunk in self.llm.stream([
            SystemMessage(content="You are a careful synthesis assistant."),
            HumanMessage(content=synth_prompt),
        ]):
            token = getattr(chunk, "content", "") or ""
            synth += token
            step3 = f"### 🧠 Étape 3 — Synthèse finale\n{synth}"
            yield render()

    # ---------------- LLM helpers ----------------

    def _stream_answer(self, query, docs, label, *prev_steps):
        prompt = ANSWER_PROMPT.format(
            mode=label,
            context=_format_docs_for_prompt(docs),
            question=query,
        )
        answer = ""
        for chunk in self.llm.stream([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]):
            token = getattr(chunk, "content", "") or ""
            answer += token
            step_answer = f"### 💬 Réponse\n{answer}"
            yield "\n\n".join([*prev_steps, step_answer])

    def _answer_from_docs(self, query: str, docs, label: str) -> str:
        prompt = ANSWER_PROMPT.format(
            mode=label,
            context=_format_docs_for_prompt(docs),
            question=query,
        )
        return self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]).content

    # ---------------- one-shot ----------------

    def ask(self, query: str) -> str:
        last = ""
        for chunk in self.stream(query):
            last = chunk
        return last
    
    def ask_with_context(
        self,
        query: str,
        top_k: int | None = None,
        theme: str | None = None,
        mode: str | None = None,
    ) -> dict:
        """
        One-shot RAG call for evaluation.

        Returns the final answer plus retrieved contexts and metadata,
        so external evaluators such as RAGAS can score the pipeline.
        """
        started = time.perf_counter()

        selected_mode = mode if mode in SEARCH_MODES else self.search_mode

        old_top_k = self.top_k
        if top_k is not None:
            self.top_k = int(top_k)

        try:
            docs = self._retrieve(
                query=query,
                mode=selected_mode,
                theme=theme,
            )

            answer = self._answer_from_docs(
                query,
                docs,
                selected_mode.upper(),
            )

        finally:
            self.top_k = old_top_k

        contexts = []
        context_ids = []
        context_metadata = []

        for i, doc in enumerate(docs):
            meta = dict(doc.metadata or {})
            contexts.append(doc.page_content or "")
            context_metadata.append(meta)

            context_id = (
                meta.get("id")
                or meta.get("_id")
                or meta.get("chunk_id")
                or meta.get("source")
                or meta.get("filename")
                or f"doc_{i}"
            )
            context_ids.append(str(context_id))

        return {
            "response": answer,
            "retrieved_contexts": contexts,
            "retrieved_context_ids": context_ids,
            "retrieved_context_metadata": context_metadata,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "theme": theme,
            "mode": selected_mode,
        }

    # ---------------- indexing helpers ----------------

    def index_file(self, file_path: str, parser) -> int:
        return len(self.vsm.add_documents(parser.load(file_path)))

    def index_folder(self, folder_path: str, parser) -> int:
        return len(self.vsm.add_documents(parser.load_folder(folder_path)))