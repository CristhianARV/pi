import gradio as gr

from rag import RAGPipeline


def _build_blocks(pipeline: RAGPipeline) -> gr.Blocks:
    def respond(message: str, history: list):
        """Generator: yields the growing response string as chunks arrive."""
        accumulated = ""

        for msg in pipeline.stream(message):
            role = getattr(msg, "type", "ai")
            if role == "human":
                continue

            if role == "tool":
                content = msg.content or ""
                preview = content + ("…" if len(content) > 600 else "")
                accumulated += f"**Retrieved context**\n```\n{preview}\n```\n\n"
                yield accumulated

            else:
                content = msg.content
                if isinstance(content, list):
                    content = "\n".join(
                        p["text"]
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                if content:
                    accumulated += content
                    yield accumulated

    with gr.Blocks(title="RAG Assistant") as app:
        gr.Markdown("## RAG Assistant")
        gr.ChatInterface(
            fn=respond,
            examples=["What is this document about?", "Summarise the key points."],
        )

    return app


class ChatApp:
    """Web-based chat interface for the RAG pipeline (works locally and over SSH)."""

    def __init__(self, pipeline: RAGPipeline):
        self._app = _build_blocks(pipeline)

    def mainloop(self, **kwargs):
        """Launch the Gradio server. VS Code Remote SSH forwards the port automatically."""
        self._app.launch(theme=gr.themes.Soft(), **kwargs)
