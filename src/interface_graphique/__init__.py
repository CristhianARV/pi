import gradio as gr

from rag import RAGPipeline
from themes import THEMES, normalize_theme


def _build_blocks(pipeline: RAGPipeline) -> gr.Blocks:
    def respond(message: str, history: list, theme: str):
        """Generator: yields the growing response string as chunks arrive."""
        selected_theme = normalize_theme(theme)
        if selected_theme == "Unknown":
            yield "Please select a valid theme before asking a question."
            return

        pipeline.set_active_theme(selected_theme)
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

        theme_dropdown = gr.Dropdown(
            choices=THEMES,
            label="Theme",
            value=THEMES[0],
            allow_custom_value=False,
            interactive=True,
        )

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[theme_dropdown],
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
