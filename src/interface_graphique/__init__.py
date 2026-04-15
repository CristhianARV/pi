import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from rag import RAGPipeline


# ── colours & fonts ──────────────────────────────────────────────────────────
BG = "#1e1e2e"
BG_INPUT = "#2a2a3d"
BG_BUBBLE_USER = "#4a4e8a"
BG_BUBBLE_AI = "#2d2d44"
BG_BUBBLE_CTX = "#3a3320"
FG = "#cdd6f4"
FG_DIM = "#6c7086"
ACCENT = "#89b4fa"
YELLOW = "#f9e2af"
GREEN = "#a6e3a1"
FONT_MAIN = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 10)


class ChatApp(tk.Tk):
    """Tkinter chat interface for the RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline):
        super().__init__()
        self.pipeline = pipeline
        self.title("RAG Assistant")
        self.configure(bg=BG)
        self.geometry("900x650")
        self.minsize(600, 450)

        self._build_ui()
        self._show_welcome()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self, bg=ACCENT, height=48)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header, text="  RAG Assistant",
            bg=ACCENT, fg=BG, font=("Segoe UI", 14, "bold"),
            anchor="w",
        ).pack(side="left", fill="y", padx=8)
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(
            header, textvariable=self._status_var,
            bg=ACCENT, fg=BG, font=FONT_SMALL, anchor="e",
        ).pack(side="right", fill="y", padx=12)

        # Chat area
        self._chat = scrolledtext.ScrolledText(
            self, bg=BG, fg=FG, font=FONT_MAIN,
            wrap="word", state="disabled",
            relief="flat", bd=0, padx=12, pady=8,
        )
        self._chat.pack(fill="both", expand=True, padx=0, pady=0)
        self._configure_tags()

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Input bar
        bar = tk.Frame(self, bg=BG_INPUT, pady=8, padx=8)
        bar.pack(fill="x")
        self._input = tk.Text(
            bar, height=3, bg=BG_INPUT, fg=FG,
            font=FONT_MAIN, relief="flat", bd=0,
            insertbackground=FG, wrap="word",
        )
        self._input.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)  # allow newline

        self._send_btn = tk.Button(
            bar, text="Send", bg=ACCENT, fg=BG,
            font=FONT_BOLD, relief="flat", bd=0,
            padx=16, pady=8, cursor="hand2",
            activebackground="#6c9cd8", activeforeground=BG,
            command=self._send,
        )
        self._send_btn.pack(side="right", fill="y")

        # Hint
        tk.Label(
            self, text="Enter · Send   Shift+Enter · New line",
            bg=BG, fg=FG_DIM, font=FONT_SMALL,
        ).pack(pady=(0, 4))

    def _configure_tags(self) -> None:
        self._chat.tag_configure("user_label", foreground=ACCENT, font=FONT_BOLD)
        self._chat.tag_configure("user_bubble", background=BG_BUBBLE_USER, foreground=FG, font=FONT_MAIN, lmargin1=8, lmargin2=8, rmargin=8, spacing1=4, spacing3=4)
        self._chat.tag_configure("ai_label", foreground=GREEN, font=FONT_BOLD)
        self._chat.tag_configure("ai_bubble", background=BG_BUBBLE_AI, foreground=FG, font=FONT_MAIN, lmargin1=8, lmargin2=8, rmargin=8, spacing1=4, spacing3=4)
        self._chat.tag_configure("ctx_label", foreground=YELLOW, font=FONT_SMALL)
        self._chat.tag_configure("ctx_bubble", background=BG_BUBBLE_CTX, foreground=FG_DIM, font=FONT_MONO, lmargin1=8, lmargin2=8, rmargin=8, spacing1=2, spacing3=2)
        self._chat.tag_configure("divider", foreground=FG_DIM, font=FONT_SMALL, justify="center", spacing1=6, spacing3=6)
        self._chat.tag_configure("thinking", foreground=FG_DIM, font=("Segoe UI", 10, "italic"))

    # ── chat helpers ──────────────────────────────────────────────────────────

    def _append(self, text: str, *tags) -> None:
        self._chat.configure(state="normal")
        self._chat.insert("end", text, tags)
        self._chat.configure(state="disabled")
        self._chat.see("end")

    def _show_welcome(self) -> None:
        self._append("─── RAG Assistant ready ───\n", "divider")

    def _add_user_bubble(self, text: str) -> None:
        self._append("You\n", "user_label")
        self._append(f"{text}\n\n", "user_bubble")

    def _add_thinking(self) -> str:
        """Insert a 'thinking…' placeholder and return its position mark."""
        mark = f"thinking_{id(self)}"
        self._chat.configure(state="normal")
        self._chat.mark_set(mark, "end")
        self._chat.mark_gravity(mark, "left")
        self._chat.configure(state="disabled")
        self._append("Assistant  ⏳ thinking…\n", "thinking")
        return mark

    def _remove_thinking(self, mark: str) -> None:
        self._chat.configure(state="normal")
        self._chat.delete(mark, "end")
        self._chat.configure(state="disabled")

    def _add_context_bubble(self, text: str) -> None:
        self._append("  Retrieved context\n", "ctx_label")
        preview = text[:600] + ("…" if len(text) > 600 else "")
        self._append(f"{preview}\n\n", "ctx_bubble")

    def _add_ai_bubble(self, text: str) -> None:
        self._append("Assistant\n", "ai_label")
        self._append(f"{text}\n\n", "ai_bubble")

    # ── events ────────────────────────────────────────────────────────────────

    def _on_enter(self, event) -> str:
        if not event.state & 0x1:  # Shift not held
            self._send()
            return "break"

    def _send(self) -> None:
        query = self._input.get("1.0", "end").strip()
        if not query:
            return
        self._input.delete("1.0", "end")
        self._add_user_bubble(query)
        self._set_busy(True)
        threading.Thread(target=self._run_query, args=(query,), daemon=True).start()

    def _run_query(self, query: str) -> None:
        mark = self._add_thinking()
        messages = list(self.pipeline.stream(query))
        self.after(0, self._remove_thinking, mark)

        for msg in messages:
            role = getattr(msg, "type", "ai")
            if role == "human":
                continue
            if role == "tool":
                self.after(0, self._add_context_bubble, msg.content or "")
            else:
                content = msg.content
                if isinstance(content, list):
                    content = "\n".join(
                        p["text"] for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                if content:
                    self.after(0, self._add_ai_bubble, content)

        self.after(0, self._set_busy, False)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self._send_btn.configure(state=state)
        self._input.configure(state=state)
        self._status_var.set("Thinking…" if busy else "Ready")
