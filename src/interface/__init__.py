from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.rule import Rule

from rag import RAGPipeline
from themes import THEMES, normalize_theme

console = Console()


class CLI:
    """Simple terminal interface for the RAG pipeline."""

    BANNER = "[bold cyan]RAG Assistant[/bold cyan]"
    HELP = "[dim]Type your question and press Enter. Commands: [bold]/quit[/bold] exit · [bold]/clear[/bold] reset screen[/dim]"

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.selected_theme = THEMES[0]

    def _print_banner(self) -> None:
        console.print(Panel(self.BANNER, subtitle=self.HELP, border_style="cyan"))

    def _stream_response(self, query: str) -> None:
        self.pipeline.set_active_theme(self.selected_theme)
        console.print(Rule("[dim]Assistant[/dim]", style="cyan"))
        final_content = ""
        with console.status("[cyan]Thinking…[/cyan]", spinner="dots"):
            messages = list(self.pipeline.stream(query))

        for msg in messages:
            role = getattr(msg, "type", "ai")
            if role == "human":
                continue
            if role == "tool":
                console.print(Panel(
                    f"[dim]{msg.content}[/dim]",
                    title="[yellow]Retrieved context[/yellow]",
                    border_style="yellow",
                    expand=False,
                ))
            else:
                content = msg.content
                if isinstance(content, list):
                    # extract text parts only
                    content = "\n".join(
                        part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"
                    )
                if content:
                    final_content = content

        if final_content:
            console.print(Markdown(final_content))
        console.print()

    def run(self) -> None:
        self._print_banner()
        console.print(f"[dim]Active theme: {self.selected_theme}[/dim]")
        console.print("[dim]Use /theme to change domain.[/dim]")
        console.print()

        while True:
            try:
                query = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not query:
                continue
            if query.lower() in ("/quit", "/exit", "quit", "exit"):
                console.print("[dim]Goodbye.[/dim]")
                break
            if query.lower() == "/clear":
                console.clear()
                self._print_banner()
                console.print(f"[dim]Active theme: {self.selected_theme}[/dim]")
                console.print()
                continue

            if query.lower() == "/theme":
                choice = Prompt.ask(
                    "Theme",
                    choices=THEMES,
                    default=self.selected_theme,
                )
                normalized = normalize_theme(choice)
                if normalized != "Unknown":
                    self.selected_theme = normalized
                    console.print(f"[green]Theme set to: {self.selected_theme}[/green]")
                else:
                    console.print("[yellow]Unknown theme, keeping previous value.[/yellow]")
                continue

            self._stream_response(query)
