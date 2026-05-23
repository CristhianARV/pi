import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rich.console import Console

from setup import build_pipeline
from interface_graphique import ChatApp


def run():
    console = Console()
    pipeline = build_pipeline(console)
    console.print(
        f"[green]Available collections:[/green] {pipeline.available_collections()}"
    )
    ChatApp(pipeline).mainloop()


if __name__ == "__main__":
    run()