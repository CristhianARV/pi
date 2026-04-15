import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def _pick_mode() -> str:
    """Return 'console' or 'gui' from argv or interactive prompt."""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("console", "gui"):
            return arg
        print(f"Unknown mode '{arg}'. Use: console | gui")
        sys.exit(1)

    # Interactive choice
    print("RAG Assistant — choose a mode:")
    print("  [1] Console")
    print("  [2] Graphical interface (GUI)")
    while True:
        choice = input("Mode [1/2]: ").strip()
        if choice in ("1", "console"):
            return "console"
        if choice in ("2", "gui"):
            return "gui"
        print("Please enter 1 or 2.")


def main():
    mode = _pick_mode()
    if mode == "console":
        from rag_console import run
    else:
        from rag_graph import run
    run()


if __name__ == "__main__":
    main()
