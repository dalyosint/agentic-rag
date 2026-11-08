
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.data_loader import init_collections  # noqa: E402
from src.graph import build_graph  # noqa: E402

REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY")


def check_env() -> None:
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        sys.exit(f"Missing keys in .env: {', '.join(missing)}. Fill them in and run again.")


def answer(app, query: str) -> dict:
    return app.invoke({"query": query, "iteration_count": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG over medical data, with web search fallback.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and re-index the ChromaDB collections.")
    parser.add_argument("--draw-graph", action="store_true", help="Save the workflow diagram as graph.png (needs internet).")
    args = parser.parse_args()

    check_env()
    init_collections(rebuild=args.rebuild)
    app = build_graph()

    if args.draw_graph:
        with open("graph.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("Saved graph.png")

    print("\nAgentic RAG ready. Type a question, or 'exit' to quit.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = answer(app, query)
        print(f"\nSource: {result.get('source')}")
        print(f"Answer: {result.get('response')}")


if __name__ == "__main__":
    main()