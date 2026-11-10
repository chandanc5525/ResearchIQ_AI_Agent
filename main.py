"""ResearchIQ command-line entry point.

Examples:
    uv run python main.py app
    uv run python main.py ingest path/to/paper.pdf
    uv run python main.py ingest data/uploaded_papers
    uv run python main.py docs
    uv run python main.py check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.config import PROJECT_ROOT, get_settings
from utils.logger import setup_logging


def _services():
    from services import create_application

    return create_application(get_settings())


def command_app(args: argparse.Namespace) -> int:
    """Launch the Streamlit application."""
    import subprocess

    script = PROJECT_ROOT / "ui" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(args.port),
    ]
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


def command_ingest(args: argparse.Namespace) -> int:
    """Ingest one PDF or every PDF in a directory."""
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"Path not found: {path}")
        return 2

    services = _services()
    if path.is_dir():
        results = services.ingestion.ingest_directory(path)
    else:
        try:
            results = [services.ingestion.ingest_file(path)]
        except Exception as exc:
            print(f"Failed: {exc}")
            return 1

    if not results:
        print("No PDF files were found.")
        return 1
    for result in results:
        label = "SKIPPED" if result.skipped else "INDEXED"
        print(
            f"[{label}] {result.record.filename} · "
            f"{result.record.page_count} pages · "
            f"{result.record.chunk_count} chunks · {result.message}"
        )
    stats = services.ingestion.stats()
    print(
        f"\nLibrary: {stats['documents']} papers · "
        f"{stats['pages']} pages · {stats['chunks']} vectors"
    )
    return 0


def command_docs(_: argparse.Namespace) -> int:
    """List indexed research papers."""
    services = _services()
    records = services.registry.list_documents()
    if not records:
        print("No papers indexed yet.")
        return 0
    for record in records:
        print(
            f"{record.doc_id}  {record.filename}  "
            f"({record.page_count} pages, {record.chunk_count} chunks)"
        )
    print(f"\nTotal vectors: {services.vector_store.count()}")
    return 0


def command_remove(args: argparse.Namespace) -> int:
    """Remove a paper and its vectors after explicit confirmation."""
    services = _services()
    record = services.registry.get(args.doc_id)
    if record is None:
        print(f"Document not found: {args.doc_id}")
        return 1

    if not args.yes:
        print(f"About to permanently remove '{record.filename}' and its vectors.")
        answer = input("Type 'remove' to continue: ").strip().lower()
        if answer != "remove":
            print("Cancelled.")
            return 1

    services.ingestion.remove_document(args.doc_id)
    print(f"Removed {record.filename} ({record.doc_id}) and its vectors.")
    return 0


def command_check(_: argparse.Namespace) -> int:
    """Validate configuration, models, embeddings, and storage."""
    services = _services()
    health = services.health()
    print(f"Ollama endpoint : {health['base_url']}")
    print(f"Ollama online   : {health['reachable']}")
    print(
        f"Selected model  : {health['model']} (installed={health['model_available']})"
    )
    if health.get("models"):
        print(f"Installed models: {', '.join(health['models'])}")
    if health.get("error"):
        print(f"Model warning   : {health['error']}")
    print(f"Embedding model : {services.settings.embedding_model}")
    print(f"Vector store    : {services.settings.vectorstore_dir}")
    print(f"Papers indexed  : {services.registry.count()}")
    print(f"Vector chunks   : {services.vector_store.count()}")
    print(
        f"LangSmith       : {'enabled' if health.get('tracing_enabled') else 'disabled'}"
    )
    return 0 if health["reachable"] and health["model_available"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchiq",
        description="ResearchIQ — agentic research intelligence with local Ollama.",
    )
    subparsers = parser.add_subparsers(dest="command")

    app_parser = subparsers.add_parser("app", help="Launch the Streamlit dashboard")
    app_parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    app_parser.set_defaults(func=command_app)

    ingest_parser = subparsers.add_parser("ingest", help="Index a PDF or directory")
    ingest_parser.add_argument("path", help="PDF file or directory of PDFs")
    ingest_parser.set_defaults(func=command_ingest)

    docs_parser = subparsers.add_parser("docs", help="List indexed papers")
    docs_parser.set_defaults(func=command_docs)

    remove_parser = subparsers.add_parser("remove", help="Remove a paper and vectors")
    remove_parser.add_argument("doc_id", help="Document ID shown by `docs`")
    remove_parser.add_argument(
        "--yes", action="store_true", help="Skip the interactive confirmation"
    )
    remove_parser.set_defaults(func=command_remove)

    check_parser = subparsers.add_parser("check", help="Check models and storage")
    check_parser.set_defaults(func=command_check)
    return parser


def main() -> int:
    settings = get_settings()
    setup_logging(
        settings.log_level, settings.logs_dir / "researchiq.log", console=False
    )
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
