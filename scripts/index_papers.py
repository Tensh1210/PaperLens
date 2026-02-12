#!/usr/bin/env python3
"""
Paper Indexing Script for PaperLens.

Downloads papers from HuggingFace and indexes them into Qdrant.

Usage:
    python scripts/index_papers.py [--limit N] [--batch-size N] [--recreate]

Examples:
    # Index first 1000 papers
    python scripts/index_papers.py --limit 1000

    # Index all papers with larger batches
    python scripts/index_papers.py --batch-size 100

    # Recreate the collection from scratch
    python scripts/index_papers.py --recreate --limit 5000
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.clients.data_loader import HuggingFaceDataLoader
from src.models.paper import Paper
from src.services.embedding import EmbeddingService
from src.services.vector_store import VectorStore

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()
console = Console()


def index_papers(
    limit: int | None = None,
    batch_size: int = 50,
    recreate: bool = False,
    categories: list[str] | None = None,
) -> dict:
    """
    Index papers from HuggingFace into Qdrant.

    Args:
        limit: Maximum papers to index. None = all.
        batch_size: Number of papers to process at once.
        recreate: Whether to recreate the collection.
        categories: Filter by ArXiv categories.

    Returns:
        Dict with indexing statistics.
    """
    stats = {
        "total_processed": 0,
        "total_indexed": 0,
        "total_skipped": 0,
        "total_errors": 0,
        "duration_seconds": 0,
    }

    start_time = time.time()

    console.print("\n[bold blue]PaperLens Indexing Script[/bold blue]\n")

    # Initialize services
    console.print("[yellow]Initializing services...[/yellow]")

    loader = HuggingFaceDataLoader()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    # Create or recreate collection
    console.print(f"[yellow]Setting up collection: {vector_store.collection_name}[/yellow]")
    vector_store.create_collection(recreate=recreate)

    # Get initial count
    initial_count = vector_store.count()
    console.print(f"[dim]Current papers in index: {initial_count}[/dim]")

    # Load dataset
    console.print(f"[yellow]Loading dataset: {loader.dataset_name}[/yellow]")
    loader.load_dataset()

    total_available = len(loader)
    total_to_process = min(limit, total_available) if limit else total_available
    console.print(f"[dim]Papers available: {total_available}[/dim]")
    console.print(f"[dim]Papers to process: {total_to_process}[/dim]\n")

    # Process in batches
    batch_papers: list[Paper] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Indexing papers...", total=total_to_process)

        for paper in loader.get_papers(limit=limit, categories=categories):
            batch_papers.append(paper)
            stats["total_processed"] += 1

            # Process batch when full
            if len(batch_papers) >= batch_size:
                indexed, errors = _process_batch(
                    batch_papers, embedding_service, vector_store
                )
                stats["total_indexed"] += indexed
                stats["total_errors"] += errors

                progress.update(task, advance=len(batch_papers))
                batch_papers = []

        # Process remaining papers
        if batch_papers:
            indexed, errors = _process_batch(
                batch_papers, embedding_service, vector_store
            )
            stats["total_indexed"] += indexed
            stats["total_errors"] += errors
            progress.update(task, advance=len(batch_papers))

    # Calculate duration
    stats["duration_seconds"] = round(time.time() - start_time, 2)

    # Get final count
    final_count = vector_store.count()
    stats["total_skipped"] = stats["total_processed"] - stats["total_indexed"] - stats["total_errors"]

    # Print summary
    console.print("\n[bold green]Indexing Complete![/bold green]\n")
    console.print(f"  Papers processed: {stats['total_processed']}")
    console.print(f"  Papers indexed:   {stats['total_indexed']}")
    console.print(f"  Papers skipped:   {stats['total_skipped']}")
    console.print(f"  Errors:           {stats['total_errors']}")
    console.print(f"  Duration:         {stats['duration_seconds']}s")
    console.print(f"\n  Total in index:   {final_count}")

    return stats


def _process_batch(
    papers: list[Paper],
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
) -> tuple[int, int]:
    """
    Process a batch of papers.

    Args:
        papers: Papers to process.
        embedding_service: Embedding service.
        vector_store: Vector store.

    Returns:
        Tuple of (indexed_count, error_count).
    """
    indexed = 0
    errors = 0

    try:
        # Filter valid papers
        valid_papers = [
            p for p in papers
            if p.arxiv_id and p.title and p.abstract
        ]

        if not valid_papers:
            return 0, len(papers)

        # Generate embeddings
        embeddings = embedding_service.embed_papers(
            valid_papers,
            batch_size=len(valid_papers),
            show_progress=False,
        )

        # Upsert to vector store
        vector_store.upsert_papers(valid_papers, embeddings, batch_size=len(valid_papers))

        indexed = len(valid_papers)
        errors = len(papers) - len(valid_papers)

    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        errors = len(papers)

    return indexed, errors


def verify_index() -> dict:
    """
    Verify the index by running some test queries.

    Returns:
        Dict with verification results.
    """
    console.print("\n[yellow]Verifying index...[/yellow]")

    from src.memory.semantic import SemanticMemory

    memory = SemanticMemory()
    stats = memory.get_stats()

    console.print(f"  Total papers: {stats.get('total_papers', 0)}")
    console.print(f"  Status: {stats.get('status', 'unknown')}")

    # Run a test query
    if stats.get('total_papers', 0) > 0:
        console.print("\n[yellow]Running test query...[/yellow]")
        results = memory.search("transformer attention mechanism", limit=3)

        if results:
            console.print(f"  Found {len(results)} results for 'transformer attention mechanism'")
            for i, r in enumerate(results):
                console.print(f"    {i+1}. {r.paper.title[:60]}... (score: {r.score:.3f})")
        else:
            console.print("  [red]No results found![/red]")

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index ML papers into PaperLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --limit 1000          # Index first 1000 papers
  %(prog)s --batch-size 100      # Use larger batch size
  %(prog)s --recreate            # Recreate collection from scratch
  %(prog)s --verify              # Just verify existing index
        """,
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of papers to index (default: all)",
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection (deletes existing data)",
    )

    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        default=None,
        help="Filter by ArXiv categories (e.g., cs.LG cs.AI)",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Just verify the existing index",
    )

    args = parser.parse_args()

    try:
        if args.verify:
            verify_index()
        else:
            # Run indexing
            stats = index_papers(
                limit=args.limit,
                batch_size=args.batch_size,
                recreate=args.recreate,
                categories=args.categories,
            )

            # Verify after indexing
            verify_index()

            # Return appropriate exit code
            if stats["total_errors"] > 0 and stats["total_indexed"] == 0:
                sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Indexing cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Indexing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
