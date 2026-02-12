"""
Data loader for HuggingFace datasets.

Loads ML papers from the CShorten/ML-ArXiv-Papers dataset.
"""

from collections.abc import Iterator

import structlog
from datasets import load_dataset

from src.config import settings
from src.models.paper import Paper

logger = structlog.get_logger()


class HuggingFaceDataLoader:
    """Load papers from HuggingFace datasets."""

    def __init__(self, dataset_name: str | None = None):
        """
        Initialize the data loader.

        Args:
            dataset_name: HuggingFace dataset name. Defaults to config value.
        """
        self.dataset_name = dataset_name or settings.dataset_name
        self._dataset = None

    def load_dataset(self, split: str = "train") -> None:
        """
        Load the dataset into memory.

        Args:
            split: Dataset split to load (usually 'train').
        """
        logger.info("Loading dataset", dataset=self.dataset_name, split=split)
        self._dataset = load_dataset(self.dataset_name, split=split)
        logger.info("Dataset loaded", num_papers=len(self._dataset))  # type: ignore[arg-type]

    def get_papers(
        self,
        limit: int | None = None,
        categories: list[str] | None = None,
    ) -> Iterator[Paper]:
        """
        Iterate over papers in the dataset.

        Args:
            limit: Maximum number of papers to return. None = all.
            categories: Filter by ArXiv categories (e.g., ['cs.LG', 'cs.AI']).

        Yields:
            Paper objects.
        """
        if self._dataset is None:
            self.load_dataset()

        count = 0
        for idx, item in enumerate(self._dataset):  # type: ignore[arg-type, var-annotated]
            # Parse paper from dataset item
            paper = self._parse_paper(item, idx)

            if paper is None:
                continue

            # Category filtering
            if categories:
                if not any(cat in paper.categories for cat in categories):
                    continue

            yield paper
            count += 1

            if limit and count >= limit:
                break

        logger.info("Papers loaded", count=count)

    def _parse_paper(self, item: dict, idx: int = 0) -> Paper | None:
        """
        Parse a dataset item into a Paper object.

        Args:
            item: Raw item from HuggingFace dataset.
            idx: Index of the item in the dataset (used for ID generation).

        Returns:
            Paper object or None if parsing fails.
        """
        try:
            # Extract title and abstract first - skip if missing
            title = str(item.get("title", "") or "").strip()
            abstract = str(item.get("abstract", "") or "").strip()

            if not title or not abstract:
                return None

            # Extract arxiv_id from various possible fields
            arxiv_id = (
                item.get("id")
                or item.get("arxiv_id")
                or item.get("paper_id")
                or item.get("Unnamed: 0")
            )

            # Convert to string and validate
            if arxiv_id is not None:
                arxiv_id = str(arxiv_id).strip()

            # Generate ID from index if not available
            if not arxiv_id:
                arxiv_id = f"paper_{idx:06d}"

            # Extract categories
            categories_raw = item.get("categories", "")
            if isinstance(categories_raw, str):
                categories = categories_raw.split()
            elif isinstance(categories_raw, list):
                categories = categories_raw
            else:
                categories = []

            # Create Paper object
            paper = Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=self._parse_authors(item.get("authors", "")),
                categories=categories,
                published=None,
                updated=None,
            )

            return paper

        except Exception as e:
            logger.warning("Failed to parse paper", error=str(e))
            return None

    def _parse_authors(self, authors_raw: str | list) -> list[str]:
        """Parse authors from various formats."""
        if isinstance(authors_raw, list):
            return authors_raw
        if isinstance(authors_raw, str):
            # Common formats: "Author1, Author2" or "Author1 and Author2"
            if "," in authors_raw:
                return [a.strip() for a in authors_raw.split(",")]
            if " and " in authors_raw:
                return [a.strip() for a in authors_raw.split(" and ")]
            return [authors_raw.strip()] if authors_raw.strip() else []
        return []

    def _parse_date(self, date_raw: str | None) -> None:
        """Parse date string to datetime. Returns None for simplicity in MVP."""
        # TODO: Implement date parsing if needed
        return None

    def get_sample(self, n: int = 5) -> list[Paper]:
        """
        Get a sample of papers for testing.

        Args:
            n: Number of papers to sample.

        Returns:
            List of Paper objects.
        """
        return list(self.get_papers(limit=n))

    def __len__(self) -> int:
        """Return total number of papers in dataset."""
        if self._dataset is None:
            self.load_dataset()
        return len(self._dataset)  # type: ignore[arg-type]


# Convenience function
def load_ml_papers(limit: int | None = None) -> list[Paper]:
    """
    Load ML papers from the default dataset.

    Args:
        limit: Maximum papers to load. None = all.

    Returns:
        List of Paper objects.
    """
    loader = HuggingFaceDataLoader()
    return list(loader.get_papers(limit=limit))


if __name__ == "__main__":
    # Quick test
    loader = HuggingFaceDataLoader()
    papers = loader.get_sample(3)

    for paper in papers:
        print(f"\n{'='*60}")
        print(f"ID: {paper.arxiv_id}")
        print(f"Title: {paper.title}")
        print(f"Categories: {paper.categories}")
        print(f"Abstract: {paper.abstract[:200]}...")
