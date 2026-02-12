"""
Paper data models for PaperLens.

Defines the core data structures for representing academic papers.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field, computed_field


class Paper(BaseModel):
    """Represents an academic paper."""

    # Core identifiers
    arxiv_id: str = Field(..., description="ArXiv paper ID (e.g., '2301.12345')")

    # Content
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")

    # Metadata
    authors: list[str] = Field(default_factory=list, description="List of author names")
    categories: list[str] = Field(default_factory=list, description="ArXiv categories")

    # Dates
    published: datetime | None = Field(default=None, description="Publication date")
    updated: datetime | None = Field(default=None, description="Last update date")

    # External data (from Semantic Scholar, etc.)
    citation_count: int = Field(default=0, description="Number of citations")
    influential_citation_count: int = Field(default=0, description="Influential citations")
    tldr: str | None = Field(default=None, description="TL;DR summary")
    venue: str | None = Field(default=None, description="Publication venue")

    # Computed fields
    @computed_field  # type: ignore[prop-decorator]
    @property
    def pdf_url(self) -> str:
        """Generate PDF URL from ArXiv ID."""
        return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def arxiv_url(self) -> str:
        """Generate ArXiv abstract page URL."""
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def year(self) -> int | None:
        """Extract publication year."""
        if self.published:
            return self.published.year
        # Try to extract from arxiv_id (format: YYMM.NNNNN)
        if self.arxiv_id and "." in self.arxiv_id:
            try:
                yymm = self.arxiv_id.split(".")[0]
                if len(yymm) == 4:
                    year = int(yymm[:2])
                    return 2000 + year if year < 50 else 1900 + year
            except (ValueError, IndexError):
                pass
        return None

    def to_embedding_text(self) -> str:
        """
        Generate text for embedding.

        SPECTER2 performs best with title + abstract combined.
        """
        return f"{self.title} {self.abstract}"

    def to_display_dict(self) -> dict:
        """Convert to dictionary for display purposes."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract[:500] + "..." if len(self.abstract) > 500 else self.abstract,
            "authors": ", ".join(self.authors[:5]) + ("..." if len(self.authors) > 5 else ""),
            "year": self.year,
            "citations": self.citation_count,
            "pdf_url": self.pdf_url,
            "arxiv_url": self.arxiv_url,
        }


class PaperSearchResult(BaseModel):
    """A paper with search relevance score."""

    paper: Paper
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")

    def __lt__(self, other: "PaperSearchResult") -> bool:
        """Enable sorting by score (descending)."""
        return self.score > other.score


class PaperComparison(BaseModel):
    """Result of comparing multiple papers."""

    query: str = Field(..., description="Original search query")
    papers: list[PaperSearchResult] = Field(..., description="Papers being compared")

    # Comparison results
    comparison_text: str = Field(..., description="LLM-generated comparison")
    timeline: list[dict] | None = Field(default=None, description="Papers ordered by date")
    key_contributions: dict[str, str] | None = Field(
        default=None, description="Key contribution per paper"
    )

    # Metadata
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_used: str = Field(default="", description="LLM model used for comparison")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in comparison quality"
    )


class IndexStats(BaseModel):
    """Statistics about the paper index."""

    total_papers: int = Field(..., description="Total papers in index")
    categories: dict[str, int] = Field(
        default_factory=dict, description="Papers per category"
    )
    year_distribution: dict[int, int] = Field(
        default_factory=dict, description="Papers per year"
    )
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
