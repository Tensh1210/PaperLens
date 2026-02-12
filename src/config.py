"""
Configuration management for PaperLens.

Uses pydantic-settings for type-safe configuration from environment variables.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars not defined in Settings
    )

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    groq_api_key: str = ""
    openai_api_key: str = ""

    # Model selection
    llm_provider: str = "groq"  # groq or openai
    llm_model: str = "llama-3.3-70b-versatile"  # Groq model

    # =========================================================================
    # Vector Database (Qdrant)
    # =========================================================================
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "papers"

    # =========================================================================
    # Embedding Model
    # =========================================================================
    embedding_model: str = "sentence-transformers/allenai-specter"
    embedding_dimension: int = 768  # SPECTER dimension

    # =========================================================================
    # API Configuration
    # =========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # =========================================================================
    # Search Configuration
    # =========================================================================
    search_top_k: int = 10  # Number of papers to retrieve
    search_min_score: float = 0.5  # Minimum similarity score
    comparison_top_k: int = 5  # Number of papers to compare

    # =========================================================================
    # Data Configuration
    # =========================================================================
    dataset_name: str = "CShorten/ML-ArXiv-Papers"
    max_papers_to_index: int | None = None  # None = all papers

    # =========================================================================
    # External APIs
    # =========================================================================
    semantic_scholar_api_key: str = ""
    arxiv_rate_limit: float = 3.0  # seconds between requests

    # =========================================================================
    # Agent Configuration
    # =========================================================================
    agent_max_iterations: int = 5  # Max ReAct loops before stopping
    agent_temperature: float = 0.7  # LLM temperature for reasoning
    agent_timeout: int = 60  # Timeout in seconds for agent execution

    # =========================================================================
    # Memory Configuration
    # =========================================================================
    memory_db_path: str = "data/memory.db"  # SQLite for episodic/belief
    memory_working_size: int = 20  # Max items in working memory
    memory_episodic_limit: int = 100  # Max episodic memories to retrieve
    memory_belief_decay: float = 0.95  # Confidence decay factor for beliefs

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: str = "INFO"
    debug: bool = False

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def llm_full_model(self) -> str:
        """Get full model name for LiteLLM."""
        if self.llm_provider == "groq":
            return f"groq/{self.llm_model}"
        elif self.llm_provider == "openai":
            return self.llm_model
        return self.llm_model


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using lru_cache ensures settings are only loaded once.
    """
    return Settings()


# Convenience export
settings = get_settings()
