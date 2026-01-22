.PHONY: help install dev test lint format up down logs index clean

# Default target
help:
	@echo "PaperLens - ML Paper Search & Comparison Engine"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies"
	@echo "  dev-install Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  dev         Run development server (API + Frontend)"
	@echo "  api         Run API server only"
	@echo "  frontend    Run Streamlit frontend only"
	@echo "  test        Run tests"
	@echo "  lint        Run linter"
	@echo "  format      Format code"
	@echo ""
	@echo "Docker:"
	@echo "  up          Start all services (Docker)"
	@echo "  down        Stop all services"
	@echo "  logs        View logs"
	@echo "  build       Build Docker images"
	@echo ""
	@echo "Data:"
	@echo "  download    Download dataset from HuggingFace"
	@echo "  index       Index papers to Qdrant"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       Clean up generated files"
	@echo "  qdrant      Start Qdrant only"

# =============================================================================
# Setup
# =============================================================================
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

# =============================================================================
# Development
# =============================================================================
dev:
	@echo "Starting development servers..."
	@make -j2 api frontend

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/app.py --server.port 8501

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --no-cov

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# =============================================================================
# Docker
# =============================================================================
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

qdrant:
	docker compose up -d qdrant

# =============================================================================
# Data
# =============================================================================
download:
	python scripts/download_data.py

index:
	python scripts/index_papers.py

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
