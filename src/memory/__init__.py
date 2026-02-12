"""
Agentic Memory System for PaperLens.

Provides multi-store memory architecture:
- Semantic: Paper vectors (Qdrant)
- Episodic: Interaction history (SQLite)
- Working: Session context (in-memory)
- Belief: User preferences (SQLite)
- Manager: Unified orchestration
"""

from src.memory.belief import BeliefMemoryStore, get_belief_store
from src.memory.episodic import EpisodicMemoryStore, get_episodic_store
from src.memory.manager import MemoryManager, get_memory_manager
from src.memory.semantic import SemanticMemory, get_semantic_memory
from src.memory.working import WorkingMemory, get_working_memory

__all__ = [
    # Working memory
    "WorkingMemory",
    "get_working_memory",
    # Semantic memory
    "SemanticMemory",
    "get_semantic_memory",
    # Episodic memory
    "EpisodicMemoryStore",
    "get_episodic_store",
    # Belief memory
    "BeliefMemoryStore",
    "get_belief_store",
    # Memory manager
    "MemoryManager",
    "get_memory_manager",
]
