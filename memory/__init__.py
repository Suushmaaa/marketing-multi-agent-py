"""
Memory Systems Package
Provides 4-layer memory architecture for the multi-agent system
"""

from .base_memory import BaseMemory, MemoryConsolidationStrategy, MemoryIndex
from .short_term import ShortTermMemorySystem
from .long_term import LongTermMemorySystem
from .episodic import EpisodicMemorySystem
from .semantic import SemanticMemorySystem
from .consolidation import MemoryConsolidationEngine, MemoryManager

__all__ = [
    # Base classes
    "BaseMemory",
    "MemoryConsolidationStrategy",
    "MemoryIndex",
    
    # Memory implementations
    "ShortTermMemorySystem",
    "LongTermMemorySystem",
    "EpisodicMemorySystem",
    "SemanticMemorySystem",
    
    # Management
    "MemoryConsolidationEngine",
    "MemoryManager"
]

__version__ = "1.0.0"