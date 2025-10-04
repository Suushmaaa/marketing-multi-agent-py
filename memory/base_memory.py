"""
Base Memory Interface - Abstract class for all memory systems
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging


class BaseMemory(ABC):
    """
    Abstract base class for memory systems.
    All memory implementations must inherit from this class.
    """
    
    def __init__(self, memory_type: str, storage_backend=None):
        """
        Initialize base memory
        
        Args:
            memory_type: Type of memory (short_term/long_term/episodic/semantic)
            storage_backend: Storage backend (Redis, SQLite, Neo4j, etc.)
        """
        self.memory_type = memory_type
        self.storage_backend = storage_backend
        self.logger = logging.getLogger(f"memory.{memory_type}")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize memory system and connections
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def store(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """
        Store data in memory
        
        Args:
            key: Unique identifier for the memory
            data: Data to store
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from memory
        
        Args:
            key: Unique identifier
            **kwargs: Additional parameters
            
        Returns:
            Retrieved data or None
        """
        pass
    
    @abstractmethod
    async def update(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """
        Update existing memory
        
        Args:
            key: Unique identifier
            data: Updated data
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete memory entry
        
        Args:
            key: Unique identifier
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def search(self, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Search memory
        
        Args:
            query: Search criteria
            **kwargs: Additional parameters
            
        Returns:
            List of matching results
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup expired or unnecessary data
        
        Returns:
            bool: True if successful
        """
        pass
    
    async def is_healthy(self) -> bool:
        """
        Check if memory system is healthy
        
        Returns:
            bool: True if healthy
        """
        try:
            if not self.is_initialized:
                return False
            
            if self.storage_backend:
                return await self._check_backend_health()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _check_backend_health(self) -> bool:
        """Check backend connection health"""
        # Override in subclasses if needed
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            Dictionary with stats
        """
        return {
            "memory_type": self.memory_type,
            "is_initialized": self.is_initialized,
            "is_healthy": await self.is_healthy()
        }
    
    def _log_operation(self, operation: str, key: str, success: bool):
        """Log memory operation"""
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"{operation} - Key: {key} - Success: {success}"
        )


class MemoryConsolidationStrategy(ABC):
    """
    Abstract class for memory consolidation strategies
    Used to move data between memory layers
    """
    
    @abstractmethod
    async def should_consolidate(
        self,
        source_memory: BaseMemory,
        key: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Determine if data should be consolidated
        
        Args:
            source_memory: Source memory system
            key: Data key
            data: Data to evaluate
            
        Returns:
            bool: True if should consolidate
        """
        pass
    
    @abstractmethod
    async def consolidate(
        self,
        source_memory: BaseMemory,
        target_memory: BaseMemory,
        key: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Consolidate data from source to target memory
        
        Args:
            source_memory: Source memory
            target_memory: Target memory
            key: Data key
            data: Data to consolidate
            
        Returns:
            bool: True if successful
        """
        pass


class MemoryIndex(ABC):
    """
    Abstract class for memory indexing
    Enables fast retrieval and search
    """
    
    @abstractmethod
    async def add_to_index(self, key: str, data: Dict[str, Any]) -> bool:
        """Add entry to index"""
        pass
    
    @abstractmethod
    async def remove_from_index(self, key: str) -> bool:
        """Remove entry from index"""
        pass
    
    @abstractmethod
    async def search_index(
        self,
        query: Dict[str, Any],
        limit: int = 10
    ) -> List[str]:
        """Search index and return keys"""
        pass
    
    @abstractmethod
    async def rebuild_index(self) -> bool:
        """Rebuild entire index"""
        pass