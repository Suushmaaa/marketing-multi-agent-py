"""
Short-Term Memory Implementation
Stores temporary conversation context with TTL (Time-To-Live)
Uses Redis for fast access and automatic expiration
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
from redis import asyncio as aioredis

from .base_memory import BaseMemory
from database.models import ShortTermMemory
from config.settings import settings


class ShortTermMemorySystem(BaseMemory):
    """
    Short-term memory for active conversations.
    Automatically expires after inactivity period.
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """
        Initialize short-term memory
        
        Args:
            redis_client: Redis client instance
        """
        super().__init__(memory_type="short_term", storage_backend=redis_client)
        self.redis = redis_client
        self.ttl = settings.SHORT_TERM_MEMORY_TTL
        self.key_prefix = "stm:"
        self.stats = {
            "total_stores": 0,
            "total_retrievals": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "expirations": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        try:
            if not self.redis:
                self.redis = await aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
            
            # Test connection
            await self.redis.ping()
            
            self.is_initialized = True
            self.logger.info("Short-term memory initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            return False
    
    async def store(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store conversation context in short-term memory
        
        Args:
            key: Conversation ID
            data: Context data
            ttl: Time-to-live in seconds (optional, uses default if not provided)
            
        Returns:
            bool: True if successful
        """
        try:
            redis_key = f"{self.key_prefix}{key}"
            
            # Validate data against model
            memory_obj = ShortTermMemory(
                conversation_id=key,
                lead_id=data.get("lead_id"),
                last_utterance_summary=data.get("last_utterance_summary", ""),
                active_intent=data.get("active_intent", "unknown"),
                slots_json=data.get("slots_json", {}),
                expires_at=datetime.now() + timedelta(seconds=ttl or self.ttl)
            )
            
            # Serialize to JSON
            serialized = json.dumps({
                **memory_obj.dict(),
                "updated_at": datetime.now().isoformat()
            })
            
            # Store with TTL
            await self.redis.setex(
                redis_key,
                ttl or self.ttl,
                serialized
            )
            
            self.stats["total_stores"] += 1
            self._log_operation("STORE", key, True)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store: {e}", exc_info=True)
            self._log_operation("STORE", key, False)
            return False
    
    async def retrieve(
        self,
        key: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation context from short-term memory
        
        Args:
            key: Conversation ID
            
        Returns:
            Context data or None if not found/expired
        """
        try:
            redis_key = f"{self.key_prefix}{key}"
            
            # Get from Redis
            data = await self.redis.get(redis_key)
            
            self.stats["total_retrievals"] += 1
            
            if data:
                self.stats["cache_hits"] += 1
                parsed = json.loads(data)
                
                # Check if expired
                expires_at = datetime.fromisoformat(parsed["expires_at"])
                if datetime.now() > expires_at:
                    await self.delete(key)
                    self.stats["expirations"] += 1
                    self.stats["cache_misses"] += 1
                    return None
                
                self._log_operation("RETRIEVE", key, True)
                return parsed
            else:
                self.stats["cache_misses"] += 1
                self._log_operation("RETRIEVE", key, False)
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve: {e}", exc_info=True)
            self.stats["cache_misses"] += 1
            return None
    
    async def update(
        self,
        key: str,
        data: Dict[str, Any],
        extend_ttl: bool = True,
        **kwargs
    ) -> bool:
        """
        Update existing short-term memory
        
        Args:
            key: Conversation ID
            data: Updated context data
            extend_ttl: Whether to extend TTL on update
            
        Returns:
            bool: True if successful
        """
        try:
            # Retrieve existing data
            existing = await self.retrieve(key)
            
            if not existing:
                # If doesn't exist, create new
                return await self.store(key, data)
            
            # Merge data
            existing.update(data)
            existing["updated_at"] = datetime.now().isoformat()
            
            # Update TTL
            if extend_ttl:
                existing["expires_at"] = (
                    datetime.now() + timedelta(seconds=self.ttl)
                ).isoformat()
            
            redis_key = f"{self.key_prefix}{key}"
            
            # Get remaining TTL
            ttl = await self.redis.ttl(redis_key)
            if ttl < 0:
                ttl = self.ttl
            
            # Store updated data
            await self.redis.setex(
                redis_key,
                ttl if not extend_ttl else self.ttl,
                json.dumps(existing)
            )
            
            self._log_operation("UPDATE", key, True)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update: {e}", exc_info=True)
            self._log_operation("UPDATE", key, False)
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete short-term memory entry
        
        Args:
            key: Conversation ID
            
        Returns:
            bool: True if successful
        """
        try:
            redis_key = f"{self.key_prefix}{key}"
            result = await self.redis.delete(redis_key)
            
            success = result > 0
            self._log_operation("DELETE", key, success)
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete: {e}", exc_info=True)
            self._log_operation("DELETE", key, False)
            return False
    
    async def search(
        self,
        query: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search short-term memory
        
        Args:
            query: Search criteria (e.g., {"lead_id": "L0001"})
            limit: Maximum results
            
        Returns:
            List of matching memory entries
        """
        try:
            pattern = f"{self.key_prefix}*"
            results = []
            
            # Scan all keys
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    if len(results) >= limit:
                        break
                    
                    data = await self.redis.get(key)
                    if data:
                        parsed = json.loads(data)
                        
                        # Check if matches query
                        if self._matches_query(parsed, query):
                            results.append(parsed)
                
                if cursor == 0 or len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search: {e}", exc_info=True)
            return []
    
    def _matches_query(self, data: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if data matches query criteria"""
        for key, value in query.items():
            if key not in data or data[key] != value:
                return False
        return True
    
    async def cleanup(self) -> bool:
        """
        Cleanup expired entries (Redis handles this automatically with TTL)
        This method manually removes entries past their expires_at time
        
        Returns:
            bool: True if successful
        """
        try:
            pattern = f"{self.key_prefix}*"
            expired_count = 0
            
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    data = await self.redis.get(key)
                    if data:
                        parsed = json.loads(data)
                        expires_at = datetime.fromisoformat(parsed["expires_at"])
                        
                        if datetime.now() > expires_at:
                            await self.redis.delete(key)
                            expired_count += 1
                
                if cursor == 0:
                    break
            
            self.logger.info(f"Cleaned up {expired_count} expired entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}", exc_info=True)
            return False
    
    async def extend_ttl(self, key: str, additional_seconds: int = None) -> bool:
        """
        Extend TTL for a conversation
        
        Args:
            key: Conversation ID
            additional_seconds: Additional seconds (uses default TTL if not provided)
            
        Returns:
            bool: True if successful
        """
        try:
            redis_key = f"{self.key_prefix}{key}"
            exists = await self.redis.exists(redis_key)
            
            if not exists:
                return False
            
            await self.redis.expire(redis_key, additional_seconds or self.ttl)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extend TTL: {e}")
            return False
    
    async def get_remaining_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a conversation
        
        Args:
            key: Conversation ID
            
        Returns:
            Remaining seconds (-1 if expired, -2 if doesn't exist)
        """
        try:
            redis_key = f"{self.key_prefix}{key}"
            return await self.redis.ttl(redis_key)
        except Exception as e:
            self.logger.error(f"Failed to get TTL: {e}")
            return -2
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        base_stats = await super().get_stats()
        
        # Get Redis info
        info = await self.redis.info("memory")
        
        return {
            **base_stats,
            "operations": self.stats,
            "redis_memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
            "redis_memory_peak_mb": info.get("used_memory_peak", 0) / 1024 / 1024,
            "default_ttl_seconds": self.ttl
        }
    
    async def _check_backend_health(self) -> bool:
        """Check Redis connection health"""
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.logger.info("Short-term memory connection closed")