"""
Memory Consolidation System
Manages data flow between different memory layers
Implements algorithms for moving data from short-term to long-term memory
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging

from .short_term import ShortTermMemorySystem
from .long_term import LongTermMemorySystem
from .episodic import EpisodicMemorySystem

from .semantic import SemanticMemorySystem
from config.settings import settings


class MemoryConsolidationEngine:
    """
    Engine for consolidating memories across different layers.
    Runs periodic consolidation tasks.
    """
    
    def __init__(
        self,
        short_term: ShortTermMemorySystem,
        long_term: LongTermMemorySystem,
        episodic: EpisodicMemorySystem,
        semantic: SemanticMemorySystem
    ):
        """
        Initialize consolidation engine
        
        Args:
            short_term: Short-term memory system
            long_term: Long-term memory system
            episodic: Episodic memory system
            semantic: Semantic memory system
        """
        self.short_term = short_term
        self.long_term = long_term
        self.episodic = episodic
        self.semantic = semantic
        
        self.logger = logging.getLogger("memory.consolidation")
        self.is_running = False
        self.consolidation_task = None
        
        self.stats = {
            "total_consolidations": 0,
            "short_to_long": 0,
            "interactions_to_episodic": 0,
            "facts_to_semantic": 0,
            "failed_consolidations": 0
        }
    
    async def start(self):
        """Start the consolidation engine"""
        if self.is_running:
            self.logger.warning("Consolidation engine already running")
            return
        
        self.is_running = True
        self.consolidation_task = asyncio.create_task(self._consolidation_loop())
        self.logger.info("Memory consolidation engine started")
    
    async def stop(self):
        """Stop the consolidation engine"""
        self.is_running = False
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Memory consolidation engine stopped")
    
    async def _consolidation_loop(self):
        """Main consolidation loop"""
        while self.is_running:
            try:
                self.logger.info("Starting memory consolidation cycle")
                
                # Run consolidation tasks
                await self.consolidate_short_to_long()
                await self.consolidate_interactions_to_episodic()
                await self.extract_semantic_knowledge()
                
                # Cleanup expired data
                await self.cleanup_all_memories()
                
                self.logger.info("Memory consolidation cycle completed")
                
                # Wait for next cycle
                await asyncio.sleep(settings.MEMORY_CONSOLIDATION_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Consolidation cycle error: {e}", exc_info=True)
                self.stats["failed_consolidations"] += 1
                await asyncio.sleep(60)  # Wait before retry
    
    async def consolidate_short_to_long(self) -> int:
        """
        Consolidate data from short-term to long-term memory
        
        Returns:
            Number of records consolidated
        """
        try:
            consolidated = 0
            
            # Get all short-term memories
            pattern = f"{self.short_term.key_prefix}*"
            cursor = 0
            
            while True:
                cursor, keys = await self.short_term.redis.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    try:
                        # Get short-term data
                        stm_data = await self.short_term.redis.get(key)
                        if not stm_data:
                            continue
                        
                        import json
                        stm_dict = json.loads(stm_data)
                        
                        lead_id = stm_dict.get("lead_id")
                        if not lead_id:
                            continue
                        
                        # Check if should consolidate
                        if await self._should_consolidate_to_long_term(stm_dict):
                            # Get or create long-term memory
                            ltm_data = await self.long_term.retrieve(lead_id)
                            
                            if not ltm_data:
                                # Create new long-term memory
                                ltm_data = {
                                    "lead_id": lead_id,
                                    "preferences_json": stm_dict.get("slots_json", {}),
                                    "interaction_count": 1
                                }
                            else:
                                # Update existing
                                ltm_data["preferences_json"].update(
                                    stm_dict.get("slots_json", {})
                                )
                                await self.long_term.increment_interaction_count(lead_id)
                            
                            # Store in long-term memory
                            await self.long_term.update(lead_id, ltm_data)
                            consolidated += 1
                            
                    except Exception as e:
                        self.logger.error(f"Failed to consolidate key {key}: {e}")
                        continue
                
                if cursor == 0:
                    break
            
            self.stats["short_to_long"] += consolidated
            self.stats["total_consolidations"] += consolidated
            self.logger.info(f"Consolidated {consolidated} short-term memories to long-term")
            
            return consolidated
            
        except Exception as e:
            self.logger.error(f"Short-to-long consolidation failed: {e}", exc_info=True)
            return 0
    
    async def _should_consolidate_to_long_term(self, stm_data: Dict[str, Any]) -> bool:
        """
        Determine if short-term memory should be consolidated
        
        Args:
            stm_data: Short-term memory data
            
        Returns:
            bool: True if should consolidate
        """
        # Consolidate if conversation is closed or idle
        # Or if memory is about to expire
        expires_at = datetime.fromisoformat(stm_data.get("expires_at", ""))
        time_until_expiry = (expires_at - datetime.now()).total_seconds()
        
        # Consolidate if less than 10 minutes until expiry
        if time_until_expiry < 600:
            return True
        
        # Consolidate if there's valuable information in slots
        slots = stm_data.get("slots_json", {})
        if len(slots) > 3:  # Has meaningful extracted data
            return True
        
        return False
    
    async def consolidate_interactions_to_episodic(self) -> int:
        """
        Extract successful patterns from interactions and store in episodic memory
        
        Returns:
            Number of episodes created
        """
        try:
            # This would typically analyze recent interactions from the database
            # For now, we'll implement a placeholder
            created = 0
            
            # In production: Query recent successful conversions/interactions
            # Analyze action sequences that led to positive outcomes
            # Store as episodic memories
            
            self.stats["interactions_to_episodic"] += created
            self.stats["total_consolidations"] += created
            
            if created > 0:
                self.logger.info(f"Created {created} new episodic memories")
            
            return created
            
        except Exception as e:
            self.logger.error(f"Episodic consolidation failed: {e}", exc_info=True)
            return 0
    
    async def extract_semantic_knowledge(self) -> int:
        """
        Extract facts and relationships for knowledge graph
        
        Returns:
            Number of facts extracted
        """
        try:
            extracted = 0
            
            # Extract relationships from long-term memory
            # For example: lead preferences, industry trends, etc.
            
            # Example: Extract industry-persona relationships
            ltm_records = await self.long_term.search({}, limit=1000)
            
            industry_persona_pairs = {}
            for record in ltm_records:
                industry = record.get("industry")
                # Extract persona from preferences if available
                prefs = record.get("preferences_json", {})
                
                if industry:
                    key = industry
                    if key not in industry_persona_pairs:
                        industry_persona_pairs[key] = {"count": 0, "conversions": 0}
                    industry_persona_pairs[key]["count"] += 1
            
            # Store top relationships in knowledge graph
            for industry, stats in industry_persona_pairs.items():
                if stats["count"] >= 5:  # Minimum threshold
                    weight = min(1.0, stats["count"] / 100)
                    
                    await self.semantic.store(
                        key=f"industry_stat_{industry}",
                        data={
                            "subject": industry,
                            "predicate": "has_lead_count",
                            "object": str(stats["count"]),
                            "weight": weight,
                            "source": "consolidation"
                        }
                    )
                    extracted += 1
            
            self.stats["facts_to_semantic"] += extracted
            self.stats["total_consolidations"] += extracted
            
            if extracted > 0:
                self.logger.info(f"Extracted {extracted} semantic facts")
            
            return extracted
            
        except Exception as e:
            self.logger.error(f"Semantic extraction failed: {e}", exc_info=True)
            return 0
    
    async def cleanup_all_memories(self):
        """Run cleanup on all memory systems"""
        try:
            self.logger.info("Running memory cleanup")
            
            # Cleanup short-term (expired entries)
            await self.short_term.cleanup()
            
            # Cleanup long-term (old inactive records)
            await self.long_term.cleanup()
            
            # Cleanup episodic (low-performing episodes)
            await self.episodic.cleanup()
            
            # Cleanup semantic (low-weight relationships)
            if self.semantic and self.semantic.is_initialized:
                await self.semantic.cleanup()
            
            self.logger.info("Memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}", exc_info=True)
        
    async def consolidate_lead_history(
        self,
        lead_id: str,
        interactions: List[Dict[str, Any]]
    ) -> bool:
        """
        Consolidate interaction history for a specific lead
        
        Args:
            lead_id: Lead ID
            interactions: List of interaction records
            
        Returns:
            bool: True if successful
        """
        try:
            # Consolidate to long-term memory
            success = await self.long_term.consolidate_from_interactions(
                lead_id,
                interactions
            )
            
            if success:
                self.stats["short_to_long"] += 1
                self.stats["total_consolidations"] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Lead history consolidation failed: {e}", exc_info=True)
            return False
    
    async def create_episodic_memory_from_success(
        self,
        scenario: str,
        actions: List[Dict[str, Any]],
        outcome_score: float,
        context: Dict[str, Any]
    ) -> bool:
        """
        Create episodic memory from successful interaction
        
        Args:
            scenario: Scenario description
            actions: Sequence of actions taken
            outcome_score: Success score (0-1)
            context: Additional context
            
        Returns:
            bool: True if successful
        """
        try:
            import uuid
            episode_id = str(uuid.uuid4())
            
            episode_data = {
                "episode_id": episode_id,
                "scenario": scenario,
                "action_sequence_json": actions,
                "outcome_score": outcome_score,
                "notes": f"Automatically generated from successful interaction",
                "lead_context": context.get("lead_context", {}),
                "campaign_context": context.get("campaign_context", {}),
                "agent_type": context.get("agent_type", "Unknown"),
                "metadata": {
                    "created_by": "consolidation_engine",
                    "created_at": datetime.now().isoformat()
                }
            }
            
            success = await self.episodic.store(episode_id, episode_data)
            
            if success:
                self.stats["interactions_to_episodic"] += 1
                self.stats["total_consolidations"] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create episodic memory: {e}", exc_info=True)
            return False
    
    async def extract_knowledge_from_episode(
        self,
        episode_id: str
    ) -> int:
        """
        Extract semantic knowledge from an episodic memory
        
        Args:
            episode_id: Episode ID
            
        Returns:
            Number of facts extracted
        """
        try:
            # Get episode
            episode = await self.episodic.retrieve(episode_id)
            if not episode:
                return 0
            
            extracted = 0
            
            # Extract relationships from action sequence
            actions = episode.get("action_sequence_json", [])
            scenario = episode.get("scenario", "")
            outcome_score = episode.get("outcome_score", 0)
            
            # Only extract from successful episodes
            if outcome_score < 0.7:
                return 0
            
            # Example: Extract action effectiveness
            for i, action in enumerate(actions):
                action_type = action.get("type")
                if action_type:
                    await self.semantic.store(
                        key=f"action_{episode_id}_{i}",
                        data={
                            "subject": scenario,
                            "predicate": "effective_action",
                            "object": action_type,
                            "weight": outcome_score,
                            "source": f"episode_{episode_id}"
                        }
                    )
                    extracted += 1
            
            self.stats["facts_to_semantic"] += extracted
            
            return extracted
            
        except Exception as e:
            self.logger.error(f"Knowledge extraction failed: {e}", exc_info=True)
            return 0
    
    async def get_consolidated_lead_profile(
        self,
        lead_id: str
    ) -> Dict[str, Any]:
        """
        Get complete consolidated profile for a lead from all memory layers
        
        Args:
            lead_id: Lead ID
            
        Returns:
            Comprehensive lead profile
        """
        try:
            profile = {
                "lead_id": lead_id,
                "short_term": None,
                "long_term": None,
                "relevant_episodes": [],
                "knowledge_connections": []
            }
            
            # Get short-term memory (active conversation)
            stm_results = await self.short_term.search({"lead_id": lead_id}, limit=1)
            if stm_results:
                profile["short_term"] = stm_results[0]
            
            # Get long-term memory
            profile["long_term"] = await self.long_term.retrieve(lead_id)
            
            # Get relevant episodic memories
            if profile["long_term"]:
                industry = profile["long_term"].get("industry", "")
                if industry:
                    episodes = await self.episodic.search(
                        {"scenario": industry, "min_outcome_score": 0.7},
                        limit=5
                    )
                    profile["relevant_episodes"] = episodes
            
            # Get semantic knowledge connections
            # This would query the knowledge graph for relevant information
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to get consolidated profile: {e}", exc_info=True)
            return {"lead_id": lead_id, "error": str(e)}
    
    async def optimize_memory_usage(self):
        """
        Optimize memory usage across all layers
        """
        try:
            self.logger.info("Optimizing memory usage")
            
            # Get stats from all memory systems
            stm_stats = await self.short_term.get_stats()
            ltm_stats = await self.long_term.get_stats()
            episodic_stats = await self.episodic.get_stats()
            semantic_stats = await self.semantic.get_stats()
            
            # Short-term optimization
            if stm_stats.get("operations", {}).get("cache_misses", 0) > 100:
                # High cache miss rate - might need to extend TTLs
                self.logger.warning("High short-term memory cache miss rate")
            
            # Long-term optimization
            total_records = ltm_stats.get("total_records", 0)
            if total_records > 100000:
                # Large database - run cleanup
                await self.long_term.cleanup()
            
            # Episodic optimization
            total_episodes = episodic_stats.get("total_episodes", 0)
            max_episodes = episodic_stats.get("max_episodes", 1000)
            
            if total_episodes > max_episodes * 0.9:
                # Near capacity - cleanup low-performing episodes
                await self.episodic.cleanup()
            
            # Semantic optimization
            total_rels = semantic_stats.get("total_relationships", 0)
            if total_rels > 50000:
                # Large graph - remove low-weight edges
                await self.semantic.cleanup()
            
            self.logger.info("Memory optimization completed")
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}", exc_info=True)
    
    async def backup_memories(self, backup_path: str) -> bool:
        """
        Create backup of all memory systems
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            bool: True if successful
        """
        try:
            import json
            from pathlib import Path
            
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup long-term memory
            ltm_data = await self.long_term.search({}, limit=10000)
            ltm_file = backup_dir / f"long_term_memory_{timestamp}.json"
            with open(ltm_file, 'w') as f:
                json.dump(ltm_data, f, indent=2)
            
            # Backup episodic memory
            episodic_data = await self.episodic.search({}, limit=10000)
            episodic_file = backup_dir / f"episodic_memory_{timestamp}.json"
            with open(episodic_file, 'w') as f:
                json.dump(episodic_data, f, indent=2)
            
            # Backup semantic knowledge (export as triples)
            semantic_data = await self.semantic.search({}, limit=10000)
            semantic_file = backup_dir / f"semantic_memory_{timestamp}.json"
            with open(semantic_file, 'w') as f:
                json.dump(semantic_data, f, indent=2)
            
            self.logger.info(f"Memory backup created at {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory backup failed: {e}", exc_info=True)
            return False
    
    async def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore memories from backup
        
        Args:
            backup_path: Path to backup files
            
        Returns:
            bool: True if successful
        """
        try:
            import json
            from pathlib import Path
            
            backup_dir = Path(backup_path)
            
            # Find latest backup files
            ltm_files = sorted(backup_dir.glob("long_term_memory_*.json"))
            episodic_files = sorted(backup_dir.glob("episodic_memory_*.json"))
            semantic_files = sorted(backup_dir.glob("semantic_memory_*.json"))
            
            if not ltm_files:
                self.logger.error("No backup files found")
                return False
            
            # Restore long-term memory
            with open(ltm_files[-1], 'r') as f:
                ltm_data = json.load(f)
            
            for record in ltm_data:
                lead_id = record.get("lead_id")
                if lead_id:
                    await self.long_term.store(lead_id, record)
            
            # Restore episodic memory
            if episodic_files:
                with open(episodic_files[-1], 'r') as f:
                    episodic_data = json.load(f)
                
                for episode in episodic_data:
                    episode_id = episode.get("episode_id")
                    if episode_id:
                        await self.episodic.store(episode_id, episode)
            
            # Restore semantic memory
            if semantic_files:
                with open(semantic_files[-1], 'r') as f:
                    semantic_data = json.load(f)
                
                # Batch import for efficiency
                if semantic_data:
                    await self.semantic.store_batch(semantic_data)
            
            self.logger.info(f"Memory restored from backup: {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory restore failed: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation engine statistics"""
        return {
            "is_running": self.is_running,
            "consolidations": self.stats,
            "consolidation_interval_seconds": settings.MEMORY_CONSOLIDATION_INTERVAL
        }


class MemoryManager:
    """
    High-level memory manager that orchestrates all memory systems
    """
    
    def __init__(self):
        self.short_term: Optional[ShortTermMemorySystem] = None
        self.long_term: Optional[LongTermMemorySystem] = None
        self.episodic: Optional[EpisodicMemorySystem] = None
        self.semantic: Optional[SemanticMemorySystem] = None
        self.consolidation: Optional[MemoryConsolidationEngine] = None
        
        self.logger = logging.getLogger("memory.manager")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all memory systems"""
        try:
            self.logger.info("Initializing memory systems...")
            
            # Initialize each memory layer
            self.short_term = ShortTermMemorySystem()
            await self.short_term.initialize()
            
            self.long_term = LongTermMemorySystem()
            await self.long_term.initialize()
            
            self.episodic = EpisodicMemorySystem()
            await self.episodic.initialize()
            
            self.semantic = SemanticMemorySystem()
            await self.semantic.initialize()
            
            # Initialize consolidation engine
            self.consolidation = MemoryConsolidationEngine(
                self.short_term,
                self.long_term,
                self.episodic,
                self.semantic
            )
            
            # Start consolidation engine
            await self.consolidation.start()
            
            self.is_initialized = True
            self.logger.info("All memory systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory system initialization failed: {e}", exc_info=True)
            return False
    
    async def shutdown(self):
        """Shutdown all memory systems"""
        try:
            self.logger.info("Shutting down memory systems...")
            
            # Stop consolidation engine
            if self.consolidation:
                await self.consolidation.stop()
            
            # Close connections
            if self.short_term:
                await self.short_term.close()
            
            if self.long_term:
                await self.long_term.close()
            
            if self.episodic:
                await self.episodic.close()
            
            if self.semantic:
                await self.semantic.close()
            
            self.logger.info("Memory systems shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Memory system shutdown error: {e}", exc_info=True)
    
    async def is_healthy(self) -> bool:
        """Check if all memory systems are healthy"""
        if not self.is_initialized:
            return False
        
        try:
            return all([
                await self.short_term.is_healthy(),
                await self.long_term.is_healthy(),
                await self.episodic.is_healthy(),
                await self.semantic.is_healthy()
            ])
        except Exception:
            return False
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all memory systems"""
        return {
            "short_term": await self.short_term.get_stats() if self.short_term else {},
            "long_term": await self.long_term.get_stats() if self.long_term else {},
            "episodic": await self.episodic.get_stats() if self.episodic else {},
            "semantic": await self.semantic.get_stats() if self.semantic else {},
            "consolidation": self.consolidation.get_stats() if self.consolidation else {}
        }