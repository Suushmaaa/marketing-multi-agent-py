"""
Episodic Memory Implementation
Stores successful problem-resolution patterns and action sequences
Enables learning from past experiences
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import aiosqlite
from collections import defaultdict

from .base_memory import BaseMemory
from config.settings import settings, agent_config


class EpisodicMemorySystem(BaseMemory):
    """
    Episodic memory for storing successful interaction patterns.
    Agents can query this to learn from past successes.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize episodic memory

        Args:
            db_path: Path to SQLite database
        """
        super().__init__(memory_type="episodic")
        self.db_path = db_path or str(settings.BASE_DIR / "episodic_memory.db")
        self.db_conn = None
        self.max_episodes = agent_config["EPISODIC_MEMORY_MAX_EPISODES"]
        self.stats = {
            "total_episodes": 0,
            "total_queries": 0,
            "successful_matches": 0
        }

    async def initialize(self) -> bool:
        """Initialize database and create tables"""
        try:
            self.db_conn = await aiosqlite.connect(self.db_path)

            # Create episodes table
            await self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    episode_id TEXT PRIMARY KEY,
                    scenario TEXT NOT NULL,
                    action_sequence_json TEXT NOT NULL,
                    outcome_score REAL NOT NULL,
                    notes TEXT,
                    lead_context TEXT,
                    campaign_context TEXT,
                    agent_type TEXT,
                    frequency_used INTEGER DEFAULT 0,
                    last_used_at TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            ''')

            # Create indexes
            await self.db_conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_scenario ON episodic_memory(scenario)'
            )
            await self.db_conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_outcome_score ON episodic_memory(outcome_score DESC)'
            )
            await self.db_conn.commit()
            self.is_initialized = True
            self.logger.info("Episodic memory initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize episodic memory: {e}")
            return False

    async def store(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """Store an episode"""
        try:
            await self.db_conn.execute('''
                INSERT OR REPLACE INTO episodic_memory (
                    episode_id, scenario, action_sequence_json, outcome_score,
                    notes, lead_context, campaign_context, agent_type,
                    frequency_used, last_used_at, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                key,
                data.get("scenario"),
                json.dumps(data.get("action_sequence", [])),
                data.get("outcome_score", 0.0),
                data.get("notes"),
                data.get("lead_context"),
                data.get("campaign_context"),
                data.get("agent_type"),
                data.get("frequency_used", 0),
                data.get("last_used_at"),
                data.get("created_at"),
                json.dumps(data.get("metadata", {}))
            ))
            await self.db_conn.commit()
            self.stats["total_episodes"] += 1
            self.logger.info(f"Stored episode {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store episode {key}: {e}")
            return False

    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve an episode by ID"""
        try:
            cursor = await self.db_conn.execute('''
                SELECT * FROM episodic_memory WHERE episode_id = ?
            ''', (key,))
            row = await cursor.fetchone()
            if row:
                return {
                    "episode_id": row[0],
                    "scenario": row[1],
                    "action_sequence": json.loads(row[2]),
                    "outcome_score": row[3],
                    "notes": row[4],
                    "lead_context": row[5],
                    "campaign_context": row[6],
                    "agent_type": row[7],
                    "frequency_used": row[8],
                    "last_used_at": row[9],
                    "created_at": row[10],
                    "metadata": json.loads(row[11]) if row[11] else {}
                }
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve episode {key}: {e}")
            return None

    async def update(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """Update an episode"""
        return await self.store(key, data, **kwargs)

    async def delete(self, key: str) -> bool:
        """Delete an episode"""
        try:
            await self.db_conn.execute('DELETE FROM episodic_memory WHERE episode_id = ?', (key,))
            await self.db_conn.commit()
            self.stats["total_episodes"] -= 1
            self.logger.info(f"Deleted episode {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete episode {key}: {e}")
            return False

    async def search(self, query: Dict[str, Any], limit: int = 10, **kwargs) -> list:
        """Search episodes matching query"""
        try:
            # Simple search by scenario substring match
            scenario = query.get("scenario", "")
            cursor = await self.db_conn.execute('''
                SELECT * FROM episodic_memory
                WHERE scenario LIKE ?
                ORDER BY outcome_score DESC
                LIMIT ?
            ''', (f"%{scenario}%", limit))
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "episode_id": row[0],
                    "scenario": row[1],
                    "action_sequence": json.loads(row[2]),
                    "outcome_score": row[3],
                    "notes": row[4],
                    "lead_context": row[5],
                    "campaign_context": row[6],
                    "agent_type": row[7],
                    "frequency_used": row[8],
                    "last_used_at": row[9],
                    "created_at": row[10],
                    "metadata": json.loads(row[11]) if row[11] else {}
                })
            self.stats["total_queries"] += 1
            self.stats["successful_matches"] += len(results)
            return results
        except Exception as e:
            self.logger.error(f"Failed to search episodes: {e}")
            return []

    async def enforce_max_episodes(self) -> None:
        """Enforce max episodes limit by deleting oldest"""
        try:
            cursor = await self.db_conn.execute('SELECT COUNT(*) FROM episodic_memory')
            count = (await cursor.fetchone())[0]
            if count > self.max_episodes:
                to_delete = count - self.max_episodes
                await self.db_conn.execute('''
                    DELETE FROM episodic_memory
                    WHERE episode_id IN (
                        SELECT episode_id FROM episodic_memory
                        ORDER BY last_used_at ASC, created_at ASC
                        LIMIT ?
                    )
                ''', (to_delete,))
                await self.db_conn.commit()
                self.stats["total_episodes"] = self.max_episodes
                self.logger.info(f"Enforced max episodes limit: removed {to_delete} episodes")
        except Exception as e:
            self.logger.error(f"Failed to enforce max episodes: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        base_stats = await super().get_stats()
        try:
            cursor = await self.db_conn.execute('SELECT AVG(outcome_score) FROM episodic_memory')
            avg_score = (await cursor.fetchone())[0] or 0
            cursor = await self.db_conn.execute('''
                SELECT scenario, COUNT(*) as count
                FROM episodic_memory
                GROUP BY scenario
                ORDER BY count DESC
                LIMIT 5
            ''')
            top_scenarios = await cursor.fetchall()
            return {
                **base_stats,
                "total_episodes": self.stats["total_episodes"],
                "max_episodes": self.max_episodes,
                "average_outcome_score": round(avg_score, 3),
                "query_stats": {
                    "total_queries": self.stats["total_queries"],
                    "successful_matches": self.stats["successful_matches"]
                },
                "top_scenarios": [
                    {"scenario": s[0], "count": s[1]}
                    for s in top_scenarios
                ]
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return base_stats

    async def _check_backend_health(self) -> bool:
        """Check database health"""
        try:
            cursor = await self.db_conn.execute('SELECT 1')
            await cursor.fetchone()
            return True
        except Exception:
            return False

    async def cleanup(self) -> bool:
        """Cleanup expired or unnecessary data"""
        try:
            await self.enforce_max_episodes()
            return True
        except Exception as e:
            self.logger.error(f"Failed to cleanup episodic memory: {e}")
            return False

    async def close(self):
        """Close database connection"""
        if self.db_conn:
            await self.db_conn.close()
            self.logger.info("Episodic memory connection closed")
