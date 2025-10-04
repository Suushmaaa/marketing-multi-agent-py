"""
Long-Term Memory Implementation
Stores persistent customer preferences, history, and behavioral patterns
Uses SQLite/PostgreSQL for durable storage
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import sqlite3
import aiosqlite

from .base_memory import BaseMemory
from database.models import LongTermMemory
from config.settings import settings


class LongTermMemorySystem(BaseMemory):
    """
    Long-term memory for customer preferences and history.
    Persists across sessions and consolidates from short-term memory.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize long-term memory
        
        Args:
            db_path: Path to SQLite database
        """
        super().__init__(memory_type="long_term")
        self.db_path = db_path or str(settings.BASE_DIR / "long_term_memory.db")
        self.db_conn = None
        self.stats = {
            "total_stores": 0,
            "total_retrievals": 0,
            "total_updates": 0,
            "total_consolidations": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize database connection and create tables"""
        try:
            self.db_conn = await aiosqlite.connect(self.db_path)
            
            # Create table
            await self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    lead_id TEXT PRIMARY KEY,
                    region TEXT,
                    industry TEXT,
                    rfm_score REAL,
                    preferences_json TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    total_value_usd REAL DEFAULT 0,
                    last_interaction_at TEXT,
                    last_updated_at TEXT,
                    created_at TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create indexes
            await self.db_conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_region ON long_term_memory(region)'
            )
            await self.db_conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_industry ON long_term_memory(industry)'
            )
            await self.db_conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_rfm_score ON long_term_memory(rfm_score)'
            )
            
            await self.db_conn.commit()
            
            self.is_initialized = True
            self.logger.info("Long-term memory initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            return False
    
    async def store(
        self,
        key: str,
        data: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Store customer data in long-term memory
        
        Args:
            key: Lead ID
            data: Customer data
            
        Returns:
            bool: True if successful
        """
        try:
            # Validate data
            memory_obj = LongTermMemory(
                lead_id=key,
                region=data.get("region", ""),
                industry=data.get("industry", ""),
                rfm_score=data.get("rfm_score", 0.5),
                preferences_json=data.get("preferences_json", {}),
                last_updated_at=datetime.now()
            )
            
            now = datetime.now().isoformat()
            
            await self.db_conn.execute('''
                INSERT INTO long_term_memory (
                    lead_id, region, industry, rfm_score,
                    preferences_json, interaction_count, total_value_usd,
                    last_interaction_at, last_updated_at, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                key,
                memory_obj.region,
                memory_obj.industry,
                memory_obj.rfm_score,
                json.dumps(memory_obj.preferences_json),
                data.get("interaction_count", 0),
                data.get("total_value_usd", 0.0),
                data.get("last_interaction_at", now),
                now,
                now,
                json.dumps(data.get("metadata", {}))
            ))
            
            await self.db_conn.commit()
            
            self.stats["total_stores"] += 1
            self._log_operation("STORE", key, True)
            return True
            
        except sqlite3.IntegrityError:
            # Already exists, update instead
            return await self.update(key, data)
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
        Retrieve customer data from long-term memory
        
        Args:
            key: Lead ID
            
        Returns:
            Customer data or None if not found
        """
        try:
            cursor = await self.db_conn.execute(
                'SELECT * FROM long_term_memory WHERE lead_id = ?',
                (key,)
            )
            
            row = await cursor.fetchone()
            
            if row:
                self.stats["total_retrievals"] += 1
                
                # Convert to dict
                columns = [description[0] for description in cursor.description]
                data = dict(zip(columns, row))
                
                # Parse JSON fields
                data["preferences_json"] = json.loads(data["preferences_json"])
                data["metadata"] = json.loads(data.get("metadata", "{}"))
                
                self._log_operation("RETRIEVE", key, True)
                return data
            else:
                self._log_operation("RETRIEVE", key, False)
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve: {e}", exc_info=True)
            return None
    
    async def update(
        self,
        key: str,
        data: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Update existing long-term memory
        
        Args:
            key: Lead ID
            data: Updated data
            
        Returns:
            bool: True if successful
        """
        try:
            # Get existing data
            existing = await self.retrieve(key)
            
            if not existing:
                # Create new if doesn't exist
                return await self.store(key, data)
            
            # Merge preferences
            if "preferences_json" in data:
                existing["preferences_json"].update(data["preferences_json"])
                data["preferences_json"] = existing["preferences_json"]
            
            # Build update query
            update_fields = []
            update_values = []
            
            allowed_fields = [
                "region", "industry", "rfm_score", "preferences_json",
                "interaction_count", "total_value_usd", "last_interaction_at",
                "metadata"
            ]
            
            for field in allowed_fields:
                if field in data:
                    update_fields.append(f"{field} = ?")
                    
                    if field in ["preferences_json", "metadata"]:
                        update_values.append(json.dumps(data[field]))
                    else:
                        update_values.append(data[field])
            
            # Always update last_updated_at
            update_fields.append("last_updated_at = ?")
            update_values.append(datetime.now().isoformat())
            
            update_values.append(key)
            
            query = f'''
                UPDATE long_term_memory 
                SET {", ".join(update_fields)}
                WHERE lead_id = ?
            '''
            
            await self.db_conn.execute(query, update_values)
            await self.db_conn.commit()
            
            self.stats["total_updates"] += 1
            self._log_operation("UPDATE", key, True)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update: {e}", exc_info=True)
            self._log_operation("UPDATE", key, False)
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete long-term memory entry
        
        Args:
            key: Lead ID
            
        Returns:
            bool: True if successful
        """
        try:
            cursor = await self.db_conn.execute(
                'DELETE FROM long_term_memory WHERE lead_id = ?',
                (key,)
            )
            await self.db_conn.commit()
            
            success = cursor.rowcount > 0
            self._log_operation("DELETE", key, success)
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete: {e}", exc_info=True)
            self._log_operation("DELETE", key, False)
            return False
    
    async def search(
        self,
        query: Dict[str, Any],
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search long-term memory
        
        Args:
            query: Search criteria
            limit: Maximum results
            
        Returns:
            List of matching memory entries
        """
        try:
            where_clauses = []
            params = []
            
            # Build WHERE clause
            for key, value in query.items():
                if key in ["region", "industry", "lead_id"]:
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                elif key == "min_rfm_score":
                    where_clauses.append("rfm_score >= ?")
                    params.append(value)
                elif key == "max_rfm_score":
                    where_clauses.append("rfm_score <= ?")
                    params.append(value)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            sql = f'''
                SELECT * FROM long_term_memory
                WHERE {where_sql}
                ORDER BY last_updated_at DESC
                LIMIT ?
            '''
            
            params.append(limit)
            
            cursor = await self.db_conn.execute(sql, params)
            rows = await cursor.fetchall()
            
            results = []
            columns = [description[0] for description in cursor.description]
            
            for row in rows:
                data = dict(zip(columns, row))
                data["preferences_json"] = json.loads(data["preferences_json"])
                data["metadata"] = json.loads(data.get("metadata", "{}"))
                results.append(data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search: {e}", exc_info=True)
            return []
    
    async def cleanup(self) -> bool:
        """
        Cleanup old or unused entries
        
        Returns:
            bool: True if successful
        """
        try:
            # Delete entries with no recent interactions (older than 2 years)
            cutoff = datetime.now().replace(year=datetime.now().year - 2).isoformat()
            
            cursor = await self.db_conn.execute('''
                DELETE FROM long_term_memory
                WHERE last_interaction_at < ?
                AND interaction_count = 0
            ''', (cutoff,))
            
            await self.db_conn.commit()
            
            deleted_count = cursor.rowcount
            self.logger.info(f"Cleaned up {deleted_count} inactive entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}", exc_info=True)
            return False
    
    async def consolidate_from_interactions(
        self,
        lead_id: str,
        interactions: List[Dict[str, Any]]
    ) -> bool:
        """
        Consolidate interaction data into long-term memory
        
        Args:
            lead_id: Lead ID
            interactions: List of interaction records
            
        Returns:
            bool: True if successful
        """
        try:
            if not interactions:
                return False
            
            # Calculate aggregations
            interaction_count = len(interactions)
            
            # Channel preferences
            channel_counts = {}
            for interaction in interactions:
                channel = interaction.get("channel", "Unknown")
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
            
            preferred_channel = max(channel_counts, key=channel_counts.get)
            
            # Time preferences (best time to contact)
            time_preferences = []
            for interaction in interactions:
                if "timestamp" in interaction:
                    ts = datetime.fromisoformat(interaction["timestamp"].replace('Z', '+00:00'))
                    time_preferences.append(ts.hour)
            
            avg_hour = sum(time_preferences) / len(time_preferences) if time_preferences else 12
            
            # Get latest interaction
            latest_interaction = max(
                interactions,
                key=lambda x: x.get("timestamp", "")
            )
            
            # Calculate RFM score (Recency, Frequency, Monetary)
            # Simplified version
            days_since_last = (
                datetime.now() - 
                datetime.fromisoformat(latest_interaction["timestamp"].replace('Z', '+00:00'))
            ).days
            
            recency_score = max(0, 1 - (days_since_last / 365))
            frequency_score = min(1, interaction_count / 50)
            
            rfm_score = (recency_score + frequency_score) / 2
            
            # Build preferences
            preferences = {
                "preferred_channel": preferred_channel,
                "channel_distribution": channel_counts,
                "best_contact_hour": int(avg_hour),
                "avg_response_time_hours": 24,  # Could be calculated from data
                "engagement_level": "high" if interaction_count > 10 else "medium" if interaction_count > 3 else "low"
            }
            
            # Update long-term memory
            update_data = {
                "rfm_score": rfm_score,
                "preferences_json": preferences,
                "interaction_count": interaction_count,
                "last_interaction_at": latest_interaction.get("timestamp"),
                "metadata": {
                    "last_consolidation": datetime.now().isoformat(),
                    "consolidation_source": "interactions"
                }
            }
            
            result = await self.update(lead_id, update_data)
            
            if result:
                self.stats["total_consolidations"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Consolidation failed: {e}", exc_info=True)
            return False
    
    async def get_high_value_leads(
        self,
        min_rfm_score: float = 0.7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get high-value leads based on RFM score
        
        Args:
            min_rfm_score: Minimum RFM score threshold
            limit: Maximum results
            
        Returns:
            List of high-value leads
        """
        return await self.search(
            query={"min_rfm_score": min_rfm_score},
            limit=limit
        )
    
    async def get_leads_by_segment(
        self,
        region: Optional[str] = None,
        industry: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get leads by segment
        
        Args:
            region: Geographic region
            industry: Industry vertical
            limit: Maximum results
            
        Returns:
            List of leads
        """
        query = {}
        if region:
            query["region"] = region
        if industry:
            query["industry"] = industry
        
        return await self.search(query=query, limit=limit)
    
    async def update_rfm_score(self, lead_id: str, new_score: float) -> bool:
        """
        Update RFM score for a lead
        
        Args:
            lead_id: Lead ID
            new_score: New RFM score (0-1)
            
        Returns:
            bool: True if successful
        """
        if not 0 <= new_score <= 1:
            self.logger.error(f"Invalid RFM score: {new_score}")
            return False
        
        return await self.update(lead_id, {"rfm_score": new_score})
    
    async def increment_interaction_count(self, lead_id: str) -> bool:
        """
        Increment interaction count for a lead
        
        Args:
            lead_id: Lead ID
            
        Returns:
            bool: True if successful
        """
        try:
            await self.db_conn.execute('''
                UPDATE long_term_memory
                SET interaction_count = interaction_count + 1,
                    last_interaction_at = ?,
                    last_updated_at = ?
                WHERE lead_id = ?
            ''', (
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                lead_id
            ))
            
            await self.db_conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to increment count: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        base_stats = await super().get_stats()
        
        try:
            # Get total records
            cursor = await self.db_conn.execute(
                'SELECT COUNT(*) FROM long_term_memory'
            )
            total_records = (await cursor.fetchone())[0]
            
            # Get average RFM score
            cursor = await self.db_conn.execute(
                'SELECT AVG(rfm_score) FROM long_term_memory'
            )
            avg_rfm = (await cursor.fetchone())[0] or 0
            
            # Get records by engagement level
            cursor = await self.db_conn.execute('''
                SELECT 
                    SUM(CASE WHEN interaction_count > 10 THEN 1 ELSE 0 END) as high,
                    SUM(CASE WHEN interaction_count BETWEEN 4 AND 10 THEN 1 ELSE 0 END) as medium,
                    SUM(CASE WHEN interaction_count <= 3 THEN 1 ELSE 0 END) as low
                FROM long_term_memory
            ''')
            engagement = await cursor.fetchone()
            
            return {
                **base_stats,
                "total_records": total_records,
                "average_rfm_score": round(avg_rfm, 3),
                "engagement_distribution": {
                    "high": engagement[0] or 0,
                    "medium": engagement[1] or 0,
                    "low": engagement[2] or 0
                },
                "operations": self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return base_stats
    
    async def _check_backend_health(self) -> bool:
        """Check database connection health"""
        try:
            cursor = await self.db_conn.execute('SELECT 1')
            await cursor.fetchone()
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close database connection"""
        if self.db_conn:
            await self.db_conn.close()
            self.logger.info("Long-term memory connection closed")