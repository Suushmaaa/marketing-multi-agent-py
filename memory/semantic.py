"""
Semantic Memory Implementation using Knowledge Graph
Stores domain knowledge as interconnected triples (subject-predicate-object)
Uses Neo4j for graph database storage
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
import logging

from .base_memory import BaseMemory
from database.models import SemanticTriple
from config.settings import settings


class SemanticMemorySystem(BaseMemory):
    """
    Semantic memory using knowledge graph (Neo4j).
    Stores domain knowledge and relationships for reasoning.
    """
    
    def __init__(self, driver: Optional[AsyncDriver] = None):
        """
        Initialize semantic memory
        
        Args:
            driver: Neo4j driver instance
        """
        super().__init__(memory_type="semantic", storage_backend=driver)
        self.driver = driver
        self.stats = {
            "total_nodes": 0,
            "total_relationships": 0,
            "total_queries": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize Neo4j connection and create constraints"""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )

        try:
            await self.driver.verify_connectivity()
            self.is_initialized = True
            self.logger.info("Semantic memory initialized successfully")
        except Exception as e:
            self.logger.warning(f"Neo4j unavailable - semantic memory disabled: {e}")
            self.is_initialized = False

        if self.is_initialized:
            # Create indexes and constraints
            async with self.driver.session() as session:
                # Create constraint on node IDs
                await session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity)
                    REQUIRE n.id IS UNIQUE
                """)

                # Create index on entity names
                await session.run("""
                    CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)
                """)

                # Create index on relationship types
                await session.run("""
                    CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)
                """)

                # Get initial counts
                result = await session.run("MATCH (n) RETURN count(n) as count")
                record = await result.single()
                self.stats["total_nodes"] = record["count"] if record else 0

                result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = await result.single()
                self.stats["total_relationships"] = record["count"] if record else 0

            self.logger.info(
                f"Semantic memory initialized - "
                f"Nodes: {self.stats['total_nodes']}, "
                f"Relationships: {self.stats['total_relationships']}"
            )

        return True

    async def store(
        self,
        key: str,
        data: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Store a single triple in semantic memory

        Args:
            key: Triple ID
            data: Triple data (subject, predicate, object, weight, source)

        Returns:
            bool: True if successful
        """
        try:
            triple = {
                "id": key,
                "subject": data.get("subject"),
                "predicate": data.get("predicate"),
                "object": data.get("object"),
                "weight": data.get("weight", 1.0),
                "source": data.get("source", "direct_store"),
                "created_at": datetime.now().isoformat()
            }

            return await self.store_batch([triple])

        except Exception as e:
            self.logger.error(f"Failed to store triple: {e}", exc_info=True)
            return False

    async def store_batch(
        self,
        triples: List[Dict[str, Any]]
    ) -> bool:
        """
        Store multiple triples in batch for efficiency
        
        Args:
            triples: List of triple dictionaries
            
        Returns:
            bool: True if successful
        """
        try:
            async with self.driver.session() as session:
                query = """
                UNWIND $triples as triple
                MERGE (s:Entity {name: triple.subject})
                MERGE (o:Entity {name: triple.object})
                MERGE (s)-[r:RELATES_TO {
                    type: triple.predicate,
                    weight: triple.weight,
                    source: triple.source,
                    created_at: triple.created_at,
                    id: triple.id
                }]->(o)
                """
                
                formatted_triples = [
                    {
                        "id": f"triple_{i}",
                        "subject": t.get("subject"),
                        "object": t.get("object"),
                        "predicate": t.get("predicate"),
                        "weight": t.get("weight", 1.0),
                        "source": t.get("source", "batch_import"),
                        "created_at": datetime.now().isoformat()
                    }
                    for i, t in enumerate(triples)
                ]
                
                await session.run(query, triples=formatted_triples)
                
                self.stats["total_relationships"] += len(triples)
                self.logger.info(f"Stored {len(triples)} triples in batch")
                return True
                
        except Exception as e:
            self.logger.error(f"Batch store failed: {e}", exc_info=True)
            return False
    
    async def retrieve(
        self,
        key: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific triple by ID
        
        Args:
            key: Triple ID
            
        Returns:
            Triple data or None
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (s)-[r:RELATES_TO {id: $triple_id}]->(o)
                RETURN s.name as subject, 
                       r.type as predicate, 
                       o.name as object,
                       r.weight as weight,
                       r.source as source,
                       r.created_at as created_at
                """
                
                result = await session.run(query, triple_id=key)
                record = await result.single()
                
                if record:
                    self._log_operation("RETRIEVE", key, True)
                    return dict(record)
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
        Update an existing triple's properties
        
        Args:
            key: Triple ID
            data: Updated data
            
        Returns:
            bool: True if successful
        """
        try:
            async with self.driver.session() as session:
                # Build SET clause
                set_clauses = []
                params = {"triple_id": key}
                
                if "weight" in data:
                    set_clauses.append("r.weight = $weight")
                    params["weight"] = data["weight"]
                
                if "source" in data:
                    set_clauses.append("r.source = $source")
                    params["source"] = data["source"]
                
                if not set_clauses:
                    return False
                
                query = f"""
                MATCH (s)-[r:RELATES_TO {{id: $triple_id}}]->(o)
                SET {', '.join(set_clauses)},
                    r.updated_at = $updated_at
                RETURN r
                """
                
                params["updated_at"] = datetime.now().isoformat()
                
                result = await session.run(query, **params)
                record = await result.single()
                
                success = record is not None
                self._log_operation("UPDATE", key, success)
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to update: {e}", exc_info=True)
            self._log_operation("UPDATE", key, False)
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a triple
        
        Args:
            key: Triple ID
            
        Returns:
            bool: True if successful
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH ()-[r:RELATES_TO {id: $triple_id}]->()
                DELETE r
                RETURN count(r) as deleted
                """
                
                result = await session.run(query, triple_id=key)
                record = await result.single()
                
                success = record["deleted"] > 0 if record else False
                if success:
                    self.stats["total_relationships"] -= 1
                
                self._log_operation("DELETE", key, success)
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to delete: {e}", exc_info=True)
            return False
    
    async def search(
        self,
        query: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for triples matching criteria
        
        Args:
            query: Search criteria (subject, predicate, object, min_weight)
            limit: Maximum results
            
        Returns:
            List of matching triples
        """
        try:
            where_clauses = []
            params = {"limit": limit}
            
            if "subject" in query:
                where_clauses.append("s.name = $subject")
                params["subject"] = query["subject"]
            
            if "predicate" in query:
                where_clauses.append("r.type = $predicate")
                params["predicate"] = query["predicate"]
            
            if "object" in query:
                where_clauses.append("o.name = $object")
                params["object"] = query["object"]
            
            if "min_weight" in query:
                where_clauses.append("r.weight >= $min_weight")
                params["min_weight"] = query["min_weight"]
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "true"
            
            cypher_query = f"""
            MATCH (s)-[r:RELATES_TO]->(o)
            WHERE {where_clause}
            RETURN s.name as subject,
                   r.type as predicate,
                   o.name as object,
                   r.weight as weight,
                   r.source as source,
                   r.id as id
            ORDER BY r.weight DESC
            LIMIT $limit
            """
            
            async with self.driver.session() as session:
                result = await session.run(cypher_query, **params)
                records = await result.data()
                
                self.stats["total_queries"] += 1
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to search: {e}", exc_info=True)
            return []
    
    async def query_neighbors(
        self,
        entity: str,
        depth: int = 1,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities in the knowledge graph
        
        Args:
            entity: Entity name
            depth: Search depth (1-3 recommended)
            direction: "incoming", "outgoing", or "both"
            
        Returns:
            List of connected entities and relationships
        """
        try:
            if direction == "outgoing":
                pattern = "(s)-[r:RELATES_TO*1..{depth}]->(o)"
            elif direction == "incoming":
                pattern = "(s)<-[r:RELATES_TO*1..{depth}]-(o)"
            else:  # both
                pattern = "(s)-[r:RELATES_TO*1..{depth}]-(o)"
            
            query = f"""
            MATCH {pattern.format(depth=depth)}
            WHERE s.name = $entity
            RETURN DISTINCT o.name as neighbor,
                   [rel in r | {{type: rel.type, weight: rel.weight}}] as path
            LIMIT 50
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, entity=entity)
                records = await result.data()
                
                self.stats["total_queries"] += 1
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to query neighbors: {e}", exc_info=True)
            return []
    
    async def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path between two entities
        
        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_depth: Maximum path length
            
        Returns:
            List of triples forming the path, or None if no path exists
        """
        try:
            query = f"""
            MATCH path = shortestPath(
                (s:Entity {{name: $start}})-[r:RELATES_TO*1..{max_depth}]-(e:Entity {{name: $end}})
            )
            RETURN [rel in relationships(path) | {{
                subject: startNode(rel).name,
                predicate: rel.type,
                object: endNode(rel).name,
                weight: rel.weight
            }}] as path
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    start=start_entity,
                    end=end_entity
                )
                record = await result.single()
                
                self.stats["total_queries"] += 1
                
                if record and record["path"]:
                    return record["path"]
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to find path: {e}", exc_info=True)
            return None
    
    async def query_by_pattern(
        self,
        pattern: str,
        params: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute custom Cypher query pattern
        
        Args:
            pattern: Cypher query pattern
            params: Query parameters
            
        Returns:
            Query results
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(pattern, **(params or {}))
                records = await result.data()
                
                self.stats["total_queries"] += 1
                return records
                
        except Exception as e:
            self.logger.error(f"Pattern query failed: {e}", exc_info=True)
            return []
    
    async def get_high_confidence_facts(
        self,
        min_weight: float = 0.8,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get high-confidence knowledge triples
        
        Args:
            min_weight: Minimum confidence weight
            limit: Maximum results
            
        Returns:
            List of high-confidence triples
        """
        return await self.search(
            query={"min_weight": min_weight},
            limit=limit
        )
    
    async def cleanup(self) -> bool:
        """
        Remove orphaned nodes and low-weight relationships
        
        Returns:
            bool: True if successful
        """
        try:
            async with self.driver.session() as session:
                # Delete low-weight relationships
                result = await session.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    WHERE r.weight < 0.3
                    DELETE r
                    RETURN count(r) as deleted
                """)
                record = await result.single()
                deleted_rels = record["deleted"] if record else 0
                
                # Delete orphaned nodes
                result = await session.run("""
                    MATCH (n:Entity)
                    WHERE NOT (n)-[:RELATES_TO]-()
                    DELETE n
                    RETURN count(n) as deleted
                """)
                record = await result.single()
                deleted_nodes = record["deleted"] if record else 0
                
                self.stats["total_relationships"] -= deleted_rels
                self.stats["total_nodes"] -= deleted_nodes
                
                self.logger.info(
                    f"Cleanup: removed {deleted_rels} relationships and "
                    f"{deleted_nodes} orphaned nodes"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}", exc_info=True)
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        base_stats = await super().get_stats()
        
        try:
            async with self.driver.session() as session:
                # Get current counts
                result = await session.run("MATCH (n) RETURN count(n) as count")
                record = await result.single()
                node_count = record["count"] if record else 0
                
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = await result.single()
                rel_count = record["count"] if record else 0
                
                # Get average weight
                result = await session.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    RETURN avg(r.weight) as avg_weight
                """)
                record = await result.single()
                avg_weight = record["avg_weight"] if record else 0
                
                # Get top predicates
                result = await session.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    RETURN r.type as predicate, count(r) as count
                    ORDER BY count DESC
                    LIMIT 5
                """)
                top_predicates = await result.data()
                
                return {
                    **base_stats,
                    "total_nodes": node_count,
                    "total_relationships": rel_count,
                    "average_weight": round(avg_weight, 3),
                    "total_queries": self.stats["total_queries"],
                    "top_predicates": top_predicates
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}", exc_info=True)
            return base_stats
    
    async def _check_backend_health(self) -> bool:
        """Check Neo4j connection health"""
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.logger.info("Semantic memory connection closed")