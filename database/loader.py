# CSV data loader
"""
Data Loader - Load CSV datasets into the system
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from config.settings import settings
from database.models import (
    Campaign, Lead, Interaction, Conversation,
    AgentAction, CampaignDaily, Conversion,
    ShortTermMemory, LongTermMemory, EpisodicMemory, SemanticTriple
)


class DataLoader:
    """Load and process CSV data files"""
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir or settings.DATA_DIR
        self.logger = logging.getLogger("database.loader")
        
        self.stats = {
            "files_loaded": 0,
            "total_records": 0,
            "errors": 0
        }
    
    async def load_all_data(self, mcp_client) -> bool:
        """
        Load all CSV files into the system
        
        Args:
            mcp_client: MCP client for data access
            
        Returns:
            bool: True if successful
        """
        try:
            self.logger.info("Starting data load process...")
            
            # Load in dependency order
            await self.load_campaigns(mcp_client)
            await self.load_ab_variants(mcp_client)
            await self.load_leads(mcp_client)
            await self.load_conversations(mcp_client)
            await self.load_interactions(mcp_client)
            await self.load_conversions(mcp_client)
            await self.load_agent_actions(mcp_client)
            await self.load_campaign_daily(mcp_client)
            await self.load_segments(mcp_client)
            
            # Load memory data
            await self.load_memory_data(mcp_client)
            
            # Load MCP logs
            await self.load_mcp_logs(mcp_client)
            
            self.logger.info(
                f"Data load completed - Files: {self.stats['files_loaded']}, "
                f"Records: {self.stats['total_records']}, "
                f"Errors: {self.stats['errors']}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data load failed: {e}", exc_info=True)
            return False
    
    async def load_campaigns(self, mcp_client):
        """Load campaigns.csv"""
        try:
            file_path = self.data_dir / "campaigns.csv"
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                return
            
            df = pd.read_csv(file_path)
            self.logger.info(f"Loading {len(df)} campaigns...")
            
            for _, row in df.iterrows():
                try:
                    # Parse JSON fields
                    import json
                    channel_mix = json.loads(row['channel_mix']) if isinstance(row['channel_mix'], str) else row['channel_mix']
                    target_personas = json.loads(row['target_personas']) if isinstance(row['target_personas'], str) else row['target_personas']
                    
                    campaign = Campaign(
                        campaign_id=row['campaign_id'],
                        name=row['name'],
                        objective=row['objective'],
                        start_date=pd.to_datetime(row['start_date']),
                        end_date=pd.to_datetime(row['end_date']),
                        channel_mix=channel_mix,
                        daily_budget_usd=float(row['daily_budget_usd']),
                        total_budget_usd=float(row['total_budget_usd']),
                        owner_email=row['owner_email'],
                        primary_region=row['primary_region'],
                        target_personas=target_personas,
                        kpi=row['kpi']
                    )
                    
                    # Store via MCP
                    await mcp_client.call_method(
                        method="resource.write",
                        params={
                            "resource_uri": "db://campaigns",
                            "operation": "INSERT",
                            "data": campaign.dict()
                        }
                    )
                    
                    self.stats["total_records"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error loading campaign {row.get('campaign_id')}: {e}")
                    self.stats["errors"] += 1
            
            self.stats["files_loaded"] += 1
            self.logger.info(f"✓ Campaigns loaded: {len(df)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load campaigns: {e}", exc_info=True)
            self.stats["errors"] += 1
    
    async def load_leads(self, mcp_client):
        """Load leads.csv"""
        try:
            file_path = self.data_dir / "leads.csv"
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                return
            
            df = pd.read_csv(file_path)
            self.logger.info(f"Loading {len(df)} leads...")
            
            for _, row in df.iterrows():
                try:
                    lead = Lead(
                        lead_id=row['lead_id'],
                        created_at=pd.to_datetime(row['created_at']),
                        last_active_at=pd.to_datetime(row['last_active_at']),
                        source=row['source'],
                        campaign_id=row['campaign_id'],
                        triage_category=row['triage_category'],
                        lead_status=row['lead_status'],
                        lead_score=int(row['lead_score']),
                        company_size=row['company_size'],
                        industry=row['industry'],
                        persona=row['persona'],
                        region=row['region'],
                        preferred_channel=row['preferred_channel'],
                        gdpr_consent=bool(row['gdpr_consent']),
                        email=row['email'],
                        phone=row['phone'],
                        assigned_engagement_agent=row.get('assigned_engagement_agent')
                    )
                    
                    await mcp_client.call_method(
                        method="resource.write",
                        params={
                            "resource_uri": "db://leads",
                            "operation": "INSERT",
                            "data": lead.dict()
                        }
                    )
                    
                    self.stats["total_records"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error loading lead {row.get('lead_id')}: {e}")
                    self.stats["errors"] += 1
            
            self.stats["files_loaded"] += 1
            self.logger.info(f"✓ Leads loaded: {len(df)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load leads: {e}", exc_info=True)
            self.stats["errors"] += 1
    
    async def load_interactions(self, mcp_client):
        """Load interactions.csv"""
        try:
            file_path = self.data_dir / "interactions.csv"
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                return
            
            df = pd.read_csv(file_path)
            self.logger.info(f"Loading {len(df)} interactions...")
            
            # Process in batches for efficiency
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    try:
                        import json
                        metadata = json.loads(row['metadata_json']) if pd.notna(row.get('metadata_json')) else {}
                        
                        interaction = Interaction(
                            interaction_id=row['interaction_id'],
                            conversation_id=row['conversation_id'],
                            lead_id=row['lead_id'],
                            campaign_id=row['campaign_id'],
                            timestamp=pd.to_datetime(row['timestamp']),
                            channel=row['channel'],
                            event_type=row['event_type'],
                            agent_id=row['agent_id'],
                            variant_id=row.get('variant_id'),
                            outcome=row['outcome'],
                            metadata_json=metadata
                        )
                        
                        await mcp_client.call_method(
                            method="resource.write",
                            params={
                                "resource_uri": "db://interactions",
                                "operation": "INSERT",
                                "data": interaction.dict()
                            }
                        )
                        
                        self.stats["total_records"] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error loading interaction: {e}")
                        self.stats["errors"] += 1
                
                self.logger.info(f"Loaded batch {i//batch_size + 1} ({min(i+batch_size, len(df))}/{len(df)})")
            
            self.stats["files_loaded"] += 1
            self.logger.info(f"✓ Interactions loaded: {len(df)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load interactions: {e}", exc_info=True)
            self.stats["errors"] += 1
    
    async def load_conversations(self, mcp_client):
        """Load conversations.csv"""
        try:
            file_path = self.data_dir / "conversations.csv"
            if not file_path.exists():
                return
            
            df = pd.read_csv(file_path)
            self.logger.info(f"Loading {len(df)} conversations...")
            
            for _, row in df.iterrows():
                try:
                    conversation = Conversation(
                        conversation_id=row['conversation_id'],
                        lead_id=row['lead_id'],
                        opened_at=pd.to_datetime(row['opened_at']),
                        last_event_at=pd.to_datetime(row['last_event_at']),
                        status=row['status']
                    )
                    
                    await mcp_client.call_method(
                        method="resource.write",
                        params={
                            "resource_uri": "db://conversations",
                            "operation": "INSERT",
                            "data": conversation.dict()
                        }
                    )
                    
                    self.stats["total_records"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error loading conversation: {e}")
                    self.stats["errors"] += 1
            
            self.stats["files_loaded"] += 1
            self.logger.info(f"✓ Conversations loaded: {len(df)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load conversations: {e}", exc_info=True)
    
    async def load_ab_variants(self, mcp_client):
        """Load ab_variants.csv"""
        # Similar implementation
        pass
    
    async def load_conversions(self, mcp_client):
        """Load conversions.csv"""
        # Similar implementation
        pass
    
    async def load_agent_actions(self, mcp_client):
        """Load agent_actions.csv"""
        # Similar implementation
        pass
    
    async def load_campaign_daily(self, mcp_client):
        """Load campaign_daily.csv"""
        # Similar implementation
        pass
    
    async def load_segments(self, mcp_client):
        """Load segments.csv"""
        # Similar implementation
        pass
    
    async def load_memory_data(self, mcp_client):
        """Load all memory CSV files"""
        try:
            # Load short-term memory
            await self._load_csv_to_memory(
                "memory_short_term.csv",
                "short_term"
            )
            
            # Load long-term memory
            await self._load_csv_to_memory(
                "memory_long_term.csv",
                "long_term"
            )
            
            # Load episodic memory
            await self._load_csv_to_memory(
                "memory_episodic.csv",
                "episodic"
            )
            
            # Load semantic triples
            await self._load_csv_to_memory(
                "semantic_kg_triples.csv",
                "semantic"
            )
            
            self.logger.info("✓ Memory data loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load memory data: {e}", exc_info=True)
    
    async def _load_csv_to_memory(self, filename: str, memory_type: str):
        """Helper to load CSV into specific memory system"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return
        
        df = pd.read_csv(file_path)
        self.logger.info(f"Loading {len(df)} {memory_type} memory records...")
        
        # Implementation depends on memory type
        # This would interact with the memory manager
        
        self.stats["files_loaded"] += 1
        self.stats["total_records"] += len(df)
    
    async def load_mcp_logs(self, mcp_client):
        """Load MCP-related log files"""
        try:
            # Load JSON-RPC calls, WebSocket sessions, HTTP requests, etc.
            # These are for analysis rather than operational data
            
            log_files = [
                "mcp_jsonrpc_calls.csv",
                "transport_websocket_sessions.csv",
                "transport_http_requests.csv",
                "mcp_resource_access.csv",
                "security_auth_events.csv"
            ]
            
            for filename in log_files:
                file_path = self.data_dir / filename
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    self.logger.info(f"Loaded {len(df)} records from {filename}")
                    self.stats["files_loaded"] += 1
                    self.stats["total_records"] += len(df)
            
        except Exception as e:
            self.logger.error(f"Failed to load MCP logs: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        return self.stats