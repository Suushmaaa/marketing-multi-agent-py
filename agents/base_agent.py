"""
Base Agent Class - Abstract implementation for all agents
Provides common functionality for agent communication, memory access, and MCP integration
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import logging
import uuid
from enum import Enum

from database.models import (
    AgentMessage, HandoffContext, Lead, Interaction,
    AgentType, ActionType, EscalationReason
)
from config.settings import settings, agent_config


class AgentCapability(str, Enum):
    
    LEAD_SCORING = "lead_scoring"
    ENGAGEMENT = "engagement"
    CAMPAIGN_OPTIMIZATION = "campaign_optimization"
    DATA_ANALYSIS = "data_analysis"
    COMMUNICATION = "communication"
    MEMORY_ACCESS = "memory_access"

class AgentStatus(str, Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class BaseAgent(ABC):
    """
    Abstract base class for all marketing agents.
    Implements common functionality and enforces interface contracts.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        settings=None,
        mcp_client=None,
        memory_manager=None
    ):
        """
        Initialize base agent

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (LeadTriage/Engagement/Optimizer)
            settings: Settings object
            mcp_client: MCP client for data access
            memory_manager: Memory system manager
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.settings = settings
        self.status = AgentStatus.INITIALIZING

        # Dependencies
        self.mcp_client = mcp_client
        self.memory_manager = memory_manager
        
        # State
        self.active_conversations: Dict[str, Any] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Metrics
        self.actions_taken = 0
        self.handoffs_sent = 0
        self.handoffs_received = 0
        self.errors_count = 0
        self.start_time = datetime.now()
        
        # Logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.logger.info(f"Initializing {self.name} ({self.agent_type})")
    
    async def initialize(self) -> bool:
        """
        Initialize agent resources and connections
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info(f"Starting initialization for {self.name}")
            
            # Initialize MCP connection
            if self.mcp_client:
                await self.mcp_client.connect()
                self.logger.info("MCP client connected")
            
            # Initialize memory systems
            if self.memory_manager:
                await self.memory_manager.initialize()
                self.logger.info("Memory manager initialized")
            
            # Agent-specific initialization
            await self._initialize_agent_specific()
            
            self.status = AgentStatus.READY
            self.logger.info(f"{self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            self.errors_count += 1
            return False
    
    async def escalate(
        self,
        context: HandoffContext,
        escalation_reason: EscalationReason,
        details: str
    ) -> bool:
        """
        Escalate a conversation to human manager
        
        Args:
            context: Conversation context
            escalation_reason: Reason for escalation
            details: Additional details
            
        Returns:
            bool: True if escalation successful
        """
        try:
            self.logger.warning(
                f"ESCALATION: Conversation {context.conversation_id} "
                f"- Reason: {escalation_reason} - {details}"
            )
            
            # Log escalation action
            await self._log_action(
                action_type=ActionType.ESCALATE,
                conversation_id=context.conversation_id,
                lead_id=context.lead_id,
                dest_agent_type=AgentType.MANAGER,
                escalation_reason=escalation_reason,
                handoff_context={
                    "context": context.dict(),
                    "details": details
                }
            )
            
            # Send escalation notification (email, Slack, etc.)
            await self._send_escalation_notification(context, escalation_reason, details)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Escalation failed: {e}", exc_info=True)
            self.errors_count += 1
            return False
    
    async def _handle_handoff(self, message: AgentMessage):
        """Handle incoming handoff from another agent"""
        try:
            context_data = message.payload.get("context")
            reason = message.payload.get("reason")
            
            context = HandoffContext(**context_data)
            
            self.logger.info(
                f"Received handoff for conversation {context.conversation_id} "
                f"from {message.from_agent} - Reason: {reason}"
            )
            
            # Add to active conversations
            self.active_conversations[context.conversation_id] = {
                "context": context,
                "received_at": datetime.now(),
                "received_from": message.from_agent
            }
            
            self.handoffs_received += 1
            
            # Process the handed-off conversation
            await self._process_handoff(context, reason)
            
        except Exception as e:
            self.logger.error(f"Error handling handoff: {e}", exc_info=True)
            self.errors_count += 1
    
    @abstractmethod
    async def _process_handoff(self, context: HandoffContext, reason: str):
        """Process handed-off conversation (to be implemented by subclasses)"""
        pass
    
    async def _handle_escalation(self, message: AgentMessage):
        """Handle escalation notification"""
        self.logger.info(f"Received escalation notification: {message.payload}")
        # Implementation depends on agent type
    
    # Memory Access Methods
    async def get_short_term_memory(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get short-term memory for a conversation"""
        if not self.memory_manager:
            return None
        
        return await self.memory_manager.get_short_term(conversation_id)
    
    async def update_short_term_memory(
        self,
        conversation_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Update short-term memory"""
        if not self.memory_manager:
            return False
        
        return await self.memory_manager.update_short_term(conversation_id, data)
    
    async def get_long_term_memory(self, lead_id: str) -> Optional[Dict[str, Any]]:
        """Get long-term memory for a lead"""
        if not self.memory_manager:
            return None
        
        return await self.memory_manager.get_long_term(lead_id)
    
    async def query_episodic_memory(
        self,
        scenario: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query episodic memory for similar scenarios"""
        if not self.memory_manager:
            return []
        
        return await self.memory_manager.query_episodic(scenario, limit)
    
    async def query_semantic_knowledge(
        self,
        query: str,
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Query semantic knowledge graph"""
        if not self.memory_manager:
            return []
        
        return await self.memory_manager.query_semantic(query, depth)
    
    # MCP Data Access Methods
    async def get_lead_data(self, lead_id: str) -> Optional[Lead]:
        """Get lead data via MCP"""
        if not self.mcp_client:
            return None
        
        try:
            response = await self.mcp_client.call_method(
                method="resource.read",
                params={
                    "resource_uri": "db://leads",
                    "filters": {"lead_id": lead_id}
                }
            )
            
            if response and response.get("data"):
                return Lead(**response["data"][0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching lead data: {e}")
            return None
    
    async def get_interaction_history(
        self,
        lead_id: str,
        limit: int = 50
    ) -> List[Interaction]:
        """Get interaction history for a lead"""
        if not self.mcp_client:
            return []
        
        try:
            response = await self.mcp_client.call_method(
                method="resource.search",
                params={
                    "resource_uri": "db://interactions",
                    "filters": {"lead_id": lead_id},
                    "limit": limit,
                    "order_by": "timestamp DESC"
                }
            )
            
            if response and response.get("data"):
                return [Interaction(**item) for item in response["data"]]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching interactions: {e}")
            return []
    
    async def update_lead(self, lead_id: str, updates: Dict[str, Any]) -> bool:
        """Update lead data"""
        if not self.mcp_client:
            return False
        
        try:
            response = await self.mcp_client.call_method(
                method="resource.write",
                params={
                    "resource_uri": "db://leads",
                    "operation": "UPDATE",
                    "filters": {"lead_id": lead_id},
                    "data": updates
                }
            )
            
            return response.get("success", False)
            
        except Exception as e:
            self.logger.error(f"Error updating lead: {e}")
            return False
    
    # Action Logging
    async def _log_action(
        self,
        action_type: ActionType,
        conversation_id: Optional[str] = None,
        lead_id: Optional[str] = None,
        dest_agent_type: AgentType = None,
        escalation_reason: EscalationReason = EscalationReason.NONE,
        handoff_context: Optional[Dict[str, Any]] = None
    ):
        """Log agent action to database"""
        if not self.mcp_client:
            return
        
        try:
            action_data = {
                "action_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "lead_id": lead_id,
                "action_type": action_type.value,
                "source_agent": self.agent_id,
                "source_agent_type": self.agent_type.value,
                "dest_agent_type": dest_agent_type.value if dest_agent_type else None,
                "handoff_context_json": handoff_context,
                "escalation_reason": escalation_reason.value
            }
            
            await self.mcp_client.call_method(
                method="resource.write",
                params={
                    "resource_uri": "db://agent_actions",
                    "operation": "INSERT",
                    "data": action_data
                }
            )
            
            self.actions_taken += 1
            
        except Exception as e:
            self.logger.error(f"Error logging action: {e}")
    
    async def _send_escalation_notification(
        self,
        context: HandoffContext,
        reason: EscalationReason,
        details: str
    ):
        """Send escalation notification to managers"""
        # Implementation: Email, Slack, PagerDuty, etc.
        self.logger.warning(
            f"ESCALATION NOTIFICATION: "
            f"Lead: {context.lead_id}, "
            f"Reason: {reason}, "
            f"Details: {details}"
        )
    
    # Utility Methods
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "status": self.status.value,
            "uptime_seconds": uptime,
            "metrics": {
                "actions_taken": self.actions_taken,
                "handoffs_sent": self.handoffs_sent,
                "handoffs_received": self.handoffs_received,
                "errors_count": self.errors_count,
                "active_conversations": len(self.active_conversations)
            }
        }
    
    def is_available(self) -> bool:
        """Check if agent is available to take new work"""
        return (
            self.status == AgentStatus.READY and
            len(self.active_conversations) < agent_config.MAX_CONCURRENT_AGENTS
        )
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check MCP connection
            if self.mcp_client and not await self.mcp_client.is_connected():
                return False
            
            # Check memory manager
            if self.memory_manager and not await self.memory_manager.is_healthy():
                return False
            
            return self.status in [AgentStatus.READY, AgentStatus.BUSY]
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type}, status={self.status})>"
    
    @abstractmethod
    async def _initialize_agent_specific(self):
        """Agent-specific initialization logic (to be implemented by subclasses)"""
        pass
    
    async def start(self):
        """Start the agent's main processing loop"""
        self.logger.info(f"Starting {self.name}")
        
        if self.status != AgentStatus.READY:
            raise RuntimeError(f"Agent not ready. Current status: {self.status}")
        
        # Start message processing loop
        asyncio.create_task(self._process_message_queue())
        
        self.logger.info(f"{self.name} started successfully")
    
    async def stop(self):
        """Gracefully stop the agent"""
        self.logger.info(f"Stopping {self.name}")
        self.status = AgentStatus.SHUTDOWN
        
        # Close MCP connection
        if self.mcp_client:
            await self.mcp_client.disconnect()
        
        # Cleanup
        await self._cleanup()
        
        self.logger.info(f"{self.name} stopped")
    
    async def _cleanup(self):
        """Agent-specific cleanup (can be overridden)"""
        pass
    
    # Core Processing Methods
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to this agent
        
        Args:
            task: Task data dictionary
            
        Returns:
            Dict containing processing results
        """
        pass
    
    async def _process_message_queue(self):
        """Background task to process incoming messages"""
        while self.status != AgentStatus.SHUTDOWN:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}", exc_info=True)
                self.errors_count += 1
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming agent message"""
        self.logger.debug(f"Received message: {message.message_type}")
        
        if message.message_type == "handoff":
            await self._handle_handoff(message)
        elif message.message_type == "task":
            await self.process_task(message.payload)
        elif message.message_type == "escalation":
            await self._handle_escalation(message)
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
    
    # Agent Communication Methods
    async def send_message(
        self,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """
        Send message to another agent
        
        Args:
            to_agent: Target agent ID
            message_type: Type of message
            payload: Message data
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            The sent message
        """
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            from_agent=self.agent_id,
            to_agent=to_agent,
            timestamp=datetime.now(),
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )
        
        # In production, this would use message broker (Redis, RabbitMQ, etc.)
        # For now, we log it
        self.logger.info(f"Sending {message_type} to {to_agent}")
        
        return message
    
    async def handoff_to_agent(
        self,
        target_agent_type: AgentType,
        context: HandoffContext,
        reason: str
    ) -> bool:
        """
        Hand off a conversation to another agent
        
        Args:
            target_agent_type: Type of agent to hand off to
            context: Handoff context with conversation data
            reason: Reason for handoff
            
        Returns:
            bool: True if handoff successful
        """
        try:
            self.logger.info(
                f"Handing off conversation {context.conversation_id} "
                f"to {target_agent_type} - Reason: {reason}"
            )
            
            # Log the handoff action
            await self._log_action(
                action_type=ActionType.HANDOFF,
                conversation_id=context.conversation_id,
                lead_id=context.lead_id,
                dest_agent_type=target_agent_type,
                handoff_context=context.dict()
            )
            
            # Send handoff message
            await self.send_message(
                to_agent=f"{target_agent_type.value}-ROUTER",
                message_type="handoff",
                payload={
                    "context": context.dict(),
                    "reason": reason
                }
            )
            
            # Remove from active conversations
            if context.conversation_id in self.active_conversations:
                del self.active_conversations[context.conversation_id]
            
            self.handoffs_sent += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Handoff failed: {e}", exc_info=True)
            self.errors_count += 1
            return False