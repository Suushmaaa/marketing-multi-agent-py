"""
MCP Server Implementation
Provides resources, tools, and prompts to AI agents
"""
import asyncio
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime
import uuid
from sqlalchemy.orm import Session

from .protocol import (
    MCPProtocolHandler, JSONRPCRequest, JSONRPCResponse,
    JSONRPCError, JSONRPCErrorCode, MCPResource, MCPTool, MCPPrompt
)
from .transport import BaseTransport, TransportManager
from database.db_manager import DatabaseManager
from database.models import (
    Campaign, Lead, Interaction, AgentAction,
    MCPResource as MCPResourceModel
)

logger = logging.getLogger(__name__)

class MCPServer:
    """MCP Server for Sales AI System"""

    def __init__(
        self,
        db_manager: DatabaseManager,
        transport_manager: TransportManager
    ):
        self.db_manager = db_manager
        self.transport_manager = transport_manager
        self.protocol_handler = MCPProtocolHandler()

        # Server state
        self.running = False
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # resource_uri -> session_ids

        # Register resources, tools, and prompts
        self.resources: Dict[str, MCPResource] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.prompts: Dict[str, MCPPrompt] = {}

        self._register_default_resources()
        self._register_default_tools()
        self._register_default_prompts()

        # Override protocol handler methods
        self._setup_protocol_handlers()

    def _setup_protocol_handlers(self) -> None:
        """Setup protocol handler overrides"""
        self.protocol_handler._handle_resources_list = self.handle_resources_list
        self.protocol_handler._handle_resources_read = self.handle_resources_read
        self.protocol_handler._handle_resources_subscribe = self.handle_resources_subscribe
        self.protocol_handler._handle_resources_unsubscribe = self.handle_resources_unsubscribe
        self.protocol_handler._handle_tools_list = self.handle_tools_list
        self.protocol_handler._handle_tools_call = self.handle_tools_call
        self.protocol_handler._handle_prompts_list = self.handle_prompts_list
        self.protocol_handler._handle_prompts_get = self.handle_prompts_get

    def _register_default_resources(self) -> None:
        """Register default MCP resources"""
        # Campaign resources
        self.resources["campaigns://all"] = MCPResource(
            uri="campaigns://all",
            name="All Campaigns",
            description="List of all marketing campaigns",
            mime_type="application/json"
        )

        self.resources["campaigns://{id}"] = MCPResource(
            uri="campaigns://{id}",
            name="Campaign Details",
            description="Detailed information about a specific campaign",
            mime_type="application/json"
        )

        # Lead resources
        self.resources["leads://all"] = MCPResource(
            uri="leads://all",
            name="All Leads",
            description="List of all leads",
            mime_type="application/json"
        )

        self.resources["leads://{id}"] = MCPResource(
            uri="leads://{id}",
            name="Lead Details",
            description="Detailed information about a specific lead",
            mime_type="application/json"
        )

        self.resources["leads://status/{status}"] = MCPResource(
            uri="leads://status/{status}",
            name="Leads by Status",
            description="Leads filtered by status",
            mime_type="application/json"
        )

        # Interaction resources
        self.resources["interactions://lead/{lead_id}"] = MCPResource(
            uri="interactions://lead/{lead_id}",
            name="Lead Interactions",
            description="All interactions for a specific lead",
            mime_type="application/json"
        )

    def _register_default_tools(self) -> None:
        """Register default MCP tools"""
        # Lead scoring tool
        self.tools["score_lead"] = MCPTool(
            name="score_lead",
            description="Calculate lead score based on behavior and demographics",
            input_schema={
                "type": "object",
                "properties": {
                    "lead_id": {
                        "type": "string",
                        "description": "The lead ID to score"
                    }
                },
                "required": ["lead_id"]
            }
        )

        # Engagement tool
        self.tools["send_message"] = MCPTool(
            name="send_message",
            description="Send a message to a lead via specified channel",
            input_schema={
                "type": "object",
                "properties": {
                    "lead_id": {"type": "string"},
                    "channel": {
                        "type": "string",
                        "enum": ["email", "sms", "whatsapp", "phone"],
                        "description": "Communication channel"
                    },
                    "message": {
                        "type": "string",
                        "description": "Message content"
                    },
                    "template_id": {
                        "type": "string",
                        "description": "Optional template ID to use"
                    }
                },
                "required": ["lead_id", "channel", "message"]
            }
        )

        # Campaign optimization tool
        self.tools["optimize_campaign"] = MCPTool(
            name="optimize_campaign",
            description="Analyze and optimize campaign performance",
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "string"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to optimize for"
                    }
                },
                "required": ["campaign_id"]
            }
        )

        # Lead qualification tool
        self.tools["qualify_lead"] = MCPTool(
            name="qualify_lead",
            description="Qualify a lead based on criteria",
            input_schema={
                "type": "object",
                "properties": {
                    "lead_id": {"type": "string"},
                    "criteria": {
                        "type": "object",
                        "description": "Qualification criteria"
                    }
                },
                "required": ["lead_id"]
            }
        )

    def _register_default_prompts(self) -> None:
        """Register default MCP prompts"""
        # Lead engagement prompt
        self.prompts["engage_lead"] = MCPPrompt(
            name="engage_lead",
            description="Generate personalized engagement message for a lead",
            arguments=[
                {"name": "lead_name", "type": "string", "required": True},
                {"name": "lead_context", "type": "string", "required": True},
                {"name": "goal", "type": "string", "required": False}
            ],
            template="""You are engaging with lead {lead_name}.

Context: {lead_context}

Goal: {goal}

Generate a personalized, warm message that:
1. Addresses their specific needs/interests
2. Provides value
3. Includes a clear call-to-action
4. Maintains professional yet friendly tone"""
        )

        # Campaign analysis prompt
        self.prompts["analyze_campaign"] = MCPPrompt(
            name="analyze_campaign",
            description="Analyze campaign performance and provide insights",
            arguments=[
                {"name": "campaign_data", "type": "object", "required": True},
                {"name": "time_period", "type": "string", "required": True}
            ],
            template="""Analyze the following campaign performance data:

{campaign_data}

Time Period: {time_period}

Provide:
1. Key performance metrics summary
2. Trends and patterns identified
3. Areas of success
4. Areas for improvement
5. Actionable recommendations"""
        )

        # Lead qualification prompt
        self.prompts["qualify_lead"] = MCPPrompt(
            name="qualify_lead",
            description="Assess lead qualification based on data",
            arguments=[
                {"name": "lead_data", "type": "object", "required": True},
                {"name": "qualification_criteria", "type": "object", "required": True}
            ],
            template="""Assess the following lead for qualification:

Lead Data: {lead_data}

Qualification Criteria: {qualification_criteria}

Evaluate and provide:
1. Overall qualification score (0-100)
2. Strengths (why they're a good fit)
3. Concerns (potential issues)
4. Recommended next actions
5. Priority level (High/Medium/Low)"""
        )

    async def start(self) -> None:
        """Start the MCP server"""
        logger.info("Starting MCP Server...")
        self.running = True

        # Connect all transports
        await self.transport_manager.connect_all()

        # Get active transport
        transport = self.transport_manager.get_transport()
        if transport:
            transport.add_message_handler(self.handle_message)
            transport.add_error_handler(self.handle_error)

        logger.info("MCP Server started successfully")

    async def stop(self) -> None:
        """Stop the MCP server"""
        logger.info("Stopping MCP Server...")
        self.running = False

        # Disconnect all transports
        await self.transport_manager.disconnect_all()

        # Clear sessions
        self.sessions.clear()
        self.subscriptions.clear()

        logger.info("MCP Server stopped")

    async def handle_message(self, message: str) -> None:
        """Handle incoming JSON-RPC message"""
        try:
            # Parse request
            request = JSONRPCRequest.from_json(message)

            # Validate request
            error = self.protocol_handler.validate_request(request)
            if error:
                response = self.protocol_handler.create_response(
                    request.id, error=error
                )
                await self.send_response(response)
                return

            # Handle request
            try:
                handler = self.protocol_handler.supported_methods.get(request.method)
                if handler:
                    result = await self._execute_handler(handler, request.params or {})
                    response = self.protocol_handler.create_response(
                        request.id, result=result
                    )
                else:
                    response = self.protocol_handler.create_error_response(
                        request.id,
                        JSONRPCErrorCode.METHOD_NOT_FOUND,
                        f"Method not found: {request.method}"
                    )

                await self.send_response(response)

            except Exception as e:
                logger.error(f"Error handling request: {e}")
                response = self.protocol_handler.create_error_response(
                    request.id,
                    JSONRPCErrorCode.INTERNAL_ERROR,
                    str(e)
                )
                await self.send_response(response)

        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            response = self.protocol_handler.create_error_response(
                None,
                JSONRPCErrorCode.PARSE_ERROR,
                "Failed to parse JSON-RPC request"
            )
            await self.send_response(response)

    async def _execute_handler(
        self,
        handler: Callable,
        params: Dict[str, Any]
    ) -> Any:
        """Execute handler (sync or async)"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(params)
        else:
            return handler(params)

    async def send_response(self, response: JSONRPCResponse) -> None:
        """Send JSON-RPC response"""
        transport = self.transport_manager.get_transport()
        if transport:
            await transport.send(response.to_json())

    async def handle_error(self, error: Exception) -> None:
        """Handle transport errors"""
        logger.error(f"Transport error: {error}")

    # Resource handlers
    async def handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available resources"""
        cursor = params.get("cursor")

        resource_list = [
            resource.to_dict()
            for resource in self.resources.values()
        ]

        return {
            "resources": resource_list,
            "nextCursor": None  # Implement pagination if needed
        }

    async def handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a specific resource"""
        uri = params.get("uri")
        if not uri:
            raise ValueError("Resource URI is required")

        # Parse URI and fetch data
        data = await self._fetch_resource_data(uri)

        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": str(data)
            }]
        }

    async def handle_resources_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Subscribe to resource updates"""
        uri = params.get("uri")
        session_id = params.get("sessionId", str(uuid.uuid4()))

        if uri not in self.subscriptions:
            self.subscriptions[uri] = []

        if session_id not in self.subscriptions[uri]:
            self.subscriptions[uri].append(session_id)

        logger.info(f"Session {session_id} subscribed to {uri}")
        return {"success": True}

    async def handle_resources_unsubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unsubscribe from resource updates"""
        uri = params.get("uri")
        session_id = params.get("sessionId")

        if uri in self.subscriptions and session_id in self.subscriptions[uri]:
            self.subscriptions[uri].remove(session_id)
            logger.info(f"Session {session_id} unsubscribed from {uri}")

        return {"success": True}

    # Tool handlers
    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools"""
        tool_list = [
            tool.to_dict()
            for tool in self.tools.values()
        ]

        return {"tools": tool_list}

    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool"""
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})

        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        # Execute tool
        result = await self._execute_tool(tool_name, tool_params)

        return {
            "content": [{
                "type": "text",
                "text": str(result)
            }]
        }

    # Prompt handlers
    async def handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available prompts"""
        prompt_list = [
            prompt.to_dict()
            for prompt in self.prompts.values()
        ]

        return {"prompts": prompt_list}

    async def handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific prompt"""
        prompt_name = params.get("name")
        prompt_args = params.get("arguments", {})

        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_name}")

        prompt = self.prompts[prompt_name]

        # Fill template with arguments
        filled_template = prompt.template
        for key, value in prompt_args.items():
            filled_template = filled_template.replace(f"{{{key}}}", str(value))

        return {
            "description": prompt.description,
            "messages": [{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": filled_template
                }
            }]
        }

    # Helper methods
    async def _fetch_resource_data(self, uri: str) -> Any:
        """Fetch data for a resource URI"""
        with self.db_manager.get_session() as session:
            # Parse URI
            if uri.startswith("campaigns://all"):
                campaigns = session.query(Campaign).all()
                return [self._campaign_to_dict(c) for c in campaigns]

            elif uri.startswith("campaigns://"):
                campaign_id = uri.split("/")[-1]
                campaign = session.query(Campaign).filter_by(
                    campaign_id=campaign_id
                ).first()
                return self._campaign_to_dict(campaign) if campaign else None

            elif uri.startswith("leads://all"):
                leads = session.query(Lead).all()
                return [self._lead_to_dict(l) for l in leads]

            elif uri.startswith("leads://status/"):
                status = uri.split("/")[-1]
                leads = session.query(Lead).filter_by(status=status).all()
                return [self._lead_to_dict(l) for l in leads]

            elif uri.startswith("leads://"):
                lead_id = uri.split("/")[-1]
                lead = session.query(Lead).filter_by(lead_id=lead_id).first()
                return self._lead_to_dict(lead) if lead else None

            elif uri.startswith("interactions://lead/"):
                lead_id = uri.split("/")[-1]
                interactions = session.query(Interaction).filter_by(
                    lead_id=lead_id
                ).all()
                return [self._interaction_to_dict(i) for i in interactions]

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters"""
        with self.db_manager.get_session() as session:
            if tool_name == "score_lead":
                return await self._tool_score_lead(session, params)
            elif tool_name == "send_message":
                return await self._tool_send_message(session, params)
            elif tool_name == "optimize_campaign":
                return await self._tool_optimize_campaign(session, params)
            elif tool_name == "qualify_lead":
                return await self._tool_qualify_lead(session, params)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

    async def _tool_score_lead(self, session: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        """Score a lead"""
        lead_id = params["lead_id"]
        lead = session.query(Lead).filter_by(lead_id=lead_id).first()

        if not lead:
            raise ValueError(f"Lead not found: {lead_id}")

        # Simple scoring algorithm (enhance as needed)
        score = 50  # Base score

        # Engagement factor
        interactions = session.query(Interaction).filter_by(lead_id=lead_id).count()
        score += min(interactions * 5, 30)

        # Status factor
        status_scores = {
            "new": 10,
            "contacted": 20,
            "qualified": 30,
            "proposal": 40,
            "negotiation": 50
        }
        score += status_scores.get(lead.status, 0)

        # Update lead score
        lead.score = min(score, 100)
        session.commit()

        return {
            "lead_id": lead_id,
            "score": lead.score,
            "factors": {
                "interactions": interactions,
                "status": lead.status
            }
        }

    async def _tool_send_message(self, session: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to a lead"""
        # Log the interaction
        interaction = Interaction(
            interaction_id=str(uuid.uuid4()),
            lead_id=params["lead_id"],
            channel=params["channel"],
            direction="outbound",
            message_content=params["message"],
            timestamp=datetime.utcnow()
        )
        session.add(interaction)
        session.commit()

        return {
            "success": True,
            "interaction_id": interaction.interaction_id,
            "timestamp": interaction.timestamp.isoformat()
        }

    async def _tool_optimize_campaign(self, session: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a campaign"""
        campaign_id = params["campaign_id"]
        campaign = session.query(Campaign).filter_by(campaign_id=campaign_id).first()

        if not campaign:
            raise ValueError(f"Campaign not found: {campaign_id}")

        # Calculate metrics
        leads_count = session.query(Lead).filter_by(campaign_id=campaign_id).count()
        interactions_count = session.query(Interaction).join(Lead).filter(
            Lead.campaign_id == campaign_id
        ).count()

        engagement_rate = (interactions_count / leads_count * 100) if leads_count > 0 else 0

        return {
            "campaign_id": campaign_id,
            "metrics": {
                "total_leads": leads_count,
                "total_interactions": interactions_count,
                "engagement_rate": round(engagement_rate, 2)
            },
            "recommendations": [
                "Increase touchpoints for low-engagement segments",
                "Test different messaging for better response rates",
                "Focus on high-value lead segments"
            ]
        }

    async def _tool_qualify_lead(self, session: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        """Qualify a lead"""
        lead_id = params["lead_id"]
        lead = session.query(Lead).filter_by(lead_id=lead_id).first()

        if not lead:
            raise ValueError(f"Lead not found: {lead_id}")

        # Qualification logic
        qualified = lead.score >= 60 and lead.status in ["qualified", "proposal", "negotiation"]

        if qualified and lead.status == "new":
            lead.status = "qualified"
            session.commit()

        return {
            "lead_id": lead_id,
            "qualified": qualified,
            "score": lead.score,
            "status": lead.status,
            "recommendation": "Move to qualified" if qualified else "Nurture further"
        }

    # Model conversion helpers
    def _campaign_to_dict(self, campaign: Campaign) -> Dict[str, Any]:
        """Convert Campaign model to dict"""
        return {
            "campaign_id": campaign.campaign_id,
            "name": campaign.name,
            "type": campaign.type,
            "status": campaign.status,
            "start_date": campaign.start_date.isoformat() if campaign.start_date else None,
            "end_date": campaign.end_date.isoformat() if campaign.end_date else None
        }

    def _lead_to_dict(self, lead: Lead) -> Dict[str, Any]:
        """Convert Lead model to dict"""
        return {
            "lead_id": lead.lead_id,
            "campaign_id": lead.campaign_id,
            "name": lead.name,
            "email": lead.email,
            "phone": lead.phone,
            "status": lead.status,
            "score": lead.score,
            "source": lead.source
        }

    def _interaction_to_dict(self, interaction: Interaction) -> Dict[str, Any]:
        """Convert Interaction model to dict"""
        return {
            "interaction_id": interaction.interaction_id,
            "lead_id": interaction.lead_id,
            "channel": interaction.channel,
            "direction": interaction.direction,
            "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None,
            "sentiment": interaction.sentiment
        }
