"""
MCP Client Implementation
Allows agents to connect to and interact with MCP servers
"""
import asyncio
from typing import Dict, List, Optional, Any, Callable
import logging
import uuid
from datetime import datetime

from .protocol import (
    JSONRPCRequest, JSONRPCResponse, JSONRPCError,
    MCPResource, MCPTool, MCPPrompt
)
from .transport import BaseTransport, TransportConfig, create_transport

logger = logging.getLogger(__name__)

class MCPClient:
    """MCP Client for AI Agents"""
    
    def __init__(
        self,
        name: str,
        transport_config: TransportConfig
    ):
        self.name = name
        self.transport_config = transport_config
        self.transport: Optional[BaseTransport] = None
        
        # Client state
        self.connected = False
        self.session_id: Optional[str] = None
        self.request_id_counter = 0
        
        # Pending requests
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Cached data
        self.resources_cache: Dict[str, MCPResource] = {}
        self.tools_cache: Dict[str, MCPTool] = {}
        self.prompts_cache: Dict[str, MCPPrompt] = {}
        
        # Subscriptions
        self.subscriptions: Dict[str, List[Callable]] = {}
        
        # Server capabilities
        self.server_capabilities: Optional[Dict[str, Any]] = None
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            logger.info(f"Client {self.name} connecting to MCP server...")
            
            # Create transport
            self.transport = create_transport(self.transport_config)
            self.transport.add_message_handler(self._handle_message)
            self.transport.add_error_handler(self._handle_error)
            
            # Connect transport
            if not await self.transport.connect():
                logger.error("Failed to connect transport")
                return False
            
            # Initialize session
            init_result = await self.initialize()
            if not init_result:
                logger.error("Failed to initialize MCP session")
                return False
            
            self.connected = True
            self.session_id = str(uuid.uuid4())
            
            # Cache resources, tools, and prompts
            await self._refresh_caches()
            
            logger.info(f"Client {self.name} connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting client: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server"""
        try:
            logger.info(f"Client {self.name} disconnecting...")
            
            self.connected = False
            
            # Unsubscribe from all resources
            for uri in list(self.subscriptions.keys()):
                await self.unsubscribe_resource(uri)
            
            # Disconnect transport
            if self.transport:
                await self.transport.disconnect()
            
            # Clear caches
            self.resources_cache.clear()
            self.tools_cache.clear()
            self.prompts_cache.clear()
            
            logger.info(f"Client {self.name} disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")
    
    async def initialize(self) -> bool:
        """Initialize MCP session"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="initialize",
            params={
                "protocolVersion": "2.0",
                "capabilities": {
                    "resources": {"subscribe": True},
                    "tools": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": self.name,
                    "version": "1.0.0"
                }
            },
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            self.server_capabilities = response.result.get("capabilities", {})
            logger.info(f"Initialized with server capabilities: {self.server_capabilities}")
            
            # Send initialized notification
            initialized = JSONRPCRequest(
                jsonrpc="2.0",
                method="initialized",
                params={}
            )
            await self._send_notification(initialized)
            
            return True
        
        return False
    
    async def list_resources(self) -> List[MCPResource]:
        """List available resources"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="resources/list",
            params={},
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            resources_data = response.result.get("resources", [])
            return [
                MCPResource(**res) 
                for res in resources_data
            ]
        
        return []
    
    async def read_resource(self, uri: str) -> Optional[Any]:
        """Read a resource by URI"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="resources/read",
            params={"uri": uri},
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            contents = response.result.get("contents", [])
            if contents:
                return contents[0].get("text")
        
        return None
    
    async def subscribe_resource(
        self, 
        uri: str, 
        callback: Callable[[str, Any], None]
    ) -> bool:
        """Subscribe to resource updates"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="resources/subscribe",
            params={
                "uri": uri,
                "sessionId": self.session_id
            },
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            if uri not in self.subscriptions:
                self.subscriptions[uri] = []
            self.subscriptions[uri].append(callback)
            logger.info(f"Subscribed to resource: {uri}")
            return True
        
        return False
    
    async def unsubscribe_resource(self, uri: str) -> bool:
        """Unsubscribe from resource updates"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="resources/unsubscribe",
            params={
                "uri": uri,
                "sessionId": self.session_id
            },
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            if uri in self.subscriptions:
                del self.subscriptions[uri]
            logger.info(f"Unsubscribed from resource: {uri}")
            return True
        
        return False
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="tools/list",
            params={},
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            tools_data = response.result.get("tools", [])
            return [
                MCPTool(
                    name=tool["name"],
                    description=tool["description"],
                    input_schema=tool["inputSchema"]
                )
                for tool in tools_data
            ]
        
        return []
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Optional[Any]:
        """Call a tool with given arguments"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            },
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            content = response.result.get("content", [])
            if content:
                return content[0].get("text")
        
        return None
    
    async def list_prompts(self) -> List[MCPPrompt]:
        """List available prompts"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="prompts/list",
            params={},
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            prompts_data = response.result.get("prompts", [])
            return [
                MCPPrompt(
                    name=prompt["name"],
                    description=prompt["description"],
                    arguments=prompt["arguments"],
                    template=prompt["template"]
                )
                for prompt in prompts_data
            ]
        
        return []
    
    async def get_prompt(
        self, 
        prompt_name: str, 
        arguments: Dict[str, Any]
    ) -> Optional[str]:
        """Get a prompt with filled arguments"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="prompts/get",
            params={
                "name": prompt_name,
                "arguments": arguments
            },
            id=self._get_next_request_id()
        )
        
        response = await self._send_request(request)
        
        if response and not response.error:
            messages = response.result.get("messages", [])
            if messages:
                return messages[0]["content"]["text"]
        
        return None
    
    # Internal methods
    def _get_next_request_id(self) -> str:
        """Generate next request ID"""
        self.request_id_counter += 1
        return f"{self.name}-{self.request_id_counter}"
    
    async def _send_request(self, request: JSONRPCRequest) -> Optional[JSONRPCResponse]:
        """Send a request and wait for response"""
        if not self.connected or not self.transport:
            logger.error("Client not connected")
            return None
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request.id] = future
        
        try:
            # Send request
            await self.transport.send(request.to_json())
            
            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Request {request.id} timed out")
            return None
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return None
        finally:
            # Clean up pending request
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
    
    async def _send_notification(self, request: JSONRPCRequest) -> None:
        """Send a notification (no response expected)"""
        if not self.connected or not self.transport:
            logger.error("Client not connected")
            return
        
        try:
            await self.transport.send(request.to_json())
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming message from server"""
        try:
            response = JSONRPCResponse.from_json(message)
            
            # Check if this is a response to a pending request
            if response.id and response.id in self.pending_requests:
                future = self.pending_requests[response.id]
                if not future.done():
                    future.set_result(response)
            
            # Check if this is a notification (resource update, etc.)
            else:
                await self._handle_notification(response)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_notification(self, response: JSONRPCResponse) -> None:
        """Handle server notifications"""
        # Handle resource updates
        if response.result and "uri" in response.result:
            uri = response.result["uri"]
            if uri in self.subscriptions:
                data = response.result.get("data")
                for callback in self.subscriptions[uri]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(uri, data)
                        else:
                            callback(uri, data)
                    except Exception as e:
                        logger.error(f"Error in subscription callback: {e}")
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle transport errors"""
        logger.error(f"Transport error: {error}")
        self.connected = False
        
        # Attempt to reconnect
        await self._attempt_reconnect()
    
    async def _attempt_reconnect(self, max_attempts: int = 3) -> bool:
        """Attempt to reconnect to server"""
        for attempt in range(max_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
            
            await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff
            
            if await self.connect():
                logger.info("Reconnection successful")
                return True
        
        logger.error("Reconnection failed")
        return False
    
    async def _refresh_caches(self) -> None:
        """Refresh cached resources, tools, and prompts"""
        try:
            # Cache resources
            resources = await self.list_resources()
            for resource in resources:
                self.resources_cache[resource.uri] = resource
            
            # Cache tools
            tools = await self.list_tools()
            for tool in tools:
                self.tools_cache[tool.name] = tool
            
            # Cache prompts
            prompts = await self.list_prompts()
            for prompt in prompts:
                self.prompts_cache[prompt.name] = prompt
            
            logger.info(f"Cached {len(resources)} resources, {len(tools)} tools, {len(prompts)} prompts")
            
        except Exception as e:
            logger.error(f"Error refreshing caches: {e}")
    
    # Helper methods for agents
    async def get_campaigns(self) -> Optional[List[Dict[str, Any]]]:
        """Get all campaigns"""
        data = await self.read_resource("campaigns://all")
        if data:
            import json
            return json.loads(data)
        return None
    
    async def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific campaign"""
        data = await self.read_resource(f"campaigns://{campaign_id}")
        if data:
            import json
            return json.loads(data)
        return None
    
    async def get_leads(self) -> Optional[List[Dict[str, Any]]]:
        """Get all leads"""
        data = await self.read_resource("leads://all")
        if data:
            import json
            return json.loads(data)
        return None
    
    async def get_lead(self, lead_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific lead"""
        data = await self.read_resource(f"leads://{lead_id}")
        if data:
            import json
            return json.loads(data)
        return None
    
    async def get_leads_by_status(self, status: str) -> Optional[List[Dict[str, Any]]]:
        """Get leads by status"""
        data = await self.read_resource(f"leads://status/{status}")
        if data:
            import json
            return json.loads(data)
        return None
    
    async def get_lead_interactions(self, lead_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get interactions for a lead"""
        data = await self.read_resource(f"interactions://lead/{lead_id}")
        if data:
            import json
            return json.loads(data)
        return None
    
    async def score_lead(self, lead_id: str) -> Optional[Dict[str, Any]]:
        """Score a lead using the MCP tool"""
        result = await self.call_tool("score_lead", {"lead_id": lead_id})
        if result:
            import json
            return json.loads(result)
        return None
    
    async def send_message(
        self,
        lead_id: str,
        channel: str,
        message: str,
        template_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Send a message to a lead"""
        params = {
            "lead_id": lead_id,
            "channel": channel,
            "message": message
        }
        if template_id:
            params["template_id"] = template_id
        
        result = await self.call_tool("send_message", params)
        if result:
            import json
            return json.loads(result)
        return None
    
    async def optimize_campaign(
        self,
        campaign_id: str,
        metrics: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Optimize a campaign"""
        params = {"campaign_id": campaign_id}
        if metrics:
            params["metrics"] = metrics
        
        result = await self.call_tool("optimize_campaign", params)
        if result:
            import json
            return json.loads(result)
        return None
    
    async def qualify_lead(
        self,
        lead_id: str,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Qualify a lead"""
        params = {"lead_id": lead_id}
        if criteria:
            params["criteria"] = criteria
        
        result = await self.call_tool("qualify_lead", params)
        if result:
            import json
            return json.loads(result)
        return None
    
    async def generate_engagement_message(
        self,
        lead_name: str,
        lead_context: str,
        goal: Optional[str] = None
    ) -> Optional[str]:
        """Generate an engagement message using a prompt"""
        args = {
            "lead_name": lead_name,
            "lead_context": lead_context,
            "goal": goal or "Build relationship and move to next stage"
        }
        
        return await self.get_prompt("engage_lead", args)
    
    async def analyze_campaign_prompt(
        self,
        campaign_data: Dict[str, Any],
        time_period: str
    ) -> Optional[str]:
        """Get campaign analysis prompt"""
        import json
        args = {
            "campaign_data": json.dumps(campaign_data),
            "time_period": time_period
        }
        
        return await self.get_prompt("analyze_campaign", args)
    
    async def get_lead_qualification_prompt(
        self,
        lead_data: Dict[str, Any],
        qualification_criteria: Dict[str, Any]
    ) -> Optional[str]:
        """Get lead qualification prompt"""
        import json
        args = {
            "lead_data": json.dumps(lead_data),
            "qualification_criteria": json.dumps(qualification_criteria)
        }
        
        return await self.get_prompt("qualify_lead", args)

class MCPClientManager:
    """Manages multiple MCP client connections"""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
    
    def add_client(self, name: str, client: MCPClient) -> None:
        """Add a client"""
        self.clients[name] = client
    
    def remove_client(self, name: str) -> None:
        """Remove a client"""
        if name in self.clients:
            del self.clients[name]
    
    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get a client by name"""
        return self.clients.get(name)
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all clients"""
        results = {}
        for name, client in self.clients.items():
            results[name] = await client.connect()
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all clients"""
        for client in self.clients.values():
            await client.disconnect()
    
    def get_all_clients(self) -> List[MCPClient]:
        """Get all clients"""
        return list(self.clients.values())

# Factory function
def create_mcp_client(
    name: str,
    url: str,
    transport_type: str = "websocket"
) -> MCPClient:
    """Create an MCP client with given configuration"""
    from .transport import TransportType, TransportConfig
    
    config = TransportConfig(
        transport_type=TransportType(transport_type),
        url=url,
        timeout=30,
        max_retries=3,
        heartbeat_interval=30
    )
    
    return MCPClient(name=name, transport_config=config)