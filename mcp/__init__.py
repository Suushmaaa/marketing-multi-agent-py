"""
Model Context Protocol (MCP) Implementation
Provides standardized communication between AI agents and data sources
"""

from .protocol import (
    JSONRPCErrorCode,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPResource,
    MCPTool,
    MCPPrompt,
    MCPProtocolHandler
)

from .transport import (
    TransportType,
    ConnectionState,
    TransportMessage,
    TransportConfig,
    BaseTransport,
    WebSocketTransport,
    HTTPTransport,
    TransportManager,
    create_transport
)

from .server import MCPServer

from .client import (
    MCPClient,
    MCPClientManager,
    create_mcp_client
)

__all__ = [
    # Protocol
    "JSONRPCErrorCode",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPResource",
    "MCPTool",
    "MCPPrompt",
    "MCPProtocolHandler",
    
    # Transport
    "TransportType",
    "ConnectionState",
    "TransportMessage",
    "TransportConfig",
    "BaseTransport",
    "WebSocketTransport",
    "HTTPTransport",
    "TransportManager",
    "create_transport",
    
    # Server
    "MCPServer",
    
    # Client
    "MCPClient",
    "MCPClientManager",
    "create_mcp_client"
]

__version__ = "1.0.0"
