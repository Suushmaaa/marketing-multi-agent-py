"""
MCP Protocol Implementation
Handles JSON-RPC 2.0 communication for MCP
"""
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging
MCPVersion = "1.0"


logger = logging.getLogger(__name__)


class JSONRPCErrorCode(Enum):
    """JSON-RPC error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class JSONRPCError:
    """JSON-RPC error object"""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "code": self.code,
            "message": self.message
        }
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class JSONRPCRequest:
    """JSON-RPC request object"""
    id: Optional[str]
    method: str
    params: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_json(cls, json_str: str) -> 'JSONRPCRequest':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(
            id=data.get("id"),
            method=data["method"],
            params=data.get("params"),
            jsonrpc=data.get("jsonrpc", "2.0")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method
        }
        if self.id is not None:
            result["id"] = self.id
        if self.params is not None:
            result["params"] = self.params
        return result


@dataclass
class JSONRPCResponse:
    """JSON-RPC response object"""
    id: Optional[str]
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            result["id"] = self.id
        if self.result is not None:
            result["result"] = self.result
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


@dataclass
class MCPResource:
    """MCP Resource"""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "uri": self.uri,
            "name": self.name
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result


@dataclass
class MCPTool:
    """MCP Tool"""
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.input_schema:
            result["inputSchema"] = self.input_schema
        return result


@dataclass
class MCPPromptArgument:
    """MCP Prompt argument"""
    name: str
    type: str
    required: bool = False
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "name": self.name,
            "type": self.type,
            "required": self.required
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class MCPPrompt:
    """MCP Prompt"""
    name: str
    description: Optional[str] = None
    arguments: Optional[List[MCPPromptArgument]] = None
    template: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.arguments:
            result["arguments"] = [arg.to_dict() for arg in self.arguments]
        return result


class MCPProtocolHandler:
    """Handles MCP protocol operations"""

    def __init__(self):
        self.supported_methods = {
            # Resource methods
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "resources/subscribe": self._handle_resources_subscribe,
            "resources/unsubscribe": self._handle_resources_unsubscribe,

            # Tool methods
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,

            # Prompt methods
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
        }

        # Override handlers (set by server)
        self._handle_resources_list = None
        self._handle_resources_read = None
        self._handle_resources_subscribe = None
        self._handle_resources_unsubscribe = None
        self._handle_tools_list = None
        self._handle_tools_call = None
        self._handle_prompts_list = None
        self._handle_prompts_get = None

    def validate_request(self, request: JSONRPCRequest) -> Optional[JSONRPCError]:
        """Validate JSON-RPC request"""
        if request.jsonrpc != "2.0":
            return JSONRPCError(
                JSONRPCErrorCode.INVALID_REQUEST,
                "Invalid JSON-RPC version"
            )

        if not request.method:
            return JSONRPCError(
                JSONRPCErrorCode.INVALID_REQUEST,
                "Method is required"
            )

        return None

    def create_response(
        self,
        request_id: Optional[str],
        result: Any = None,
        error: JSONRPCError = None
    ) -> JSONRPCResponse:
        """Create JSON-RPC response"""
        return JSONRPCResponse(
            id=request_id,
            result=result,
            error=error
        )

    def create_error_response(
        self,
        request_id: Optional[str],
        error_code: JSONRPCErrorCode,
        message: str,
        data: Any = None
    ) -> JSONRPCResponse:
        """Create JSON-RPC error response"""
        error = JSONRPCError(error_code.value, message, data)
        return self.create_response(request_id, error=error)

    # Default handlers (can be overridden)
    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default resources list handler"""
        if self._handle_resources_list:
            return await self._handle_resources_list(params)
        return {"resources": []}

    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default resources read handler"""
        if self._handle_resources_read:
            return await self._handle_resources_read(params)
        raise ValueError("Resource not found")

    async def _handle_resources_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default resources subscribe handler"""
        if self._handle_resources_subscribe:
            return await self._handle_resources_subscribe(params)
        return {"success": True}

    async def _handle_resources_unsubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default resources unsubscribe handler"""
        if self._handle_resources_unsubscribe:
            return await self._handle_resources_unsubscribe(params)
        return {"success": True}

    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default tools list handler"""
        if self._handle_tools_list:
            return await self._handle_tools_list(params)
        return {"tools": []}

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default tools call handler"""
        if self._handle_tools_call:
            return await self._handle_tools_call(params)
        raise ValueError("Tool not found")

    async def _handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default prompts list handler"""
        if self._handle_prompts_list:
            return await self._handle_prompts_list(params)
        return {"prompts": []}

    async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default prompts get handler"""
        if self._handle_prompts_get:
            return await self._handle_prompts_get(params)
        raise ValueError("Prompt not found")
