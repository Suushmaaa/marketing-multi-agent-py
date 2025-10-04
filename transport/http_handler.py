"""
HTTP Transport Handler
Implements HTTP-based communication for request-response agent interactions
"""
from typing import Dict, Any, Optional, Callable
import asyncio
import logging
from datetime import datetime
import uuid

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time

from config.settings import settings


# Request/Response Models
class AgentRequest(BaseModel):
    """Agent request model"""
    method: str
    params: Dict[str, Any]
    agent_id: Optional[str] = None
    correlation_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent response model"""
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    request_id: str
    timestamp: str
    duration_ms: Optional[float] = None


class HTTPTransportHandler:
    """
    HTTP handler for agent communication.
    Implements REST API endpoints for synchronous agent interactions.
    """
    
    def __init__(self, app: Optional[FastAPI] = None):
        """
        Initialize HTTP handler
        
        Args:
            app: FastAPI application instance
        """
        self.app = app or FastAPI(
            title="Marketing Multi-Agent System API",
            version="1.0.0",
            description="HTTP API for agent communication"
        )
        
        self.logger = logging.getLogger("transport.http")
        
        # Request handlers
        self.request_handlers: Dict[str, Callable] = {}
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_bytes_in": 0,
            "total_bytes_out": 0,
            "average_response_time_ms": 0
        }
        
        # Response times for averaging
        self._response_times = []
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            request.state.start_time = time.time()
            
            # Log request
            self.logger.info(
                f"Request {request_id}: {request.method} {request.url.path}"
            )
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - request.state.start_time) * 1000
            
            # Update stats
            self.stats["total_requests"] += 1
            self._response_times.append(duration_ms)
            
            # Keep only last 1000 response times
            if len(self._response_times) > 1000:
                self._response_times = self._response_times[-1000:]
            
            self.stats["average_response_time_ms"] = (
                sum(self._response_times) / len(self._response_times)
            )
            
            # Log response
            self.logger.info(
                f"Response {request_id}: {response.status_code} "
                f"({duration_ms:.2f}ms)"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "Marketing Multi-Agent System",
                "version": "1.0.0",
                "status": "operational"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "stats": self.get_stats()
            }
        
        @self.app.post("/agent/invoke")
        async def invoke_agent(request: AgentRequest, req: Request):
            """Invoke agent method"""
            return await self._handle_agent_request(request, req)
        
        @self.app.post("/agent/message")
        async def send_agent_message(payload: Dict[str, Any], req: Request):
            """Send message to agent"""
            return await self._handle_agent_message(payload, req)
        
        @self.app.get("/agent/{agent_id}/status")
        async def get_agent_status(agent_id: str):
            """Get agent status"""
            return await self._handle_agent_status(agent_id)
        
        @self.app.get("/stats")
        async def get_statistics():
            """Get transport statistics"""
            return self.get_stats()
    
    async def _handle_agent_request(
        self,
        request: AgentRequest,
        req: Request
    ) -> AgentResponse:
        """Handle agent method invocation"""
        start_time = time.time()
        request_id = req.state.request_id
        
        try:
            # Check if handler registered
            if request.method not in self.request_handlers:
                self.stats["failed_requests"] += 1
                raise HTTPException(
                    status_code=404,
                    detail=f"Method not found: {request.method}"
                )
            
            # Execute handler
            handler = self.request_handlers[request.method]
            result = await handler(request.params, request.agent_id)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self.stats["successful_requests"] += 1
            
            return AgentResponse(
                status="success",
                data=result,
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                duration_ms=duration_ms
            )
            
        except HTTPException:
            raise
        
        except Exception as e:
            self.logger.error(f"Request error: {e}", exc_info=True)
            self.stats["failed_requests"] += 1
            
            duration_ms = (time.time() - start_time) * 1000
            
            return AgentResponse(
                status="error",
                error=str(e),
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                duration_ms=duration_ms
            )
    
    async def _handle_agent_message(
        self,
        payload: Dict[str, Any],
        req: Request
    ) -> Dict[str, Any]:
        """Handle agent message"""
        request_id = req.state.request_id
        
        try:
            # Validate message
            required_fields = ["from_agent", "to_agent", "message_type", "data"]
            if not all(field in payload for field in required_fields):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid message format"
                )
            
            # Log message
            self.logger.info(
                f"Agent message: {payload['from_agent']} -> {payload['to_agent']} "
                f"(type: {payload['message_type']})"
            )
            
            # In production, route to target agent
            # For now, return acknowledgment
            self.stats["successful_requests"] += 1
            
            return {
                "status": "delivered",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        
        except Exception as e:
            self.logger.error(f"Message handling error: {e}", exc_info=True)
            self.stats["failed_requests"] += 1
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        try:
            # In production, query actual agent status
            # For now, return mock status
            return {
                "agent_id": agent_id,
                "status": "active",
                "last_heartbeat": datetime.now().isoformat(),
                "active_conversations": 0
            }
            
        except Exception as e:
            self.logger.error(f"Status query error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    def register_handler(self, method: str, handler: Callable):
        """
        Register a request handler
        
        Args:
            method: Method name
            handler: Async function to handle the request
        """
        self.request_handlers[method] = handler
        self.logger.info(f"Registered handler for method: {method}")
    
    def get_app(self) -> FastAPI:
        """Get FastAPI application instance"""
        return self.app
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        success_rate = 0
        if self.stats["total_requests"] > 0:
            success_rate = (
                self.stats["successful_requests"] / 
                self.stats["total_requests"] * 100
            )
        
        return {
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate_percent": round(success_rate, 2),
            "average_response_time_ms": round(
                self.stats["average_response_time_ms"], 2
            ),
            "registered_handlers": len(self.request_handlers)
        }


class HTTPClient:
    """
    HTTP client for making requests to other agents/services
    """
    
    def __init__(self, base_url: str):
        """
        Initialize HTTP client
        
        Args:
            base_url: Base URL for requests
        """
        self.base_url = base_url
        self.logger = logging.getLogger("transport.http.client")
        
        import httpx
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(settings.JSONRPC_TIMEOUT)
        )
    
    async def invoke_agent(
        self,
        method: str,
        params: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke agent method via HTTP
        
        Args:
            method: Method name
            params: Method parameters
            agent_id: Target agent ID
            
        Returns:
            Response data
        """
        try:
            request = AgentRequest(
                method=method,
                params=params,
                agent_id=agent_id,
                correlation_id=str(uuid.uuid4())
            )
            
            response = await self.client.post(
                "/agent/invoke",
                json=request.dict()
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("status") == "error":
                raise Exception(result.get("error", "Unknown error"))
            
            return result.get("data", {})
            
        except Exception as e:
            self.logger.error(f"Agent invocation failed: {e}", exc_info=True)
            raise
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Send message to agent
        
        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID
            message_type: Message type
            data: Message data
            
        Returns:
            bool: True if delivered
        """
        try:
            payload = {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message_type": message_type,
                "data": data
            }
            
            response = await self.client.post(
                "/agent/message",
                json=payload
            )
            
            response.raise_for_status()
            
            result = response.json()
            return result.get("status") == "delivered"
            
        except Exception as e:
            self.logger.error(f"Message send failed: {e}", exc_info=True)
            return False
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        try:
            response = await self.client.get(f"/agent/{agent_id}/status")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Status query failed: {e}", exc_info=True)
            return {}
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()