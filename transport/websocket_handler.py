"""
WebSocket Transport Handler
Implements WebSocket-based communication for real-time agent interactions
"""
from typing import Dict, Any, List, Optional, Set
import asyncio
import json
import logging
from datetime import datetime
import uuid

import websockets
from websockets.server import WebSocketServerProtocol

from config.settings import settings


class WebSocketSession:
    """Represents a WebSocket connection session"""
    
    def __init__(self, websocket: WebSocketServerProtocol, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.metadata: Dict[str, Any] = {}
    
    async def send(self, message: Dict[str, Any]):
        """Send message through WebSocket"""
        message_str = json.dumps(message)
        await self.websocket.send(message_str)
        self.messages_sent += 1
        self.bytes_out += len(message_str.encode('utf-8'))
        self.last_activity = datetime.now()
    
    async def receive(self) -> Dict[str, Any]:
        """Receive message from WebSocket"""
        message_str = await self.websocket.recv()
        self.messages_received += 1
        self.bytes_in += len(message_str.encode('utf-8'))
        self.last_activity = datetime.now()
        return json.loads(message_str)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        duration = (datetime.now() - self.connected_at).total_seconds()
        return {
            "session_id": self.session_id,
            "connected_at": self.connected_at.isoformat(),
            "duration_seconds": duration,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "last_activity": self.last_activity.isoformat()
        }


class WebSocketHandler:
    """
    WebSocket handler for agent communication.
    Manages connections, message routing, and heartbeats.
    """
    
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize WebSocket handler
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host or settings.MCP_HOST
        self.port = port or settings.MCP_PORT
        self.logger = logging.getLogger("transport.websocket")
        
        # Active sessions
        self.sessions: Dict[str, WebSocketSession] = {}
        
        # Message handlers
        self.message_handlers: Dict[str, callable] = {}
        
        # Server instance
        self.server = None
        
        # Stats
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_messages": 0,
            "errors": 0
        }
        
        # Heartbeat task
        self.heartbeat_task = None
        self.is_running = False
    
    async def start(self):
        """Start WebSocket server"""
        try:
            self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=settings.WS_MAX_MESSAGE_SIZE,
                ping_interval=settings.WS_HEARTBEAT_INTERVAL,
                ping_timeout=settings.WS_TIMEOUT
            )
            
            self.is_running = True
            
            # Start heartbeat monitor
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            self.logger.info("WebSocket server started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop WebSocket server"""
        try:
            self.logger.info("Stopping WebSocket server...")
            self.is_running = False
            
            # Cancel heartbeat task
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Close all sessions
            for session in list(self.sessions.values()):
                await session.websocket.close()
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            self.logger.info("WebSocket server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}", exc_info=True)
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        session_id = str(uuid.uuid4())
        session = WebSocketSession(websocket, session_id)
        
        self.sessions[session_id] = session
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.sessions)
        
        self.logger.info(f"New WebSocket connection: {session_id} from {websocket.remote_address}")
        
        try:
            # Send welcome message
            await session.send({
                "type": "connection_established",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Handle messages
            async for message_str in websocket:
                try:
                    message = json.loads(message_str)
                    session.messages_received += 1
                    session.bytes_in += len(message_str.encode('utf-8'))
                    session.last_activity = datetime.now()
                    
                    self.stats["total_messages"] += 1
                    
                    # Route message to appropriate handler
                    await self._route_message(session, message)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON from {session_id}: {e}")
                    await session.send({
                        "type": "error",
                        "error": "Invalid JSON format"
                    })
                    self.stats["errors"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}", exc_info=True)
                    await session.send({
                        "type": "error",
                        "error": str(e)
                    })
                    self.stats["errors"] += 1
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {session_id}")
        
        except Exception as e:
            self.logger.error(f"Connection error for {session_id}: {e}", exc_info=True)
        
        finally:
            # Cleanup
            if session_id in self.sessions:
                del self.sessions[session_id]
                self.stats["active_connections"] = len(self.sessions)
            
            self.logger.info(f"Session {session_id} disconnected. Stats: {session.get_stats()}")
    
    async def _route_message(self, session: WebSocketSession, message: Dict[str, Any]):
        """Route message to appropriate handler"""
        message_type = message.get("type")
        
        if not message_type:
            await session.send({
                "type": "error",
                "error": "Message type not specified"
            })
            return
        
        # Check for registered handler
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            response = await handler(session, message)
            
            if response:
                await session.send(response)
        
        # Built-in handlers
        elif message_type == "ping":
            await self._handle_ping(session, message)
        
        elif message_type == "agent_message":
            await self._handle_agent_message(session, message)
        
        elif message_type == "subscribe":
            await self._handle_subscribe(session, message)
        
        else:
            await session.send({
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            })
    
    async def _handle_ping(self, session: WebSocketSession, message: Dict[str, Any]):
        """Handle ping message"""
        await session.send({
            "type": "pong",
            "timestamp": datetime.now().isoformat(),
            "original_timestamp": message.get("timestamp")
        })
    
    async def _handle_agent_message(self, session: WebSocketSession, message: Dict[str, Any]):
        """Handle agent-to-agent message"""
        # Validate message structure
        required_fields = ["from_agent", "to_agent", "payload"]
        if not all(field in message for field in required_fields):
            await session.send({
                "type": "error",
                "error": "Invalid agent message format"
            })
            return
        
        # Log the message
        self.logger.info(
            f"Agent message: {message['from_agent']} -> {message['to_agent']}"
        )
        
        # In production, this would route to the target agent
        # For now, send acknowledgment
        await session.send({
            "type": "message_ack",
            "message_id": message.get("id"),
            "status": "delivered"
        })
    
    async def _handle_subscribe(self, session: WebSocketSession, message: Dict[str, Any]):
        """Handle subscription to agent updates"""
        agent_id = message.get("agent_id")
        
        if not agent_id:
            await session.send({
                "type": "error",
                "error": "agent_id required for subscription"
            })
            return
        
        # Store subscription in session metadata
        if "subscriptions" not in session.metadata:
            session.metadata["subscriptions"] = set()
        
        session.metadata["subscriptions"].add(agent_id)
        
        await session.send({
            "type": "subscription_confirmed",
            "agent_id": agent_id
        })
        
        self.logger.info(f"Session {session.session_id} subscribed to agent {agent_id}")
    
    async def _heartbeat_monitor(self):
        """Monitor and cleanup inactive sessions"""
        while self.is_running:
            try:
                await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
                
                now = datetime.now()
                inactive_sessions = []
                
                for session_id, session in self.sessions.items():
                    inactive_seconds = (now - session.last_activity).total_seconds()
                    
                    if inactive_seconds > settings.WS_TIMEOUT:
                        inactive_sessions.append(session_id)
                
                # Close inactive sessions
                for session_id in inactive_sessions:
                    session = self.sessions.get(session_id)
                    if session:
                        self.logger.warning(
                            f"Closing inactive session: {session_id} "
                            f"(inactive for {inactive_seconds}s)"
                        )
                        await session.websocket.close()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}", exc_info=True)
    
    def register_handler(self, message_type: str, handler: callable):
        """
        Register a message handler
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")
    
    async def broadcast(self, message: Dict[str, Any], agent_id: Optional[str] = None):
        """
        Broadcast message to all connected sessions or specific agent subscribers
        
        Args:
            message: Message to broadcast
            agent_id: Optional agent ID to filter subscribers
        """
        sent_count = 0
        
        for session in self.sessions.values():
            try:
                # If agent_id specified, only send to subscribers
                if agent_id:
                    subscriptions = session.metadata.get("subscriptions", set())
                    if agent_id not in subscriptions:
                        continue
                
                await session.send(message)
                sent_count += 1
                
            except Exception as e:
                self.logger.error(f"Broadcast error to {session.session_id}: {e}")
        
        return sent_count
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message to specific session
        
        Args:
            session_id: Target session ID
            message: Message to send
            
        Returns:
            bool: True if sent successfully
        """
        session = self.sessions.get(session_id)
        
        if not session:
            self.logger.warning(f"Session not found: {session_id}")
            return False
        
        try:
            await session.send(message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send to {session_id}: {e}")
            return False
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session"""
        session = self.sessions.get(session_id)
        return session.get_stats() if session else None
    
    def get_all_session_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all sessions"""
        return [session.get_stats() for session in self.sessions.values()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            "is_running": self.is_running,
            "host": self.host,
            "port": self.port,
            "stats": self.stats,
            "active_sessions": len(self.sessions)
        }