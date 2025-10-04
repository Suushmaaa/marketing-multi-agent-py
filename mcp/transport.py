"""
Transport layer for MCP communication
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """Transport types supported by MCP"""
    WEBSOCKET = "websocket"
    HTTP = "http"


class ConnectionState(Enum):
    """Connection states for transports"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class TransportMessage:
    """Message structure for transport communication"""
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None


@dataclass
class TransportConfig:
    """Configuration for transport connections"""
    transport_type: TransportType
    url: str
    timeout: int = 30
    max_retries: int = 3
    heartbeat_interval: int = 30
    headers: Optional[Dict[str, str]] = None


class BaseTransport(ABC):
    """Abstract base class for MCP transports"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.message_handlers: List[Callable[[str], None]] = []
        self.error_handlers: List[Callable[[Exception], None]] = []

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the transport"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the transport"""
        pass

    @abstractmethod
    async def send(self, message: str) -> None:
        """Send a message"""
        pass

    @abstractmethod
    async def receive(self) -> Optional[str]:
        """Receive a message"""
        pass

    def add_message_handler(self, handler: Callable[[str], None]) -> None:
        """Add a message handler"""
        self.message_handlers.append(handler)

    def add_error_handler(self, handler: Callable[[Exception], None]) -> None:
        """Add an error handler"""
        self.error_handlers.append(handler)

    def _notify_message(self, message: str) -> None:
        """Notify message handlers"""
        for handler in self.message_handlers:
            try:
                asyncio.create_task(handler(message))
            except Exception as e:
                logger.error(f"Error in message handler: {e}")

    def _notify_error(self, error: Exception) -> None:
        """Notify error handlers"""
        for handler in self.error_handlers:
            try:
                handler(error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")


class TransportManager:
    """Manages multiple transports"""

    def __init__(self):
        self.transports: Dict[str, BaseTransport] = {}
        self.active_transport: Optional[BaseTransport] = None

    def add_transport(self, name: str, transport: BaseTransport) -> None:
        """Add a transport"""
        self.transports[name] = transport

    def get_transport(self, name: Optional[str] = None) -> Optional[BaseTransport]:
        """Get a transport by name or the active one"""
        if name:
            return self.transports.get(name)
        return self.active_transport

    def set_active_transport(self, name: str) -> None:
        """Set the active transport"""
        if name in self.transports:
            self.active_transport = self.transports[name]

    async def connect_all(self) -> None:
        """Connect all transports"""
        for transport in self.transports.values():
            try:
                await transport.connect()
            except Exception as e:
                logger.error(f"Failed to connect transport: {e}")

    async def disconnect_all(self) -> None:
        """Disconnect all transports"""
        for transport in self.transports.values():
            try:
                await transport.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect transport: {e}")

    def get_available_transports(self) -> List[str]:
        """Get list of available transport names"""
        return list(self.transports.keys())


class WebSocketTransport(BaseTransport):
    """WebSocket transport adapter for MCP"""

    def __init__(self, config: TransportConfig):
        super().__init__(config.__dict__)
        self.websocket = None
        self.receive_task = None

    async def connect(self) -> bool:
        """Connect to WebSocket server as client"""
        try:
            import websockets
            uri = f"ws://{self.config['url']}"
            logger.info(f"Connecting to WebSocket server at {uri}")

            self.websocket = await websockets.connect(uri)
            self.connected = True

            # Start receiving messages
            self.receive_task = asyncio.create_task(self._receive_loop())

            logger.info("WebSocket client connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect WebSocket client: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server"""
        try:
            self.connected = False

            if self.receive_task:
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass

            if self.websocket:
                await self.websocket.close()

            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket client: {e}")

    async def send(self, message: str) -> None:
        """Send a message through WebSocket"""
        if not self.connected or not self.websocket:
            raise Exception("WebSocket not connected")

        try:
            await self.websocket.send(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            raise

    async def receive(self) -> Optional[str]:
        """Receive is handled by the receive loop"""
        return None

    async def _receive_loop(self) -> None:
        """Loop to receive messages"""
        try:
            while self.connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    self._notify_message(message)
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error receiving WebSocket message: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            self.connected = False
            self._notify_error(Exception("WebSocket connection lost"))


class HTTPTransport(BaseTransport):
    """HTTP transport adapter for MCP"""

    def __init__(self, config: TransportConfig):
        super().__init__(config.__dict__)

    async def connect(self) -> bool:
        """Connect (no-op for HTTP)"""
        self.connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect (no-op for HTTP)"""
        self.connected = False

    async def send(self, message: str) -> None:
        """Send HTTP request"""
        # Implement HTTP request sending if needed
        pass

    async def receive(self) -> Optional[str]:
        """Receive HTTP response"""
        # Implement HTTP response receiving if needed
        return None


def create_transport(config: TransportConfig) -> BaseTransport:
    """Factory function to create transport based on config"""
    if config.transport_type == TransportType.WEBSOCKET:
        return WebSocketTransport(config)
    elif config.transport_type == TransportType.HTTP:
        return HTTPTransport(config)
    else:
        raise ValueError(f"Unsupported transport type: {config.transport_type}")
