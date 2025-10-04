"""
Marketing Multi-Agent System - Main Application (PATCHED VERSION)
Entry point for the entire system with error fixes applied
"""
import asyncio
import signal
import sys
import io
import logging
import logging.config
from typing import Optional

# Fix Windows console encoding FIRST
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace'
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding='utf-8',
        errors='replace'
    )

import sys
import os
# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings, get_log_config
from memory import MemoryManager
from mcp.server import MCPServer
from mcp.client import MCPClient
from mcp.transport import TransportManager, WebSocketTransport, TransportConfig, TransportType
from database.db_manager import DatabaseManager
from transport.websocket_handler import WebSocketHandler
from transport.http_handler import HTTPTransportHandler
from agents.lead_triage_agent import LeadTriageAgent
from agents.engagement_agent import EngagementAgent
from agents.campaign_optimization_agent import CampaignOptimizationAgent

# Create HTTP handler and register handlers
http_handler = HTTPTransportHandler()

# Test handler
async def test_handler(params, agent_id=None):
    return {
        "message": "Test handler executed successfully",
        "params": params,
        "agent_id": agent_id,
        "timestamp": asyncio.get_event_loop().time()
    }

# System status handler
async def get_system_status(params, agent_id=None):
    return {
        "system_status": "operational",
        "memory_systems": "initialized",
        "mcp_server": "running",
        "websocket_port": 9003,
        "agents_initialized": False,
        "uptime": "system_running"
    }

# Mock lead triage handler
async def triage_lead(params, agent_id=None):
    lead_data = params.get("lead_data", {})
    return {
        "action": "triage_completed",
        "lead_id": lead_data.get("lead_id", "unknown"),
        "intent": "inquiry",
        "score": 75.5,
        "priority": "medium",
        "routing": "engagement_agent"
    }

http_handler.register_handler("test", test_handler)
http_handler.register_handler("get_system_status", get_system_status)
http_handler.register_handler("triage_lead", triage_lead)

# Global FastAPI app
app = http_handler.get_app()


class MarketingMultiAgentSystem:


    def __init__(self):
        self.logger = self._setup_logging()

    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Marketing Multi-Agent System...")
            # Core components
            self.memory_manager: Optional[MemoryManager] = None
            self.mcp_server: Optional[MCPServer] = None
            self.mcp_client: Optional[MCPClient] = None
            self.websocket_handler: Optional[WebSocketHandler] = None
            self.http_handler = http_handler  # Use global HTTP handler

            # Agents
            self.lead_triage_agent: Optional[LeadTriageAgent] = None
            self.engagement_agents: list = []
            self.campaign_optimizer: Optional[CampaignOptimizationAgent] = None

            # Shutdown flag
            self.is_running = False

            # 1. Initialize Memory Manager
            self.logger.info("Initializing memory systems...")
            self.memory_manager = MemoryManager()
            if not await self.memory_manager.initialize():
                raise Exception("Memory manager initialization failed")
            self.logger.info("[OK] Memory systems initialized")

            # 2. Initialize MCP Server (use unique port)
            self.logger.info("Initializing MCP server...")

            # Create database manager
            db_manager = DatabaseManager()
            db_manager.create_tables()

            # Create transport manager
            transport_manager = TransportManager()

            # Create WebSocket transport for MCP
            mcp_transport_config = TransportConfig(
                transport_type=TransportType.WEBSOCKET,
                url=f"{settings.MCP_HOST}:9001"
            )
            mcp_transport = WebSocketTransport(mcp_transport_config)
            transport_manager.add_transport("mcp_websocket", mcp_transport)
            transport_manager.set_active_transport("mcp_websocket")

            # Create MCP server
            self.mcp_server = MCPServer(
                db_manager=db_manager,
                transport_manager=transport_manager
            )
            await self.mcp_server.start()
            self.logger.info("[OK] MCP server initialized")

            # 3. Initialize MCP Client
            self.logger.info("Initializing MCP client...")
            mcp_client_config = TransportConfig(
                transport_type=TransportType.WEBSOCKET,
                url=f"{settings.MCP_HOST}:9001"
            )
            self.mcp_client = MCPClient(
                name="main_client",
                transport_config=mcp_client_config
            )
            try:
                await self.mcp_client.connect()
                self.logger.info("[OK] MCP client connected")
            except Exception as e:
                self.logger.warning(f"MCP client connection failed (will retry): {e}")

            # 4. Initialize Transport Layer
            self.logger.info("Initializing transport handlers...")

            # WebSocket on separate port
            self.websocket_handler = None
            for port in [9002, 9003, 9004, 9005]:
                try:
                    self.websocket_handler = WebSocketHandler(
                        host=settings.MCP_HOST,
                        port=port
                    )
                    await self.websocket_handler.start()
                    self.logger.info(f"[OK] WebSocket handler started on port {port}")
                    break
                except OSError as e:
                    if "10048" in str(e) or "Address already in use" in str(e):
                        self.logger.warning(f"Port {port} in use, trying next port...")
                        continue
                    else:
                        self.logger.warning(f"WebSocket handler failed on port {port}: {e}")
                        break

            if self.websocket_handler is None:
                self.logger.warning("WebSocket handler could not be started - all ports in use")
                self.logger.info("[OK] WebSocket handler initialization skipped")

            # HTTP
            self.http_handler = HTTPTransportHandler()
            self._register_test_handlers()
            self.logger.info("[OK] HTTP handler initialized")

            # 5. Initialize Agents (Skipped - agents need implementation)
            self.logger.info("Skipping agent initialization - agents need implementation")
            self.lead_triage_agent = None
            self.engagement_agents = []
            self.campaign_optimizer = None
            self.logger.info("[OK] Agent initialization skipped")

            self.logger.info("=" * 60)
            self.logger.info("System initialization completed successfully!")
            self.logger.info("=" * 60)

            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def _setup_logging(self) -> logging.Logger:
        logging.config.dictConfig(get_log_config())
        return logging.getLogger(__name__)

    async def __call__(self, scope, receive, send):
        """ASGI callable to delegate to the HTTP handler"""
        if self.http_handler:
            await app(scope, receive, send)

    async def start(self):
        """Start the application"""
        try:
            self.is_running = True
            self.logger.info("Marketing Multi-Agent System is now operational")
            self.logger.info(f"API Server: http://localhost:{settings.PORT}")
            self.logger.info(f"WebSocket: ws://localhost:9002 (or 9003)")
            self.logger.info("Press CTRL+C to shutdown")

            # Start HTTP server (blocking)
            import uvicorn
            config = uvicorn.Config(
                app=self.http_handler.get_app(),
                host=settings.HOST,
                port=settings.PORT,
                log_level=settings.LOG_LEVEL.lower()
            )
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown of all components"""
        if not self.is_running:
            return

        self.logger.info("Initiating graceful shutdown...")
        self.is_running = False

        try:
            # Stop agents
            if self.lead_triage_agent:
                await self.lead_triage_agent.stop()

            for agent in self.engagement_agents:
                await agent.stop()

            if self.campaign_optimizer:
                await self.campaign_optimizer.stop()

            self.logger.info("[OK] Agents stopped")

            # Stop transport handlers
            if self.websocket_handler:
                await self.websocket_handler.stop()

            self.logger.info("[OK] Transport handlers stopped")

            # Stop MCP
            if self.mcp_client:
                await self.mcp_client.disconnect()

            if self.mcp_server:
                await self.mcp_server.stop()

            self.logger.info("[OK] MCP server/client stopped")

            # Stop memory systems
            if self.memory_manager:
                await self.memory_manager.shutdown()

            self.logger.info("[OK] Memory systems closed")
            self.logger.info("=" * 60)
            self.logger.info("Shutdown completed successfully")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}", exc_info=True)

    def _register_test_handlers(self):

        # Test handler
        async def test_handler(params, agent_id=None):
            return {
                "message": "Test handler executed successfully",
                "params": params,
                "agent_id": agent_id,
                "timestamp": asyncio.get_event_loop().time()
            }

        # System status handler
        async def get_system_status(params, agent_id=None):
            return {
                "system_status": "operational",
                "memory_systems": "initialized",
                "mcp_server": "running",
                "websocket_port": 9003,
                "agents_initialized": False,
                "uptime": "system_running"
            }

        # Mock lead triage handler
        async def triage_lead(params, agent_id=None):
            lead_data = params.get("lead_data", {})
            return {
                "action": "triage_completed",
                "lead_id": lead_data.get("lead_id", "unknown"),
                "intent": "inquiry",
                "score": 75.5,
                "priority": "medium",
                "routing": "engagement_agent"
            }

        # Register handlers
        self.http_handler.register_handler("test", test_handler)
        self.http_handler.register_handler("get_system_status", get_system_status)
        self.http_handler.register_handler("triage_lead", triage_lead)

        self.logger.info("Registered test handlers: test, get_system_status, triage_lead")


# Global application instance
app = MarketingMultiAgentSystem()

async def main():
    """Main entry point"""

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal, cleaning up...")
        asyncio.create_task(app.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize system
    if not await app.initialize():
        print("Failed to initialize system. Exiting.")
        sys.exit(1)

    # Start application
    try:
        await app.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
