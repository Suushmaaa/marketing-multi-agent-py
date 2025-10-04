"""
Configuration settings for the Marketing Multi-Agent System
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/marketing.db")

# Debug mode
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Agent configurations
agent_config = {
    "EPISODIC_MEMORY_MAX_EPISODES": 10000,
    "SHORT_TERM_MEMORY_TTL_HOURS": 24,
    "LONG_TERM_MEMORY_UPDATE_INTERVAL": 7,  # days
    "LEAD_SCORING_WEIGHTS": {
        "engagement": 0.4,
        "demographics": 0.3,
        "behavior": 0.3
    }
}

# MCP settings
mcp_config = {
    "DEFAULT_PORT": 8080,
    "WEBSOCKET_ENABLED": True,
    "HTTP_ENABLED": True,
    "MAX_CONNECTIONS": 100,
    "TIMEOUT": 30
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_log_config():
    """Get logging configuration dictionary"""
    import logging
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': LOG_FORMAT
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': LOG_LEVEL,
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': BASE_DIR / 'logs' / 'marketing_system.log',
                'formatter': 'standard',
                'level': LOG_LEVEL,
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': LOG_LEVEL,
                'propagate': True
            }
        }
    }
    
# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Host and Port settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# MCP settings
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "9001"))

# Separate ports for services
WS_PORT = int(os.getenv("WS_PORT", "9002"))

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Workers
WORKERS = int(os.getenv("WORKERS", "4"))

# Agent IDs
LEAD_TRIAGE_AGENT_ID = os.getenv("LEAD_TRIAGE_AGENT_ID", "lead_triage_001")
ENGAGEMENT_AGENT_PREFIX = os.getenv("ENGAGEMENT_AGENT_PREFIX", "engagement_")
CAMPAIGN_OPT_AGENT_ID = os.getenv("CAMPAIGN_OPT_AGENT_ID", "campaign_opt_001")

# Memory settings
SHORT_TERM_MEMORY_TTL = int(os.getenv("SHORT_TERM_MEMORY_TTL", "3600"))  # 1 hour
MEMORY_CONSOLIDATION_INTERVAL = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "300"))  # 5 minutes

# Neo4j settings (for semantic memory)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Redis settings (for short-term memory)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
JWT_EXPIRATION_HOURS = 24

# External services
EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL")
SMS_SERVICE_URL = os.getenv("SMS_SERVICE_URL")

# Feature flags
FEATURES = {
    "episodic_memory": True,
    "real_time_analytics": True,
    "multi_channel_support": True,
    "ai_driven_insights": True
}

# Create data directory if it doesn't exist
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Settings object for easy access
class Settings:
    def __init__(self):
        self.BASE_DIR = BASE_DIR
        self.DATABASE_URL = DATABASE_URL
        self.DEBUG = DEBUG
        self.agent_config = agent_config
        self.mcp_config = mcp_config
        self.LOG_LEVEL = LOG_LEVEL
        self.LOG_FORMAT = LOG_FORMAT
        self.API_HOST = API_HOST
        self.API_PORT = API_PORT
        self.HOST = HOST
        self.PORT = PORT
        self.MCP_HOST = MCP_HOST
        self.MCP_PORT = MCP_PORT
        self.WS_PORT = WS_PORT
        self.ENVIRONMENT = ENVIRONMENT
        self.WORKERS = WORKERS
        self.LEAD_TRIAGE_AGENT_ID = LEAD_TRIAGE_AGENT_ID
        self.ENGAGEMENT_AGENT_PREFIX = ENGAGEMENT_AGENT_PREFIX
        self.CAMPAIGN_OPT_AGENT_ID = CAMPAIGN_OPT_AGENT_ID
        self.SHORT_TERM_MEMORY_TTL = SHORT_TERM_MEMORY_TTL
        self.MEMORY_CONSOLIDATION_INTERVAL = MEMORY_CONSOLIDATION_INTERVAL
        self.NEO4J_URI = NEO4J_URI
        self.NEO4J_USER = NEO4J_USER
        self.NEO4J_PASSWORD = NEO4J_PASSWORD
        self.REDIS_URL = REDIS_URL
        self.SECRET_KEY = SECRET_KEY
        self.JWT_EXPIRATION_HOURS = JWT_EXPIRATION_HOURS
        self.EMAIL_SERVICE_URL = EMAIL_SERVICE_URL
        self.SMS_SERVICE_URL = SMS_SERVICE_URL
        self.FEATURES = FEATURES
        self.DATA_DIR = DATA_DIR
        self.WS_MAX_MESSAGE_SIZE = int(os.getenv("WS_MAX_MESSAGE_SIZE", "1048576"))  # 1MB default
        self.WS_HEARTBEAT_INTERVAL = int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))  # 30 seconds default
        self.WS_TIMEOUT = int(os.getenv("WS_TIMEOUT", "60"))  # 60 seconds default
        self.ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:8000").split(",")
settings = Settings()
