# Data models
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

Base = declarative_base()


class Campaign(Base):
    """Campaign model"""
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String)  # email, social, paid, etc.
    status = Column(String, default="active")  # active, paused, completed
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    budget = Column(Float)
    target_audience = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    leads = relationship("Lead", back_populates="campaign")


class Lead(Base):
    """Lead model"""
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(String, unique=True, index=True)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    status = Column(String, default="new")  # new, contacted, qualified, proposal, negotiation, closed
    score = Column(Float, default=0.0)
    source = Column(String)  # website, social, referral, etc.
    company = Column(String)
    job_title = Column(String)
    location = Column(String)
    interests = Column(Text)  # JSON string of interests
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    campaign = relationship("Campaign", back_populates="leads")
    interactions = relationship("Interaction", back_populates="lead")
    score = relationship("LeadScore", back_populates="lead", uselist=False)
    handoffs = relationship("AgentHandoff", back_populates="lead")
    messages = relationship("Message", back_populates="lead")


class Interaction(Base):
    """Interaction model"""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    interaction_id = Column(String, unique=True, index=True)
    lead_id = Column(String, ForeignKey("leads.lead_id"))
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=True)
    channel = Column(String)  # email, phone, chat, social
    direction = Column(String)  # inbound, outbound
    message_content = Column(Text)
    sentiment = Column(String)  # positive, neutral, negative
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration = Column(Integer)  # in seconds for calls
    outcome = Column(String)  # successful, unsuccessful, pending

    # Relationships
    lead = relationship("Lead", back_populates="interactions")
    agent = relationship("Agent", back_populates="interactions")


class Agent(Base):
    """Agent model"""
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String)  # lead_triage, engagement, campaign_optimization
    status = Column(String, default="active")  # active, inactive
    config = Column(Text)  # JSON string of agent configuration
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    interactions = relationship("Interaction", back_populates="agent")
    actions = relationship("AgentAction", back_populates="agent")


class AgentAction(Base):
    """Agent action model"""
    __tablename__ = "agent_actions"

    id = Column(Integer, primary_key=True, index=True)
    action_id = Column(String, unique=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"))
    lead_id = Column(String, ForeignKey("leads.lead_id"), nullable=True)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"), nullable=True)
    action_type = Column(String)  # score_lead, send_message, optimize_campaign, etc.
    action_data = Column(Text)  # JSON string of action data
    result = Column(Text)  # JSON string of action result
    success = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    agent = relationship("Agent", back_populates="actions")


class MCPResource(Base):
    """MCP Resource model"""
    __tablename__ = "mcp_resources"

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(String, unique=True, index=True)
    uri = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    mime_type = Column(String, default="application/json")
    data = Column(Text)  # JSON string of resource data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ShortTermMemory(Base):
    """Short-term memory model"""
    __tablename__ = "short_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    lead_id = Column(String, index=True)
    slots_json = Column(Text)  # JSON string of slot data
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LongTermMemory(Base):
    """Long-term memory model"""
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(String, unique=True, index=True)
    preferences_json = Column(Text)  # JSON string of preferences
    interaction_count = Column(Integer, default=0)
    last_interaction = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EpisodicMemory(Base):
    """Episodic memory model"""
    __tablename__ = "episodic_memory"

    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(String, unique=True, index=True)
    scenario = Column(String, index=True)
    action_sequence_json = Column(Text)  # JSON string of actions
    outcome_score = Column(Float)
    notes = Column(Text)
    lead_context = Column(Text)  # JSON string
    campaign_context = Column(Text)  # JSON string
    agent_type = Column(String)
    episode_metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class SemanticMemory(Base):
    """Semantic memory model (knowledge graph triples)"""
    __tablename__ = "semantic_memory"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    subject = Column(String, index=True)
    predicate = Column(String, index=True)
    object = Column(String, index=True)
    weight = Column(Float, default=1.0)
    source = Column(String)
    semantic_metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SemanticTriple:
    """Data structure for semantic triples (not a database model)"""
    def __init__(self, subject: str, predicate: str, object: str, weight: float = 1.0, source: str = "system"):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.weight = weight
        self.source = source


class AgentType(str, Enum):
    """Agent types"""
    LEAD_TRIAGE = "lead_triage"
    ENGAGEMENT = "engagement"
    CAMPAIGN_OPTIMIZATION = "campaign_optimization"
    MANAGER = "manager"


class ActionType(str, Enum):
    """Agent action types"""
    SCORE_LEAD = "score_lead"
    SEND_MESSAGE = "send_message"
    UPDATE_LEAD = "update_lead"
    HANDOFF = "handoff"
    ESCALATE = "escalate"
    QUERY_DATA = "query_data"


class EscalationReason(str, Enum):
    """Reasons for escalating to human manager"""
    NONE = "none"
    TECHNICAL_ISSUE = "technical_issue"
    COMPLEX_CASE = "complex_case"
    CUSTOMER_FRUSTRATION = "customer_frustration"
    POLICY_VIOLATION = "policy_violation"
    OUT_OF_SCOPE = "out_of_scope"


class AgentMessage(Base):
    """Agent message model"""
    __tablename__ = "agent_messages"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True)
    from_agent = Column(String, index=True)
    to_agent = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String)
    payload = Column(Text)  # JSON string
    correlation_id = Column(String, nullable=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class HandoffContext(Base):
    """Handoff context model"""
    __tablename__ = "handoff_contexts"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, unique=True, index=True)
    lead_id = Column(String, index=True)
    current_agent = Column(String)
    target_agent_type = Column(String)
    reason = Column(String)
    context_data = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class LeadScore(Base):
    """Lead scoring model"""
    __tablename__ = "lead_scores"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(String, ForeignKey("leads.lead_id"), unique=True, index=True)
    score = Column(Float, default=0.0)
    score_factors = Column(Text)  # JSON string of scoring factors
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    lead = relationship("Lead", back_populates="score")


class AgentHandoff(Base):
    """Agent handoff model"""
    __tablename__ = "agent_handoffs"

    id = Column(Integer, primary_key=True, index=True)
    handoff_id = Column(String, unique=True, index=True)
    conversation_id = Column(String, index=True)
    lead_id = Column(String, ForeignKey("leads.lead_id"), index=True)
    from_agent = Column(String, index=True)
    to_agent = Column(String, index=True)
    reason = Column(String)
    context_data = Column(Text)  # JSON string
    status = Column(String, default="pending")  # pending, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    lead = relationship("Lead", back_populates="handoffs")


class Message(Base):
    """Message model"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True)
    lead_id = Column(String, ForeignKey("leads.lead_id"), index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=True)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"), nullable=True)
    channel = Column(String)  # email, sms, chat, etc.
    direction = Column(String)  # inbound, outbound
    content = Column(Text)
    status = Column(String, default="sent")  # sent, delivered, read, failed
    sent_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    read_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    lead = relationship("Lead", back_populates="messages")
    agent = relationship("Agent", back_populates="messages")
    campaign = relationship("Campaign", back_populates="messages")


class ABTest(Base):
    """A/B test model"""
    __tablename__ = "ab_tests"

    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String, unique=True, index=True)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"), index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    test_type = Column(String)  # subject_line, content, timing, etc.
    variants = Column(Text)  # JSON string of test variants
    winner_variant = Column(String, nullable=True)
    status = Column(String, default="running")  # running, completed, stopped
    start_date = Column(DateTime)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    campaign = relationship("Campaign", back_populates="ab_tests")


class CampaignMetrics(Base):
    """Campaign metrics model"""
    __tablename__ = "campaign_metrics"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"), index=True)
    metric_date = Column(DateTime, index=True)
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    spend = Column(Float, default=0.0)
    revenue = Column(Float, default=0.0)
    ctr = Column(Float, default=0.0)  # Click-through rate
    cpc = Column(Float, default=0.0)  # Cost per click
    cpa = Column(Float, default=0.0)  # Cost per acquisition
    roas = Column(Float, default=0.0)  # Return on ad spend
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    campaign = relationship("Campaign", back_populates="metrics")


class Conversation(Base):
    """Conversation model"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, unique=True, index=True)
    lead_id = Column(String, index=True)
    opened_at = Column(DateTime)
    last_event_at = Column(DateTime)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class CampaignDaily(Base):
    """Campaign daily metrics model"""
    __tablename__ = "campaign_daily"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"), index=True)
    date = Column(DateTime, index=True)
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    spend = Column(Float, default=0.0)
    revenue = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Conversion(Base):
    """Conversion model"""
    __tablename__ = "conversions"

    id = Column(Integer, primary_key=True, index=True)
    conversion_id = Column(String, unique=True, index=True)
    lead_id = Column(String, ForeignKey("leads.lead_id"), index=True)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"), index=True)
    conversion_type = Column(String)  # purchase, signup, demo, etc.
    value = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
