"""
Engagement Agent - Personalized Conversations & Multi-Channel Communication
Handles lead engagement, personalization, and relationship building.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

from agents.base_agent import BaseAgent, AgentCapability
from database.models import Lead, Interaction, Campaign, Message
from config.settings import Settings


class EngagementStrategy(Enum):
    """Engagement strategy types"""
    IMMEDIATE_RESPONSE = "immediate_response"
    NURTURE_SEQUENCE = "nurture_sequence"
    REACTIVATION = "reactivation"
    RELATIONSHIP_BUILDING = "relationship_building"
    EDUCATION = "education"


class MessageTone(Enum):
    """Message tone variations"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"
    URGENT = "urgent"


class EngagementAgent(BaseAgent):
    """
    Specialized agent for lead engagement operations.
    
    Capabilities:
    - Personalized message generation
    - Multi-channel communication
    - Conversation management
    - Sentiment analysis
    - Engagement timing optimization
    - A/B testing message variants
    """
    
    def __init__(self, settings: Settings):
        super().__init__(
            agent_id="engagement_001",
            agent_type="engagement",
            settings=settings
        )
        
        # Personalization tokens
        self.personalization_tokens = {
            "first_name", "last_name", "company", "industry",
            "location", "job_title", "recent_interaction"
        }
        
        # Channel-specific configurations
        self.channel_config = {
            "email": {
                "max_length": 2000,
                "format": "html",
                "optimal_send_time": "09:00-11:00"
            },
            "sms": {
                "max_length": 160,
                "format": "plain",
                "optimal_send_time": "10:00-16:00"
            },
            "phone": {
                "max_duration": 300,  # seconds
                "format": "script",
                "optimal_call_time": "14:00-17:00"
            },
            "chat": {
                "max_length": 500,
                "format": "plain",
                "response_time": 30  # seconds
            }
        }
        
        # Engagement templates by strategy
        self.templates = self._initialize_templates()
        
        # Active conversations tracking
        self.active_conversations: Set[str] = set()
        
        self.logger.info(f"Engagement Agent initialized: {self.agent_id}")
    
    async def engage_lead(
        self,
        lead: Lead,
        strategy: EngagementStrategy,
        channel: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Execute engagement strategy for a lead.
        
        Args:
            lead: Lead to engage
            strategy: Engagement strategy to use
            channel: Communication channel
            context: Additional context for personalization
            
        Returns:
            Dict with engagement results
        """
        try:
            self.logger.info(
                f"Engaging lead {lead.lead_id} via {channel} "
                f"with strategy {strategy.value}"
            )
            
            # Step 1: Gather personalization data
            personalization_data = await self._gather_personalization_data(
                lead, context
            )
            
            # Step 2: Determine optimal tone
            tone = await self._determine_message_tone(lead, strategy)
            
            # Step 3: Generate personalized message
            message = await self.generate_message(
                lead=lead,
                strategy=strategy,
                channel=channel,
                tone=tone,
                personalization_data=personalization_data
            )
            
            # Step 4: Validate and optimize message
            validated_message = self._validate_message(message, channel)
            
            # Step 5: Schedule or send message
            send_result = await self._send_message(
                lead=lead,
                message=validated_message,
                channel=channel,
                strategy=strategy
            )
            
            # Step 6: Log interaction
            await self._log_engagement(
                lead=lead,
                message=validated_message,
                channel=channel,
                strategy=strategy,
                result=send_result
            )
            
            result = {
                "lead_id": lead.lead_id,
                "strategy": strategy.value,
                "channel": channel,
                "message": validated_message,
                "send_result": send_result,
                "status": "success"
            }
            
            self.logger.info(f"Engagement completed for lead {lead.lead_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error engaging lead {lead.lead_id}: {str(e)}")
            return {
                "lead_id": lead.lead_id,
                "status": "error",
                "error": str(e)
            }
    
    async def generate_message(
        self,
        lead: Lead,
        strategy: EngagementStrategy,
        channel: str,
        tone: MessageTone,
        personalization_data: Dict
    ) -> str:
        """
        Generate personalized message content.
        
        Args:
            lead: Lead to generate message for
            strategy: Engagement strategy
            channel: Communication channel
            tone: Message tone
            personalization_data: Data for personalization
            
        Returns:
            Generated message string
        """
        # Get base template
        template = self.templates.get(strategy, {}).get(tone, "")
        
        if not template:
            template = self._get_fallback_template(strategy)
        
        # Apply personalization
        message = template
        for token, value in personalization_data.items():
            placeholder = f"{{{token}}}"
            message = message.replace(placeholder, str(value))
        
        # Apply channel-specific formatting
        message = self._format_for_channel(message, channel)
        
        # Add dynamic elements
        message = await self._add_dynamic_elements(message, lead, strategy)
        
        return message
    
    async def handle_conversation(
        self,
        lead_id: str,
        incoming_message: str,
        channel: str
    ) -> Dict:
        """
        Handle incoming message in an active conversation.
        
        Args:
            lead_id: ID of lead sending message
            incoming_message: Message content
            channel: Communication channel
            
        Returns:
            Dict with response and conversation state
        """
        try:
            self.logger.info(f"Handling conversation for lead {lead_id}")
            
            # Mark conversation as active
            self.active_conversations.add(lead_id)
            
            # Analyze incoming message
            sentiment = await self._analyze_sentiment(incoming_message)
            intent = await self._detect_conversation_intent(incoming_message)
            
            # Retrieve conversation context
            context = await self._get_conversation_context(lead_id)
            
            # Determine if escalation needed
            if await self._should_escalate(incoming_message, sentiment, context):
                return await self._escalate_conversation(
                    lead_id, incoming_message, context
                )
            
            # Generate contextual response
            response = await self._generate_conversation_response(
                lead_id=lead_id,
                incoming_message=incoming_message,
                sentiment=sentiment,
                intent=intent,
                context=context,
                channel=channel
            )
            
            # Send response
            send_result = await self._send_message(
                lead=await self._get_lead(lead_id),
                message=response,
                channel=channel,
                strategy=EngagementStrategy.IMMEDIATE_RESPONSE
            )
            
            # Update conversation state
            await self._update_conversation_state(
                lead_id, incoming_message, response, sentiment
            )
            
            return {
                "lead_id": lead_id,
                "response": response,
                "sentiment": sentiment,
                "intent": intent,
                "send_result": send_result,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(
                f"Error handling conversation for {lead_id}: {str(e)}"
            )
            return {
                "lead_id": lead_id,
                "status": "error",
                "error": str(e)
            }
    
    async def execute_nurture_sequence(
        self,
        lead: Lead,
        campaign_id: str,
        sequence_type: str = "standard"
    ) -> Dict:
        """
        Execute multi-touch nurture sequence.
        
        Args:
            lead: Lead to nurture
            campaign_id: Associated campaign ID
            sequence_type: Type of sequence (standard, accelerated, etc.)
            
        Returns:
            Dict with sequence execution details
        """
        try:
            self.logger.info(
                f"Starting nurture sequence for lead {lead.lead_id}"
            )
            
            # Define sequence steps
            sequence_steps = self._get_sequence_steps(sequence_type)
            
            results = []
            for step in sequence_steps:
                # Check if lead is still eligible
                if not await self._is_lead_eligible_for_sequence(lead):
                    break
                
                # Wait for step delay
                if step.get("delay_hours", 0) > 0:
                    await asyncio.sleep(step["delay_hours"] * 3600)
                
                # Execute step
                step_result = await self.engage_lead(
                    lead=lead,
                    strategy=step["strategy"],
                    channel=step["channel"],
                    context={"sequence_step": step["name"]}
                )
                
                results.append(step_result)
                
                # Check for engagement
                if await self._check_lead_engagement(lead.lead_id):
                    self.logger.info(
                        f"Lead {lead.lead_id} engaged, ending sequence"
                    )
                    break
            
            await self.log_action(
                action_type="nurture_sequence",
                entity_id=lead.lead_id,
                details={
                    "campaign_id": campaign_id,
                    "sequence_type": sequence_type,
                    "steps_completed": len(results),
                    "results": results
                }
            )
            
            return {
                "lead_id": lead.lead_id,
                "campaign_id": campaign_id,
                "steps_completed": len(results),
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(
                f"Error executing nurture sequence for {lead.lead_id}: {str(e)}"
            )
            return {
                "lead_id": lead.lead_id,
                "status": "error",
                "error": str(e)
            }
    
    async def optimize_send_time(
        self,
        lead: Lead,
        channel: str
    ) -> datetime:
        """
        Determine optimal send time for a lead.
        
        Args:
            lead: Lead to optimize for
            channel: Communication channel
            
        Returns:
            Optimal datetime for sending
        """
        # Get lead's historical engagement patterns
        engagement_history = await self._get_engagement_history(lead.lead_id)
        
        # Analyze engagement times
        optimal_hour = self._analyze_optimal_time(
            engagement_history, channel
        )
        
        # Get next available slot
        now = datetime.now()
        next_send = now.replace(
            hour=optimal_hour,
            minute=0,
            second=0,
            microsecond=0
        )
        
        # If optimal time has passed today, schedule for tomorrow
        if next_send <= now:
            next_send += timedelta(days=1)
        
        # Respect channel-specific constraints
        next_send = self._apply_channel_constraints(next_send, channel)
        
        return next_send
    
    async def run_ab_test(
        self,
        lead_ids: List[str],
        variant_a: Dict,
        variant_b: Dict,
        metric: str = "response_rate"
    ) -> Dict:
        """
        Run A/B test on message variants.
        
        Args:
            lead_ids: List of lead IDs to test
            variant_a: First message variant
            variant_b: Second message variant
            metric: Success metric to measure
            
        Returns:
            Dict with test results
        """
        try:
            # Split leads into two groups
            mid_point = len(lead_ids) // 2
            group_a = lead_ids[:mid_point]
            group_b = lead_ids[mid_point:]
            
            # Send variant A to group A
            results_a = []
            for lead_id in group_a:
                lead = await self._get_lead(lead_id)
                if lead:
                    result = await self.engage_lead(
                        lead=lead,
                        strategy=variant_a["strategy"],
                        channel=variant_a["channel"],
                        context={"ab_test": "variant_a"}
                    )
                    results_a.append(result)
            
            # Send variant B to group B
            results_b = []
            for lead_id in group_b:
                lead = await self._get_lead(lead_id)
                if lead:
                    result = await self.engage_lead(
                        lead=lead,
                        strategy=variant_b["strategy"],
                        channel=variant_b["channel"],
                        context={"ab_test": "variant_b"}
                    )
                    results_b.append(result)
            
            # Wait for results to accumulate
            await asyncio.sleep(3600)  # Wait 1 hour
            
            # Calculate metrics
            metric_a = await self._calculate_test_metric(group_a, metric)
            metric_b = await self._calculate_test_metric(group_b, metric)
            
            # Determine winner
            winner = "variant_a" if metric_a > metric_b else "variant_b"
            confidence = abs(metric_a - metric_b) / max(metric_a, metric_b)
            
            test_results = {
                "variant_a": {
                    "leads_count": len(group_a),
                    "metric_value": metric_a,
                    "results": results_a
                },
                "variant_b": {
                    "leads_count": len(group_b),
                    "metric_value": metric_b,
                    "results": results_b
                },
                "winner": winner,
                "confidence": confidence,
                "metric": metric,
                "status": "success"
            }
            
            await self.log_action(
                action_type="ab_test",
                entity_id="engagement_agent",
                details=test_results
            )
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Error running A/B test: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    # Private helper methods
    
    def _initialize_templates(self) -> Dict:
        """Initialize message templates by strategy and tone."""
        return {
            EngagementStrategy.IMMEDIATE_RESPONSE: {
                MessageTone.PROFESSIONAL: (
                    "Hello {first_name},\n\n"
                    "Thank you for your inquiry regarding {topic}. "
                    "I'd be happy to assist you with {specific_need}.\n\n"
                    "Best regards"
                ),
                MessageTone.FRIENDLY: (
                    "Hi {first_name}!\n\n"
                    "Thanks for reaching out! I saw you're interested in {topic}. "
                    "Let me help you with that.\n\n"
                    "Cheers"
                ),
                MessageTone.CASUAL: (
                    "Hey {first_name},\n\n"
                    "Got your message about {topic}. "
                    "Here's what I can do for you...\n\n"
                    "Talk soon"
                )
            },
            EngagementStrategy.NURTURE_SEQUENCE: {
                MessageTone.PROFESSIONAL: (
                    "Dear {first_name},\n\n"
                    "I wanted to follow up on our previous conversation about {topic}. "
                    "Many {company} professionals have found value in {benefit}.\n\n"
                    "Would you like to learn more?\n\n"
                    "Best regards"
                ),
                MessageTone.FRIENDLY: (
                    "Hi {first_name},\n\n"
                    "Hope you're doing well! I thought you might be interested in "
                    "{benefit} based on what we discussed about {topic}.\n\n"
                    "Let me know if you'd like more info!\n\n"
                    "Cheers"
                )
            },
            EngagementStrategy.REACTIVATION: {
                MessageTone.FRIENDLY: (
                    "Hi {first_name},\n\n"
                    "It's been a while since we last connected! "
                    "I noticed you were interested in {topic} - "
                    "we've made some exciting updates since then.\n\n"
                    "Worth catching up?\n\n"
                    "Best"
                ),
                MessageTone.URGENT: (
                    "Hi {first_name},\n\n"
                    "Quick heads up - the opportunity you were interested in "
                    "regarding {topic} is ending soon. "
                    "Don't miss out!\n\n"
                    "Let's reconnect.\n\n"
                    "Best"
                )
            }
        }
    
    async def _gather_personalization_data(
        self,
        lead: Lead,
        context: Optional[Dict]
    ) -> Dict:
        """Gather all available personalization data."""
        data = {
            "first_name": lead.first_name or "there",
            "last_name": lead.last_name or "",
            "company": lead.company or "your organization",
            "email": lead.email or "",
            "topic": "our services",
            "specific_need": "your goals",
            "benefit": "significant value"
        }
        
        # Add context data
        if context:
            data.update(context)
        
        # Add recent interaction data
        recent_interactions = await self._get_lead_interactions(lead.lead_id, limit=1)
        if recent_interactions:
            data["recent_interaction"] = recent_interactions[0].message_content[:50]
        
        return data
    
    async def _determine_message_tone(
        self,
        lead: Lead,
        strategy: EngagementStrategy
    ) -> MessageTone:
        """Determine appropriate message tone."""
        # Analyze lead profile
        interactions = await self._get_lead_interactions(lead.lead_id)
        
        # Default tones by strategy
        strategy_tones = {
            EngagementStrategy.IMMEDIATE_RESPONSE: MessageTone.FRIENDLY,
            EngagementStrategy.NURTURE_SEQUENCE: MessageTone.PROFESSIONAL,
            EngagementStrategy.REACTIVATION: MessageTone.FRIENDLY,
            EngagementStrategy.RELATIONSHIP_BUILDING: MessageTone.FRIENDLY,
            EngagementStrategy.EDUCATION: MessageTone.PROFESSIONAL
        }
        
        return strategy_tones.get(strategy, MessageTone.PROFESSIONAL)
    
    def _validate_message(self, message: str, channel: str) -> str:
        """Validate and adjust message for channel constraints."""
        config = self.channel_config.get(channel, {})
        max_length = config.get("max_length", 2000)
        
        if len(message) > max_length:
            # Truncate with ellipsis
            message = message[:max_length-3] + "..."
        
        return message
    
    def _format_for_channel(self, message: str, channel: str) -> str:
        """Apply channel-specific formatting."""
        config = self.channel_config.get(channel, {})
        format_type = config.get("format", "plain")
        
        if format_type == "html" and channel == "email":
            # Add basic HTML formatting
            message = message.replace("\n\n", "</p><p>")
            message = f"<p>{message}</p>"
        
        elif format_type == "plain" and channel == "sms":
            # Remove extra whitespace for SMS
            message = " ".join(message.split())
        
        return message
    
    async def _add_dynamic_elements(
        self,
        message: str,
        lead: Lead,
        strategy: EngagementStrategy
    ) -> str:
        """Add dynamic elements like CTAs, urgency, social proof."""
        # Add CTA based on strategy
        if strategy == EngagementStrategy.IMMEDIATE_RESPONSE:
            message += "\n\nReply to this message to continue our conversation."
        
        elif strategy == EngagementStrategy.NURTURE_SEQUENCE:
            message += "\n\nClick here to schedule a quick call: [LINK]"
        
        return message
    
    async def _send_message(
        self,
        lead: Lead,
        message: str,
        channel: str,
        strategy: EngagementStrategy
    ) -> Dict:
        """Send message through specified channel."""
        # Placeholder for actual sending logic
        # Would integrate with email service, SMS gateway, etc.
        
        return {
            "sent": True,
            "timestamp": datetime.now().isoformat(),
            "channel": channel,
            "message_id": f"msg_{lead.lead_id}_{datetime.now().timestamp()}"
        }
    
    async def _log_engagement(
        self,
        lead: Lead,
        message: str,
        channel: str,
        strategy: EngagementStrategy,
        result: Dict
    ):
        """Log engagement to database."""
        await self.log_action(
            action_type="engagement",
            entity_id=lead.lead_id,
            details={
                "channel": channel,
                "strategy": strategy.value,
                "message_length": len(message),
                "result": result
            }
        )
    
    async def _analyze_sentiment(self, message: str) -> str:
        """Analyze sentiment of incoming message."""
        # Simple keyword-based sentiment (could use ML model)
        positive_words = ["great", "thanks", "interested", "love", "perfect"]
        negative_words = ["not", "never", "disappointed", "issue", "problem"]
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"
    
    async def _detect_conversation_intent(self, message: str) -> str:
        """Detect intent from incoming message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["buy", "purchase", "price"]):
            return "purchase_intent"
        elif any(word in message_lower for word in ["help", "support", "issue"]):
            return "support_needed"
        elif any(word in message_lower for word in ["more info", "tell me", "learn"]):
            return "information_request"
        
        return "general_inquiry"
    
    async def _get_conversation_context(self, lead_id: str) -> Dict:
        """Retrieve conversation context."""
        interactions = await self._get_lead_interactions(lead_id, limit=5)
        
        return {
            "interaction_count": len(interactions),
            "recent_topics": [],
            "sentiment_trend": "neutral",
            "last_interaction_date": interactions[0].interaction_date if interactions else None
        }
    
    async def _should_escalate(
        self,
        message: str,
        sentiment: str,
        context: Dict
    ) -> bool:
        """Determine if conversation should be escalated."""
        # Escalate on negative sentiment with specific keywords
        escalation_keywords = ["refund", "cancel", "angry", "terrible", "lawsuit"]
        
        if sentiment == "negative":
            message_lower = message.lower()
            if any(keyword in message_lower for keyword in escalation_keywords):
                return True
        
        return False
    
    async def _escalate_conversation(
        self,
        lead_id: str,
        message: str,
        context: Dict
    ) -> Dict:
        """Escalate conversation to human agent."""
        escalation_result = await self.request_handoff(
            to_agent_type="human_agent",
            reason="negative_sentiment_escalation",
            context={
                "lead_id": lead_id,
                "triggering_message": message,
                "context": context
            }
        )
        
        return {
            "lead_id": lead_id,
            "action": "escalated",
            "escalation_result": escalation_result,
            "status": "success"
        }
    
    async def _generate_conversation_response(
        self,
        lead_id: str,
        incoming_message: str,
        sentiment: str,
        intent: str,
        context: Dict,
        channel: str
    ) -> str:
        """Generate contextual conversation response."""
        # Base response on intent
        if intent == "purchase_intent":
            response = "I'm glad you're interested! Let me get you the information you need..."
        elif intent == "support_needed":
            response = "I'm here to help. Can you tell me more about what you're experiencing?"
        elif intent == "information_request":
            response = "I'd be happy to provide more details. What specifically would you like to know?"
        else:
            response = "Thanks for your message. How can I assist you today?"
        
        # Adjust for sentiment
        if sentiment == "negative":
            response = "I understand your concern. " + response
        
        return response
    
    async def _update_conversation_state(
        self,
        lead_id: str,
        incoming_message: str,
        response: str,
        sentiment: str
    ):
        """Update conversation state in database."""
        # Placeholder for database update
        pass
    
    def _get_sequence_steps(self, sequence_type: str) -> List[Dict]:
        """Get nurture sequence steps definition."""
        sequences = {
            "standard": [
                {
                    "name": "initial_welcome",
                    "delay_hours": 0,
                    "strategy": EngagementStrategy.IMMEDIATE_RESPONSE,
                    "channel": "email"
                },
                {
                    "name": "value_proposition",
                    "delay_hours": 48,
                    "strategy": EngagementStrategy.EDUCATION,
                    "channel": "email"
                },
                {
                    "name": "social_proof",
                    "delay_hours": 96,
                    "strategy": EngagementStrategy.RELATIONSHIP_BUILDING,
                    "channel": "email"
                },
                {
                    "name": "final_cta",
                    "delay_hours": 168,
                    "strategy": EngagementStrategy.NURTURE_SEQUENCE,
                    "channel": "sms"
                }
            ],
            "accelerated": [
                {
                    "name": "urgent_offer",
                    "delay_hours": 0,
                    "strategy": EngagementStrategy.IMMEDIATE_RESPONSE,
                    "channel": "sms"
                },
                {
                    "name": "follow_up",
                    "delay_hours": 24,
                    "strategy": EngagementStrategy.NURTURE_SEQUENCE,
                    "channel": "email"
                }
            ]
        }
        
        return sequences.get(sequence_type, sequences["standard"])
    
    async def _is_lead_eligible_for_sequence(self, lead: Lead) -> bool:
        """Check if lead is still eligible for sequence."""
        # Check if lead has converted or unsubscribed
        return True  # Placeholder
    
    async def _check_lead_engagement(self, lead_id: str) -> bool:
        """Check if lead has engaged recently."""
        interactions = await self._get_lead_interactions(lead_id, limit=1)
        if interactions:
            last_interaction = interactions[0]
            hours_since = (datetime.now() - last_interaction.interaction_date).total_seconds() / 3600
            return hours_since < 24
        return False
    
    async def _get_engagement_history(self, lead_id: str) -> List[Dict]:
        """Get engagement history for timing analysis."""
        interactions = await self._get_lead_interactions(lead_id)
        return [
            {
                "timestamp": i.interaction_date,
                "channel": i.channel,
                "engaged": i.direction == "inbound"
            }
            for i in interactions
        ]
    
    def _analyze_optimal_time(
        self,
        engagement_history: List[Dict],
        channel: str
    ) -> int:
        """Analyze optimal hour for engagement."""
        if not engagement_history:
            # Default optimal times
            defaults = {"email": 10, "sms": 14, "phone": 15}
            return defaults.get(channel, 10)
        
        # Count engagements by hour
        hour_counts = {}
        for event in engagement_history:
            if event.get("engaged"):
                hour = event["timestamp"].hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if hour_counts:
            return max(hour_counts, key=hour_counts.get)
        
        return 10  # Default fallback
    
    def _apply_channel_constraints(
        self,
        send_time: datetime,
        channel: str
    ) -> datetime:
        """Apply channel-specific time constraints."""
        # Ensure within business hours for phone
        if channel == "phone":
            if send_time.hour < 9:
                send_time = send_time.replace(hour=9)
            elif send_time.hour >= 18:
                send_time = send_time.replace(hour=14)
                send_time += timedelta(days=1)
        
        # Avoid weekends
        if send_time.weekday() >= 5:  # Saturday or Sunday
            days_to_add = 7 - send_time.weekday()
            send_time += timedelta(days=days_to_add)
        
        return send_time
    
    async def _calculate_test_metric(
        self,
        lead_ids: List[str],
        metric: str
    ) -> float:
        """Calculate A/B test metric."""
        if metric == "response_rate":
            responses = 0
            for lead_id in lead_ids:
                if await self._check_lead_engagement(lead_id):
                    responses += 1
            return responses / len(lead_ids) if lead_ids else 0.0
        
        return 0.0
    
    def _get_fallback_template(self, strategy: EngagementStrategy) -> str:
        """Get fallback template."""
        return "Hello {first_name}, thank you for your interest. How can we help you today?"
    
    async def _get_lead(self, lead_id: str) -> Optional[Lead]:
        """Fetch lead from database."""
        # Placeholder
        pass
    
    async def _get_lead_interactions(
        self,
        lead_id: str,
        limit: Optional[int] = None
    ) -> List[Interaction]:
        """Fetch lead interactions."""
        # Placeholder
        return []
    
    async def health_check(self) -> Dict:
        """Perform health check for Engagement Agent."""
        base_health = await super().health_check()
        
        engagement_health = {
            "active_conversations": len(self.active_conversations),
            "template_count": sum(len(tones) for tones in self.templates.values()),
            "supported_channels": list(self.channel_config.keys()),
            "capabilities": [
                "message_generation",
                "conversation_management",
                "nurture_sequences",
                "ab_testing"
            ]
        }
        
        base_health.update(engagement_health)
        return base_health