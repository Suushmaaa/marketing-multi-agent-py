"""
Lead Triage Agent - Intent Classification, Scoring & Routing
Handles initial lead assessment and intelligent routing decisions.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from agents.base_agent import BaseAgent, AgentCapability
from database.models import Lead, Interaction, LeadScore, AgentHandoff
from config.settings import Settings


class IntentCategory(Enum):
    """Lead intent categories"""
    PURCHASE = "purchase"
    INQUIRY = "inquiry"
    SUPPORT = "support"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    UNQUALIFIED = "unqualified"


class LeadPriority(Enum):
    """Lead priority levels"""
    CRITICAL = "critical"  # Hot leads, immediate action
    HIGH = "high"  # Qualified, high intent
    MEDIUM = "medium"  # Interested, needs nurturing
    LOW = "low"  # Cold, minimal engagement
    DISQUALIFIED = "disqualified"  # Not a fit


class LeadTriageAgent(BaseAgent):
    """
    Specialized agent for lead triage operations.
    
    Capabilities:
    - Intent classification from interactions
    - Multi-factor lead scoring
    - Intelligent routing decisions
    - Real-time priority updates
    - Handoff coordination
    """
    
    def __init__(self, settings: Settings):
        super().__init__(
            agent_id="lead_triage_001",
            agent_type="lead_triage",
            settings=settings
        )
        
        # Intent classification weights
        self.intent_keywords = {
            IntentCategory.PURCHASE: [
                "buy", "purchase", "price", "cost", "quote",
                "order", "payment", "checkout", "billing"
            ],
            IntentCategory.INQUIRY: [
                "information", "details", "learn more", "tell me",
                "explain", "how does", "what is", "features"
            ],
            IntentCategory.SUPPORT: [
                "help", "issue", "problem", "not working",
                "error", "troubleshoot", "fix", "assistance"
            ],
            IntentCategory.FEEDBACK: [
                "feedback", "suggestion", "review", "comment",
                "opinion", "thoughts", "experience"
            ],
            IntentCategory.COMPLAINT: [
                "disappointed", "unhappy", "frustrated", "angry",
                "terrible", "worst", "refund", "cancel"
            ]
        }
        
        # Scoring factors and weights
        self.scoring_weights = {
            "engagement_score": 0.25,
            "intent_score": 0.30,
            "timing_score": 0.15,
            "firmographic_score": 0.20,
            "behavioral_score": 0.10
        }
        
        self.logger.info(f"Lead Triage Agent initialized: {self.agent_id}")
    
    async def process_new_lead(self, lead: Lead) -> Dict:
        """
        Process a new lead through complete triage workflow.
        
        Args:
            lead: Lead object to process
            
        Returns:
            Dict with triage results including score, intent, and routing
        """
        try:
            self.logger.info(f"Processing new lead: {lead.lead_id}")
            
            # Step 1: Classify intent from available data
            intent = await self.classify_intent(lead)
            
            # Step 2: Calculate comprehensive lead score
            score = await self.calculate_lead_score(lead, intent)
            
            # Step 3: Determine priority level
            priority = self.determine_priority(score, intent)
            
            # Step 4: Make routing decision
            routing = await self.determine_routing(lead, intent, priority)
            
            # Step 5: Store results
            await self._store_triage_results(lead, intent, score, priority, routing)
            
            # Step 6: Log the action
            await self.log_action(
                action_type="lead_triage",
                entity_id=lead.lead_id,
                details={
                    "intent": intent.value,
                    "score": score,
                    "priority": priority.value,
                    "routing": routing
                }
            )
            
            result = {
                "lead_id": lead.lead_id,
                "intent": intent.value,
                "score": score,
                "priority": priority.value,
                "routing": routing,
                "status": "success"
            }
            
            self.logger.info(f"Lead triage completed: {lead.lead_id} - {priority.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing lead {lead.lead_id}: {str(e)}")
            return {
                "lead_id": lead.lead_id,
                "status": "error",
                "error": str(e)
            }
    
    async def classify_intent(self, lead: Lead) -> IntentCategory:
        """
        Classify lead intent from available signals.
        
        Args:
            lead: Lead to classify
            
        Returns:
            IntentCategory enum
        """
        # Get recent interactions
        interactions = await self._get_lead_interactions(lead.lead_id, limit=5)
        
        if not interactions:
            return IntentCategory.INQUIRY  # Default for new leads
        
        # Aggregate text from interactions
        combined_text = " ".join([
            i.message_content.lower() 
            for i in interactions 
            if i.message_content
        ])
        
        # Score each intent category
        intent_scores = {}
        for category, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            intent_scores[category] = score
        
        # Get highest scoring intent
        if max(intent_scores.values()) == 0:
            return IntentCategory.INQUIRY
        
        return max(intent_scores, key=intent_scores.get)
    
    async def calculate_lead_score(
        self,
        lead: Lead,
        intent: IntentCategory
    ) -> float:
        """
        Calculate comprehensive lead score (0-100).
        
        Args:
            lead: Lead to score
            intent: Classified intent
            
        Returns:
            Float score between 0 and 100
        """
        scores = {}
        
        # 1. Engagement Score (0-100)
        scores["engagement_score"] = await self._calculate_engagement_score(lead)
        
        # 2. Intent Score (0-100)
        scores["intent_score"] = self._calculate_intent_score(intent)
        
        # 3. Timing Score (0-100)
        scores["timing_score"] = self._calculate_timing_score(lead)
        
        # 4. Firmographic Score (0-100)
        scores["firmographic_score"] = self._calculate_firmographic_score(lead)
        
        # 5. Behavioral Score (0-100)
        scores["behavioral_score"] = await self._calculate_behavioral_score(lead)
        
        # Calculate weighted final score
        final_score = sum(
            scores[factor] * weight
            for factor, weight in self.scoring_weights.items()
        )
        
        return round(final_score, 2)
    
    def determine_priority(
        self,
        score: float,
        intent: IntentCategory
    ) -> LeadPriority:
        """
        Determine lead priority based on score and intent.
        
        Args:
            score: Lead score (0-100)
            intent: Classified intent
            
        Returns:
            LeadPriority enum
        """
        # Critical: High score + purchase intent
        if score >= 80 and intent == IntentCategory.PURCHASE:
            return LeadPriority.CRITICAL
        
        # High: Good score or purchase intent
        if score >= 70 or intent == IntentCategory.PURCHASE:
            return LeadPriority.HIGH
        
        # Medium: Moderate score or inquiry/support
        if score >= 50 or intent in [IntentCategory.INQUIRY, IntentCategory.SUPPORT]:
            return LeadPriority.MEDIUM
        
        # Disqualified: Unqualified intent
        if intent == IntentCategory.UNQUALIFIED:
            return LeadPriority.DISQUALIFIED
        
        # Low: Everything else
        return LeadPriority.LOW
    
    async def determine_routing(
        self,
        lead: Lead,
        intent: IntentCategory,
        priority: LeadPriority
    ) -> Dict:
        """
        Determine routing strategy for the lead.
        
        Args:
            lead: Lead to route
            intent: Classified intent
            priority: Determined priority
            
        Returns:
            Dict with routing decision
        """
        routing = {
            "assigned_agent": None,
            "channel": None,
            "action": None,
            "timing": None
        }
        
        # Critical priority: Immediate engagement
        if priority == LeadPriority.CRITICAL:
            routing["assigned_agent"] = "engagement_agent"
            routing["channel"] = "phone"  # Direct call
            routing["action"] = "immediate_contact"
            routing["timing"] = "now"
        
        # High priority: Quick engagement
        elif priority == LeadPriority.HIGH:
            routing["assigned_agent"] = "engagement_agent"
            routing["channel"] = self._select_best_channel(lead)
            routing["action"] = "personalized_outreach"
            routing["timing"] = "within_1_hour"
        
        # Medium priority: Nurture campaign
        elif priority == LeadPriority.MEDIUM:
            routing["assigned_agent"] = "engagement_agent"
            routing["channel"] = "email"
            routing["action"] = "nurture_sequence"
            routing["timing"] = "within_24_hours"
        
        # Low priority: Automated follow-up
        elif priority == LeadPriority.LOW:
            routing["assigned_agent"] = "campaign_optimization"
            routing["channel"] = "email"
            routing["action"] = "automated_drip"
            routing["timing"] = "within_7_days"
        
        # Disqualified: Archive
        else:
            routing["assigned_agent"] = None
            routing["channel"] = None
            routing["action"] = "archive"
            routing["timing"] = "immediate"
        
        # Handle special intents
        if intent == IntentCategory.COMPLAINT:
            routing["assigned_agent"] = "human_agent"
            routing["action"] = "escalate"
            routing["timing"] = "immediate"
        
        elif intent == IntentCategory.SUPPORT:
            routing["assigned_agent"] = "support_agent"
            routing["action"] = "support_ticket"
            routing["timing"] = "within_2_hours"
        
        return routing
    
    async def re_score_lead(self, lead_id: str) -> Dict:
        """
        Re-score an existing lead with updated data.
        
        Args:
            lead_id: ID of lead to re-score
            
        Returns:
            Dict with updated scoring results
        """
        try:
            # Fetch lead data
            lead = await self._get_lead(lead_id)
            if not lead:
                return {"status": "error", "error": "Lead not found"}
            
            # Re-run triage process
            result = await self.process_new_lead(lead)
            
            await self.log_action(
                action_type="lead_re_score",
                entity_id=lead_id,
                details=result
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error re-scoring lead {lead_id}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def bulk_triage(self, lead_ids: List[str]) -> List[Dict]:
        """
        Process multiple leads in parallel.
        
        Args:
            lead_ids: List of lead IDs to process
            
        Returns:
            List of triage results
        """
        tasks = []
        for lead_id in lead_ids:
            lead = await self._get_lead(lead_id)
            if lead:
                tasks.append(self.process_new_lead(lead))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            r if not isinstance(r, Exception) else {"status": "error", "error": str(r)}
            for r in results
        ]
    
    # Private helper methods
    
    async def _calculate_engagement_score(self, lead: Lead) -> float:
        """Calculate engagement score based on interaction history."""
        interactions = await self._get_lead_interactions(lead.lead_id)
        
        if not interactions:
            return 30.0  # Base score for new leads
        
        # Factors: recency, frequency, quality
        score = 30.0
        
        # Recency bonus (up to +30 points)
        if interactions:
            days_since_last = (datetime.now() - interactions[0].interaction_date).days
            recency_score = max(0, 30 - days_since_last)
            score += recency_score
        
        # Frequency bonus (up to +20 points)
        interaction_count = len(interactions)
        frequency_score = min(20, interaction_count * 2)
        score += frequency_score
        
        # Quality bonus (up to +20 points)
        positive_interactions = sum(
            1 for i in interactions
            if i.sentiment and i.sentiment.lower() in ["positive", "neutral"]
        )
        quality_score = min(20, (positive_interactions / max(1, interaction_count)) * 20)
        score += quality_score
        
        return min(100.0, score)
    
    def _calculate_intent_score(self, intent: IntentCategory) -> float:
        """Score based on intent category."""
        intent_scores = {
            IntentCategory.PURCHASE: 100.0,
            IntentCategory.INQUIRY: 70.0,
            IntentCategory.SUPPORT: 60.0,
            IntentCategory.FEEDBACK: 50.0,
            IntentCategory.COMPLAINT: 40.0,
            IntentCategory.UNQUALIFIED: 10.0
        }
        return intent_scores.get(intent, 50.0)
    
    def _calculate_timing_score(self, lead: Lead) -> float:
        """Score based on timing factors."""
        score = 50.0
        
        # Lead age (newer is better)
        if lead.created_at:
            days_old = (datetime.now() - lead.created_at).days
            age_score = max(0, 50 - (days_old * 2))
            score += age_score
        
        return min(100.0, score)
    
    def _calculate_firmographic_score(self, lead: Lead) -> float:
        """Score based on firmographic data."""
        score = 50.0
        
        # Company size indicator
        if lead.company and lead.company.lower() not in ["none", "individual", ""]:
            score += 20
        
        # Email domain quality
        if lead.email and not any(
            domain in lead.email.lower()
            for domain in ["gmail", "yahoo", "hotmail", "outlook"]
        ):
            score += 15  # Business email
        
        # Phone provided
        if lead.phone:
            score += 15
        
        return min(100.0, score)
    
    async def _calculate_behavioral_score(self, lead: Lead) -> float:
        """Score based on behavioral signals."""
        interactions = await self._get_lead_interactions(lead.lead_id)
        
        if not interactions:
            return 50.0
        
        score = 50.0
        
        # Response rate
        responses = [i for i in interactions if i.direction == "inbound"]
        if len(interactions) > 0:
            response_rate = len(responses) / len(interactions)
            score += response_rate * 30
        
        # Engagement depth (message length)
        avg_length = sum(
            len(i.message_content or "")
            for i in responses
        ) / max(1, len(responses))
        
        if avg_length > 100:
            score += 20
        elif avg_length > 50:
            score += 10
        
        return min(100.0, score)
    
    def _select_best_channel(self, lead: Lead) -> str:
        """Select best communication channel for lead."""
        # Prefer channels with existing interactions
        # For now, return default logic
        if lead.phone:
            return "sms"
        return "email"
    
    async def _store_triage_results(
        self,
        lead: Lead,
        intent: IntentCategory,
        score: float,
        priority: LeadPriority,
        routing: Dict
    ):
        """Store triage results to database."""
        # This would use your database session
        # Placeholder for actual DB operations
        pass
    
    async def _get_lead(self, lead_id: str) -> Optional[Lead]:
        """Fetch lead from database."""
        # Placeholder - implement with actual DB
        pass
    
    async def _get_lead_interactions(
        self,
        lead_id: str,
        limit: Optional[int] = None
    ) -> List[Interaction]:
        """Fetch lead interactions from database."""
        # Placeholder - implement with actual DB
        return []
    
    async def health_check(self) -> Dict:
        """Perform health check for Lead Triage Agent."""
        base_health = await super().health_check()
        
        # Add triage-specific checks
        triage_health = {
            "intent_categories": len(self.intent_keywords),
            "scoring_factors": len(self.scoring_weights),
            "capabilities": ["intent_classification", "lead_scoring", "routing"]
        }
        
        base_health.update(triage_health)
        return base_health