"""
Campaign Optimization Agent - Performance Analysis & Strategy Refinement
Handles campaign analytics, optimization, and strategic recommendations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict

from agents.base_agent import BaseAgent, AgentCapability
from database.models import Campaign, Lead, Interaction, ABTest, CampaignMetrics
from config.settings import Settings


class OptimizationGoal(Enum):
    """Campaign optimization goals"""
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MAXIMIZE_ENGAGEMENT = "maximize_engagement"
    MINIMIZE_COST = "minimize_cost"
    IMPROVE_QUALITY = "improve_quality"
    INCREASE_REACH = "increase_reach"


class MetricType(Enum):
    """Campaign metric types"""
    CONVERSION_RATE = "conversion_rate"
    ENGAGEMENT_RATE = "engagement_rate"
    RESPONSE_RATE = "response_rate"
    COST_PER_LEAD = "cost_per_lead"
    ROI = "roi"
    LEAD_QUALITY_SCORE = "lead_quality_score"


class CampaignOptimizationAgent(BaseAgent):
    """
    Specialized agent for campaign optimization operations.
    
    Capabilities:
    - Performance tracking and analytics
    - A/B test analysis and recommendations
    - Budget optimization
    - Audience segmentation refinement
    - Predictive performance modeling
    - Strategic recommendations
    """
    
    def __init__(self, settings: Settings):
        super().__init__(
            agent_id="campaign_opt_001",
            agent_type="campaign_optimization",
            settings=settings
        )
        
        # Performance thresholds
        self.performance_thresholds = {
            MetricType.CONVERSION_RATE: {
                "excellent": 0.15,
                "good": 0.10,
                "fair": 0.05,
                "poor": 0.02
            },
            MetricType.ENGAGEMENT_RATE: {
                "excellent": 0.40,
                "good": 0.25,
                "fair": 0.15,
                "poor": 0.05
            },
            MetricType.RESPONSE_RATE: {
                "excellent": 0.30,
                "good": 0.20,
                "fair": 0.10,
                "poor": 0.03
            }
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            "underperforming": [
                "refine_targeting",
                "test_new_messaging",
                "adjust_timing",
                "increase_budget"
            ],
            "performing_well": [
                "scale_budget",
                "expand_audience",
                "test_variations"
            ],
            "saturated": [
                "pause_campaign",
                "find_new_segments",
                "refresh_creative"
            ]
        }
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        self.logger.info(f"Campaign Optimization Agent initialized: {self.agent_id}")
    
    async def analyze_campaign_performance(
        self,
        campaign_id: str,
        time_period: Optional[int] = 30  # days
    ) -> Dict:
        """
        Comprehensive campaign performance analysis.
        
        Args:
            campaign_id: ID of campaign to analyze
            time_period: Number of days to analyze
            
        Returns:
            Dict with complete performance analysis
        """
        try:
            self.logger.info(f"Analyzing campaign performance: {campaign_id}")
            
            # Check cache first
            cache_key = f"{campaign_id}_{time_period}"
            if cache_key in self.analysis_cache:
                cached_time, cached_data = self.analysis_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_ttl:
                    self.logger.info("Returning cached analysis")
                    return cached_data
            
            # Fetch campaign data
            campaign = await self._get_campaign(campaign_id)
            if not campaign:
                return {"status": "error", "error": "Campaign not found"}
            
            # Calculate key metrics
            metrics = await self._calculate_campaign_metrics(
                campaign_id, time_period
            )
            
            # Analyze trends
            trends = await self._analyze_metric_trends(campaign_id, time_period)
            
            # Compare to benchmarks
            benchmarks = self._compare_to_benchmarks(metrics)
            
            # Identify issues and opportunities
            issues = await self._identify_issues(campaign_id, metrics)
            opportunities = await self._identify_opportunities(campaign_id, metrics)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                campaign_id, metrics, trends, issues, opportunities
            )
            
            # Calculate health score
            health_score = self._calculate_campaign_health_score(metrics)
            
            analysis = {
                "campaign_id": campaign_id,
                "campaign_name": campaign.name,
                "time_period_days": time_period,
                "metrics": metrics,
                "trends": trends,
                "benchmarks": benchmarks,
                "health_score": health_score,
                "issues": issues,
                "opportunities": opportunities,
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat(),
                "status": "success"
            }
            
            # Cache the analysis
            self.analysis_cache[cache_key] = (datetime.now(), analysis)
            
            # Log the action
            await self.log_action(
                action_type="campaign_analysis",
                entity_id=campaign_id,
                details={
                    "health_score": health_score,
                    "issues_found": len(issues),
                    "recommendations_made": len(recommendations)
                }
            )
            
            self.logger.info(
                f"Campaign analysis completed: {campaign_id} "
                f"(Health: {health_score:.1f})"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(
                f"Error analyzing campaign {campaign_id}: {str(e)}"
            )
            return {
                "campaign_id": campaign_id,
                "status": "error",
                "error": str(e)
            }
    
    async def optimize_campaign(
        self,
        campaign_id: str,
        goal: OptimizationGoal,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Execute optimization for a campaign.
        
        Args:
            campaign_id: ID of campaign to optimize
            goal: Optimization goal
            constraints: Budget, time, or other constraints
            
        Returns:
            Dict with optimization actions and expected impact
        """
        try:
            self.logger.info(
                f"Optimizing campaign {campaign_id} for goal: {goal.value}"
            )
            
            # Get current analysis
            analysis = await self.analyze_campaign_performance(campaign_id)
            if analysis.get("status") == "error":
                return analysis
            
            # Determine optimization actions
            actions = await self._determine_optimization_actions(
                campaign_id, goal, analysis, constraints
            )
            
            # Simulate expected impact
            expected_impact = await self._simulate_optimization_impact(
                campaign_id, actions, analysis
            )
            
            # Execute approved actions
            execution_results = await self._execute_optimization_actions(
                campaign_id, actions
            )
            
            optimization_result = {
                "campaign_id": campaign_id,
                "goal": goal.value,
                "actions_planned": actions,
                "expected_impact": expected_impact,
                "execution_results": execution_results,
                "status": "success"
            }
            
            await self.log_action(
                action_type="campaign_optimization",
                entity_id=campaign_id,
                details=optimization_result
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(
                f"Error optimizing campaign {campaign_id}: {str(e)}"
            )
            return {
                "campaign_id": campaign_id,
                "status": "error",
                "error": str(e)
            }
    
    async def run_ab_test_analysis(
        self,
        test_id: str,
        min_sample_size: int = 100
    ) -> Dict:
        """
        Analyze A/B test results with statistical significance.
        
        Args:
            test_id: ID of A/B test to analyze
            min_sample_size: Minimum sample size per variant
            
        Returns:
            Dict with test analysis and recommendations
        """
        try:
            self.logger.info(f"Analyzing A/B test: {test_id}")
            
            # Fetch test data
            test_data = await self._get_ab_test_data(test_id)
            
            # Check sample size
            if test_data["variant_a"]["sample_size"] < min_sample_size:
                return {
                    "test_id": test_id,
                    "status": "insufficient_data",
                    "message": "Not enough data for statistical analysis"
                }
            
            # Calculate metrics for each variant
            variant_a_metrics = await self._calculate_variant_metrics(
                test_data["variant_a"]
            )
            variant_b_metrics = await self._calculate_variant_metrics(
                test_data["variant_b"]
            )
            
            # Statistical significance test
            significance = self._calculate_statistical_significance(
                variant_a_metrics, variant_b_metrics
            )
            
            # Determine winner
            winner = self._determine_test_winner(
                variant_a_metrics, variant_b_metrics, significance
            )
            
            # Calculate lift
            lift = self._calculate_lift(variant_a_metrics, variant_b_metrics)
            
            # Generate insights
            insights = self._generate_test_insights(
                test_data, variant_a_metrics, variant_b_metrics, winner, lift
            )
            
            # Recommendation
            recommendation = self._generate_test_recommendation(
                winner, significance, lift
            )
            
            analysis = {
                "test_id": test_id,
                "test_name": test_data.get("name", "Unnamed Test"),
                "variant_a": variant_a_metrics,
                "variant_b": variant_b_metrics,
                "winner": winner,
                "statistical_significance": significance,
                "lift": lift,
                "insights": insights,
                "recommendation": recommendation,
                "status": "success"
            }
            
            await self.log_action(
                action_type="ab_test_analysis",
                entity_id=test_id,
                details=analysis
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing A/B test {test_id}: {str(e)}")
            return {
                "test_id": test_id,
                "status": "error",
                "error": str(e)
            }
    
    async def predict_campaign_performance(
        self,
        campaign_config: Dict,
        historical_data: Optional[Dict] = None
    ) -> Dict:
        """
        Predict campaign performance before launch.
        
        Args:
            campaign_config: Configuration for new campaign
            historical_data: Historical campaign data for modeling
            
        Returns:
            Dict with performance predictions
        """
        try:
            self.logger.info("Predicting campaign performance")
            
            # Get historical benchmark data
            if not historical_data:
                historical_data = await self._get_historical_benchmarks(
                    campaign_config.get("campaign_type"),
                    campaign_config.get("target_audience")
                )
            
            # Build prediction model features
            features = self._extract_prediction_features(
                campaign_config, historical_data
            )
            
            # Generate predictions
            predictions = {
                "expected_reach": self._predict_reach(features),
                "expected_engagement_rate": self._predict_engagement_rate(features),
                "expected_conversion_rate": self._predict_conversion_rate(features),
                "expected_cost_per_lead": self._predict_cost_per_lead(features),
                "confidence_interval": "80%"
            }
            
            # Calculate expected ROI
            predictions["expected_roi"] = self._calculate_expected_roi(
                campaign_config, predictions
            )
            
            # Risk assessment
            risk_assessment = self._assess_campaign_risk(
                campaign_config, predictions
            )
            
            # Recommendations for improvement
            pre_launch_recommendations = self._generate_pre_launch_recommendations(
                campaign_config, predictions, risk_assessment
            )
            
            result = {
                "campaign_config": campaign_config,
                "predictions": predictions,
                "risk_assessment": risk_assessment,
                "recommendations": pre_launch_recommendations,
                "predicted_at": datetime.now().isoformat(),
                "status": "success"
            }
            
            await self.log_action(
                action_type="performance_prediction",
                entity_id="new_campaign",
                details=result
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting performance: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def segment_audience(
        self,
        campaign_id: str,
        segmentation_criteria: List[str]
    ) -> Dict:
        """
        Segment campaign audience for targeted optimization.
        
        Args:
            campaign_id: ID of campaign
            segmentation_criteria: Criteria for segmentation
            
        Returns:
            Dict with segment analysis and recommendations
        """
        try:
            self.logger.info(f"Segmenting audience for campaign {campaign_id}")
            
            # Get campaign leads
            leads = await self._get_campaign_leads(campaign_id)
            
            # Perform segmentation
            segments = await self._perform_segmentation(
                leads, segmentation_criteria
            )
            
            # Analyze each segment
            segment_analyses = {}
            for segment_name, segment_leads in segments.items():
                analysis = await self._analyze_segment(
                    segment_name, segment_leads, campaign_id
                )
                segment_analyses[segment_name] = analysis
            
            # Identify best and worst performing segments
            best_segment = max(
                segment_analyses.items(),
                key=lambda x: x[1]["performance_score"]
            )
            worst_segment = min(
                segment_analyses.items(),
                key=lambda x: x[1]["performance_score"]
            )
            
            # Generate segment-specific recommendations
            recommendations = self._generate_segment_recommendations(
                segment_analyses
            )
            
            result = {
                "campaign_id": campaign_id,
                "total_leads": len(leads),
                "segments": segment_analyses,
                "best_performing_segment": best_segment[0],
                "worst_performing_segment": worst_segment[0],
                "recommendations": recommendations,
                "status": "success"
            }
            
            await self.log_action(
                action_type="audience_segmentation",
                entity_id=campaign_id,
                details=result
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error segmenting audience for {campaign_id}: {str(e)}"
            )
            return {
                "campaign_id": campaign_id,
                "status": "error",
                "error": str(e)
            }
    
    async def optimize_budget_allocation(
        self,
        campaign_ids: List[str],
        total_budget: float
    ) -> Dict:
        """
        Optimize budget allocation across multiple campaigns.
        
        Args:
            campaign_ids: List of campaign IDs
            total_budget: Total available budget
            
        Returns:
            Dict with optimal budget allocation
        """
        try:
            self.logger.info(
                f"Optimizing budget allocation for {len(campaign_ids)} campaigns"
            )
            
            # Get performance data for all campaigns
            campaign_performances = {}
            for campaign_id in campaign_ids:
                analysis = await self.analyze_campaign_performance(campaign_id)
                if analysis.get("status") == "success":
                    campaign_performances[campaign_id] = analysis
            
            # Calculate efficiency scores
            efficiency_scores = {}
            for campaign_id, performance in campaign_performances.items():
                efficiency_scores[campaign_id] = self._calculate_efficiency_score(
                    performance["metrics"]
                )
            
            # Optimize allocation using weighted distribution
            allocations = self._optimize_allocation(
                efficiency_scores, total_budget
            )
            
            # Calculate expected impact
            expected_impact = self._calculate_allocation_impact(
                allocations, campaign_performances
            )
            
            result = {
                "total_budget": total_budget,
                "campaign_count": len(campaign_ids),
                "allocations": allocations,
                "efficiency_scores": efficiency_scores,
                "expected_impact": expected_impact,
                "status": "success"
            }
            
            await self.log_action(
                action_type="budget_optimization",
                entity_id="multi_campaign",
                details=result
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing budget: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    # Private helper methods
    
    async def _calculate_campaign_metrics(
        self,
        campaign_id: str,
        time_period: int
    ) -> Dict:
        """Calculate all key campaign metrics."""
        cutoff_date = datetime.now() - timedelta(days=time_period)
        
        # Get campaign data
        leads = await self._get_campaign_leads(campaign_id, since=cutoff_date)
        interactions = await self._get_campaign_interactions(
            campaign_id, since=cutoff_date
        )
        
        total_leads = len(leads)
        engaged_leads = len([l for l in leads if l.last_interaction_date])
        converted_leads = len([l for l in leads if l.status == "converted"])
        
        metrics = {
            "total_leads": total_leads,
            "engaged_leads": engaged_leads,
            "converted_leads": converted_leads,
            "total_interactions": len(interactions),
            "conversion_rate": converted_leads / max(1, total_leads),
            "engagement_rate": engaged_leads / max(1, total_leads),
            "response_rate": 0.0,  # Calculate from interactions
            "avg_lead_score": 0.0,
            "cost_per_lead": 0.0,
            "roi": 0.0
        }
        
        # Calculate response rate
        outbound = [i for i in interactions if i.direction == "outbound"]
        inbound = [i for i in interactions if i.direction == "inbound"]
        if outbound:
            metrics["response_rate"] = len(inbound) / len(outbound)
        
        # Calculate average lead score
        if leads:
            total_score = sum(l.lead_score or 0 for l in leads)
            metrics["avg_lead_score"] = total_score / len(leads)
        
        return metrics
    
    async def _analyze_metric_trends(
        self,
        campaign_id: str,
        time_period: int
    ) -> Dict:
        """Analyze trends in metrics over time."""
        # Split time period into weekly buckets
        weeks = time_period // 7
        trends = {}
        
        for metric in [MetricType.CONVERSION_RATE, MetricType.ENGAGEMENT_RATE]:
            weekly_values = []
            for week in range(weeks):
                week_metrics = await self._calculate_campaign_metrics(
                    campaign_id, 7
                )
                weekly_values.append(week_metrics.get(metric.value, 0))
            
            # Calculate trend direction
            if len(weekly_values) >= 2:
                trend_direction = "increasing" if weekly_values[-1] > weekly_values[0] else "decreasing"
                trend_strength = abs(weekly_values[-1] - weekly_values[0]) / max(0.01, weekly_values[0])
            else:
                trend_direction = "stable"
                trend_strength = 0.0
            
            trends[metric.value] = {
                "direction": trend_direction,
                "strength": trend_strength,
                "weekly_values": weekly_values
            }
        
        return trends
    
    def _compare_to_benchmarks(self, metrics: Dict) -> Dict:
        """Compare metrics to industry benchmarks."""
        benchmarks = {}
        
        for metric_name, metric_value in metrics.items():
            if metric_name in [m.value for m in MetricType]:
                metric_type = MetricType(metric_name)
                if metric_type in self.performance_thresholds:
                    thresholds = self.performance_thresholds[metric_type]
                    
                    if metric_value >= thresholds["excellent"]:
                        rating = "excellent"
                    elif metric_value >= thresholds["good"]:
                        rating = "good"
                    elif metric_value >= thresholds["fair"]:
                        rating = "fair"
                    else:
                        rating = "poor"
                    
                    benchmarks[metric_name] = {
                        "value": metric_value,
                        "rating": rating,
                        "thresholds": thresholds
                    }
        
        return benchmarks
    
    async def _identify_issues(
        self,
        campaign_id: str,
        metrics: Dict
    ) -> List[Dict]:
        """Identify performance issues."""
        issues = []
        
        # Low conversion rate
        if metrics.get("conversion_rate", 0) < 0.05:
            issues.append({
                "type": "low_conversion_rate",
                "severity": "high",
                "description": "Conversion rate below acceptable threshold",
                "metric_value": metrics["conversion_rate"]
            })
        
        # Low engagement
        if metrics.get("engagement_rate", 0) < 0.15:
            issues.append({
                "type": "low_engagement",
                "severity": "medium",
                "description": "Engagement rate indicates poor audience targeting",
                "metric_value": metrics["engagement_rate"]
            })
        
        # Poor response rate
        if metrics.get("response_rate", 0) < 0.10:
            issues.append({
                "type": "poor_response_rate",
                "severity": "medium",
                "description": "Low response rate suggests messaging issues",
                "metric_value": metrics["response_rate"]
            })
        
        return issues
    
    async def _identify_opportunities(
        self,
        campaign_id: str,
        metrics: Dict
    ) -> List[Dict]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # High engagement, low conversion - optimize funnel
        if (metrics.get("engagement_rate", 0) > 0.30 and 
            metrics.get("conversion_rate", 0) < 0.10):
            opportunities.append({
                "type": "funnel_optimization",
                "potential": "high",
                "description": "High engagement but low conversion - optimize conversion funnel"
            })
        
        # Good performance - scale opportunity
        if metrics.get("conversion_rate", 0) > 0.15:
            opportunities.append({
                "type": "scale_campaign",
                "potential": "high",
                "description": "Strong performance indicates scaling opportunity"
            })
        
        return opportunities
    
    async def _generate_recommendations(
        self,
        campaign_id: str,
        metrics: Dict,
        trends: Dict,
        issues: List[Dict],
        opportunities: List[Dict]
    ) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Address issues
        for issue in issues:
            if issue["type"] == "low_conversion_rate":
                recommendations.append({
                    "priority": "high",
                    "action": "refine_targeting",
                    "description": "Narrow audience targeting to higher-quality leads",
                    "expected_impact": "+30% conversion rate"
                })
            elif issue["type"] == "low_engagement":
                recommendations.append({
                    "priority": "high",
                    "action": "test_new_messaging",
                    "description": "A/B test new message variants to improve engagement",
                    "expected_impact": "+20% engagement rate"
                })
        
        # Leverage opportunities
        for opportunity in opportunities:
            if opportunity["type"] == "scale_campaign":
                recommendations.append({
                    "priority": "medium",
                    "action": "increase_budget",
                    "description": "Increase budget by 50% to scale successful campaign",
                    "expected_impact": "+50% conversions"
                })
        
        return recommendations
    
    def _calculate_campaign_health_score(self, metrics: Dict) -> float:
        """Calculate overall campaign health score (0-100)."""
        score = 0.0
        weights = {
            "conversion_rate": 0.40,
            "engagement_rate": 0.30,
            "response_rate": 0.20,
            "avg_lead_score": 0.10
        }
        
        for metric, weight in weights.items():
            metric_value = metrics.get(metric, 0)
            
            # Normalize to 0-100 scale
            if metric == "conversion_rate":
                normalized = min(100, metric_value * 1000)
            elif metric == "engagement_rate":
                normalized = min(100, metric_value * 250)
            elif metric == "response_rate":
                normalized = min(100, metric_value * 333)
            else:
                normalized = metric_value  # Already 0-100
            
            score += normalized * weight
        
        return round(score, 1)
    
    async def _determine_optimization_actions(
        self,
        campaign_id: str,
        goal: OptimizationGoal,
        analysis: Dict,
        constraints: Optional[Dict]
    ) -> List[Dict]:
        """Determine specific optimization actions."""
        actions = []
        
        if goal == OptimizationGoal.MAXIMIZE_CONVERSIONS:
            actions.append({
                "type": "refine_targeting",
                "parameters": {"min_lead_score": 70}
            })
            actions.append({
                "type": "optimize_messaging",
                "parameters": {"focus": "conversion_cta"}
            })
        
        elif goal == OptimizationGoal.MAXIMIZE_ENGAGEMENT:
            actions.append({
                "type": "test_send_times",
                "parameters": {"variants": ["morning", "afternoon", "evening"]}
            })
            actions.append({
                "type": "personalize_content",
                "parameters": {"level": "high"}
            })
        
        return actions
    
    async def _simulate_optimization_impact(
        self,
        campaign_id: str,
        actions: List[Dict],
        analysis: Dict
    ) -> Dict:
        """Simulate expected impact of optimization actions."""
        current_metrics = analysis["metrics"]
        
        # Simple impact estimation (would use ML model in production)
        impact = {
            "conversion_rate_change": "+15%",
            "engagement_rate_change": "+10%",
            "estimated_additional_conversions": 25,
            "confidence": "medium"
        }
        
        return impact
    
    async def _execute_optimization_actions(
        self,
        campaign_id: str,
        actions: List[Dict]
    ) -> List[Dict]:
        """Execute optimization actions."""
        results = []
        
        for action in actions:
            # Execute action (placeholder)
            result = {
                "action": action["type"],
                "status": "executed",
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
        
        return results
    
    async def _get_ab_test_data(self, test_id: str) -> Dict:
        """Fetch A/B test data."""
        # Placeholder
        return {
            "name": "Test Campaign",
            "variant_a": {"sample_size": 150, "conversions": 15},
            "variant_b": {"sample_size": 150, "conversions": 22}
        }
    
    async def _calculate_variant_metrics(self, variant_data: Dict) -> Dict:
        """Calculate metrics for a test variant."""
        sample_size = variant_data["sample_size"]
        conversions = variant_data["conversions"]
        
        return {
            "sample_size": sample_size,
            "conversions": conversions,
            "conversion_rate": conversions / max(1, sample_size)
        }
    
    def _calculate_statistical_significance(
        self,
        variant_a: Dict,
        variant_b: Dict
    ) -> Dict:
        """Calculate statistical significance using z-test."""
        # Simplified z-test calculation
        p_a = variant_a["conversion_rate"]
        p_b = variant_b["conversion_rate"]
        n_a = variant_a["sample_size"]
        n_b = variant_b["sample_size"]
        
        # Pooled proportion
        p_pool = (variant_a["conversions"] + variant_b["conversions"]) / (n_a + n_b)
        
        # Standard error
        se = (p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) ** 0.5
        
        # Z-score
        z_score = abs(p_a - p_b) / se if se > 0 else 0
        
        # P-value approximation
        p_value = 0.05 if z_score > 1.96 else 0.20
        
        return {
            "z_score": round(z_score, 3),
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "confidence_level": "95%" if p_value < 0.05 else "80%"
        }
    
    def _determine_test_winner(
        self,
        variant_a: Dict,
        variant_b: Dict,
        significance: Dict
    ) -> str:
        """Determine test winner."""
        if not significance["is_significant"]:
            return "inconclusive"
        
        return "variant_b" if variant_b["conversion_rate"] > variant_a["conversion_rate"] else "variant_a"
    
    def _calculate_lift(
        self,
        variant_a: Dict,
        variant_b: Dict
    ) -> Dict:
        """Calculate lift between variants."""
        baseline = variant_a["conversion_rate"]
        treatment = variant_b["conversion_rate"]
        
        if baseline > 0:
            lift_pct = ((treatment - baseline) / baseline) * 100
        else:
            lift_pct = 0
        
        return {
            "percentage": round(lift_pct, 2),
            "absolute": round(treatment - baseline, 4)
        }
    
    def _generate_test_insights(
        self,
        test_data: Dict,
        variant_a: Dict,
        variant_b: Dict,
        winner: str,
        lift: Dict
    ) -> List[str]:
        """Generate insights from test results."""
        insights = []
        
        if winner != "inconclusive":
            insights.append(
                f"Variant B showed {lift['percentage']}% improvement over Variant A"
            )
        else:
            insights.append("No statistically significant difference between variants")
        
        return insights
    
    def _generate_test_recommendation(
        self,
        winner: str,
        significance: Dict,
        lift: Dict
    ) -> str:
        """Generate recommendation from test."""
        if winner == "inconclusive":
            return "Continue test to gather more data"
        elif significance["is_significant"] and lift["percentage"] > 10:
            return f"Implement {winner} across all campaigns"
        else:
            return "Consider additional testing with larger sample size"
    
    def _predict_reach(self, features: Dict) -> int:
        """Predict campaign reach."""
        # Simplified prediction
        base_reach = features.get("audience_size", 1000)
        return int(base_reach * 0.7)
    
    def _predict_engagement_rate(self, features: Dict) -> float:
        """Predict engagement rate."""
        # Simplified prediction
        return 0.25
    
    def _predict_conversion_rate(self, features: Dict) -> float:
        """Predict conversion rate."""
        # Simplified prediction
        return 0.08
    
    def _predict_cost_per_lead(self, features: Dict) -> float:
        """Predict cost per lead."""
        # Simplified prediction
        return 25.0
    
    def _calculate_expected_roi(
        self,
        campaign_config: Dict,
        predictions: Dict
    ) -> float:
        """Calculate expected ROI."""
        budget = campaign_config.get("budget", 1000)
        avg_deal_value = campaign_config.get("avg_deal_value", 500)
        
        expected_conversions = predictions["expected_reach"] * predictions["expected_conversion_rate"]
        expected_revenue = expected_conversions * avg_deal_value
        roi = ((expected_revenue - budget) / budget) * 100
        
        return round(roi, 2)
    
    def _assess_campaign_risk(
        self,
        campaign_config: Dict,
        predictions: Dict
    ) -> Dict:
        """Assess campaign risk."""
        risk_factors = []
        overall_risk = "low"
        
        if predictions["expected_conversion_rate"] < 0.05:
            risk_factors.append("Low predicted conversion rate")
            overall_risk = "high"
        
        if predictions["expected_roi"] < 50:
            risk_factors.append("Low expected ROI")
            overall_risk = "medium" if overall_risk == "low" else "high"
        
        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors
        }
    
    def _generate_pre_launch_recommendations(
        self,
        campaign_config: Dict,
        predictions: Dict,
        risk_assessment: Dict
    ) -> List[str]:
        """Generate pre-launch recommendations."""
        recommendations = []
        
        if risk_assessment["overall_risk"] == "high":
            recommendations.append("Consider testing with smaller budget first")
        
        if predictions["expected_engagement_rate"] < 0.20:
            recommendations.append("Refine messaging before launch")
        
        return recommendations
    
    async def _get_campaign(self, campaign_id: str):
        """Fetch campaign from database."""
        # Placeholder
        pass
    
    async def _get_campaign_leads(
        self,
        campaign_id: str,
        since: Optional[datetime] = None
    ) -> List[Lead]:
        """Fetch campaign leads."""
        # Placeholder
        return []
    
    async def _get_campaign_interactions(
        self,
        campaign_id: str,
        since: Optional[datetime] = None
    ) -> List[Interaction]:
        """Fetch campaign interactions."""
        # Placeholder
        return []
    
    async def _get_historical_benchmarks(
        self,
        campaign_type: str,
        audience: str
    ) -> Dict:
        """Get historical benchmark data."""
        # Placeholder
        return {}
    
    def _extract_prediction_features(
        self,
        campaign_config: Dict,
        historical_data: Dict
    ) -> Dict:
        """Extract features for prediction model."""
        return {
            "audience_size": campaign_config.get("audience_size", 1000),
            "campaign_type": campaign_config.get("campaign_type", "email"),
            "budget": campaign_config.get("budget", 1000)
        }
    
    async def _perform_segmentation(
        self,
        leads: List[Lead],
        criteria: List[str]
    ) -> Dict[str, List[Lead]]:
        """Perform audience segmentation."""
        segments = defaultdict(list)
        
        for lead in leads:
            # Simple segmentation by lead score
            if lead.lead_score >= 80:
                segments["high_value"].append(lead)
            elif lead.lead_score >= 50:
                segments["medium_value"].append(lead)
            else:
                segments["low_value"].append(lead)
        
        return dict(segments)
    
    async def _analyze_segment(
        self,
        segment_name: str,
        leads: List[Lead],
        campaign_id: str
    ) -> Dict:
        """Analyze a specific segment."""
        return {
            "segment_name": segment_name,
            "lead_count": len(leads),
            "avg_lead_score": sum(l.lead_score or 0 for l in leads) / max(1, len(leads)),
            "performance_score": 75.0  # Simplified
        }
    
    def _generate_segment_recommendations(
        self,
        segment_analyses: Dict
    ) -> List[str]:
        """Generate segment-specific recommendations."""
        recommendations = []
        
        for segment_name, analysis in segment_analyses.items():
            if analysis["performance_score"] < 50:
                recommendations.append(
                    f"Consider excluding {segment_name} segment from future campaigns"
                )
        
        return recommendations
    
    def _calculate_efficiency_score(self, metrics: Dict) -> float:
        """Calculate campaign efficiency score."""
        conversion_rate = metrics.get("conversion_rate", 0)
        engagement_rate = metrics.get("engagement_rate", 0)
        
        return (conversion_rate * 0.6 + engagement_rate * 0.4) * 100
    
    def _optimize_allocation(
        self,
        efficiency_scores: Dict,
        total_budget: float
    ) -> Dict:
        """Optimize budget allocation."""
        total_score = sum(efficiency_scores.values())
        
        allocations = {}
        for campaign_id, score in efficiency_scores.items():
            weight = score / total_score if total_score > 0 else 1 / len(efficiency_scores)
            allocations[campaign_id] = round(total_budget * weight, 2)
        
        return allocations
    
    def _calculate_allocation_impact(
        self,
        allocations: Dict,
        performances: Dict
    ) -> Dict:
        """Calculate expected impact of allocation."""
        return {
            "total_budget": sum(allocations.values()),
            "expected_conversions": 150,  # Simplified
            "expected_roi": 250.0
        }
    
    async def health_check(self) -> Dict:
        """Perform health check for Campaign Optimization Agent."""
        base_health = await super().health_check()
        
        optimization_health = {
            "cached_analyses": len(self.analysis_cache),
            "optimization_strategies": len(self.optimization_strategies),
            "performance_thresholds": len(self.performance_thresholds),
            "capabilities": [
                "performance_analysis",
                "ab_testing",
                "budget_optimization",
                "audience_segmentation",
                "predictive_modeling"
            ]
        }
        
        base_health.update(optimization_health)
        return base_health