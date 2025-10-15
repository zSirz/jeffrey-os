import asyncio
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from .models import Decision, EventSource, Proposal, ProposalStatus, ProposalType, RiskLevel, VerdictType


class ProposalManager:
    """Manages the lifecycle of proposals from creation to implementation"""

    def __init__(self, event_signer=None, alert_chainer=None):
        self.proposals: dict[str, Proposal] = {}
        self.pending_queue: list[str] = []
        self.event_signer = event_signer
        self.alert_chainer = alert_chainer
        self.impact_weights = {
            "performance": 0.3,
            "security": 0.4,
            "user_experience": 0.2,
            "maintainability": 0.1,
        }
        self.risk_thresholds = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0,
        }

    def create_proposal(
        self,
        proposal_type: ProposalType,
        description: str,
        detailed_plan: str,
        sources: list[EventSource],
        metadata: dict[str, Any] | None = None,
    ) -> Proposal:
        """Create a new proposal with automatic impact and risk assessment"""
        proposal = Proposal(
            type=proposal_type,
            description=description,
            detailed_plan=detailed_plan,
            sources=sources,
            metadata=metadata or {},
        )

        # Calculate impact score based on sources
        proposal.impact_score = self._calculate_impact_score(proposal)

        # Evaluate risk level
        proposal.risk_level = self._evaluate_risk_level(proposal)

        # Sign the proposal if signer available
        if self.event_signer:
            proposal.signature = self._sign_proposal(proposal)

        # Store proposal
        self.proposals[proposal.id] = proposal

        return proposal

    def _calculate_impact_score(self, proposal: Proposal) -> float:
        """Calculate impact score based on proposal type and sources"""
        base_scores = {
            ProposalType.OPTIMIZATION: 0.6,
            ProposalType.SECURITY: 0.9,
            ProposalType.FEATURE: 0.5,
            ProposalType.BUGFIX: 0.8,
        }

        score = base_scores.get(proposal.type, 0.5)

        # Adjust based on event sources
        if proposal.sources:
            # More events = higher impact
            event_multiplier = min(1.5, 1 + (len(proposal.sources) * 0.1))
            score *= event_multiplier

            # Check for critical events
            for source in proposal.sources:
                if "critical" in source.description.lower():
                    score *= 1.2
                if "security" in source.event_type.lower():
                    score *= 1.15

        # Extract metrics from sources for refined scoring
        all_metrics = {}
        for source in proposal.sources:
            all_metrics.update(source.metrics)

        # Adjust based on specific metrics
        if "performance_degradation" in all_metrics:
            degradation = all_metrics["performance_degradation"]
            if degradation > 0.3:  # 30% degradation
                score *= 1.3

        if "affected_users" in all_metrics:
            users = all_metrics["affected_users"]
            if users > 1000:
                score *= 1.2
            elif users > 10000:
                score *= 1.5

        return min(1.0, score)  # Cap at 1.0

    def _evaluate_risk_level(self, proposal: Proposal) -> RiskLevel:
        """Evaluate risk level based on proposal characteristics"""
        risk_score = 0.0

        # Base risk by type
        type_risks = {
            ProposalType.OPTIMIZATION: 0.3,
            ProposalType.SECURITY: 0.2,  # Security fixes are usually low risk
            ProposalType.FEATURE: 0.5,
            ProposalType.BUGFIX: 0.1,
        }

        risk_score = type_risks.get(proposal.type, 0.5)

        # Adjust based on complexity indicators in plan
        if proposal.detailed_plan:
            plan_lower = proposal.detailed_plan.lower()

            # High risk keywords
            high_risk_keywords = [
                "database",
                "migration",
                "refactor",
                "breaking change",
                "api change",
                "dependency update",
                "algorithm change",
            ]

            for keyword in high_risk_keywords:
                if keyword in plan_lower:
                    risk_score += 0.15

            # Low risk keywords
            low_risk_keywords = [
                "logging",
                "monitoring",
                "documentation",
                "test",
                "comment",
                "typo",
                "formatting",
            ]

            for keyword in low_risk_keywords:
                if keyword in plan_lower:
                    risk_score -= 0.1

        # Consider impact score - high impact changes are riskier
        risk_score += proposal.impact_score * 0.2

        # Normalize
        risk_score = max(0.0, min(1.0, risk_score))

        # Map to risk level
        if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def submit_for_review(self, proposal_id: str) -> bool:
        """Submit proposal for human review"""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        # Validate proposal completeness
        if not self._validate_proposal(proposal):
            return False

        # Add to review queue if not already there
        if proposal_id not in self.pending_queue:
            self.pending_queue.append(proposal_id)

        # Alert if urgent (high impact + high risk)
        if self._is_urgent(proposal) and self.alert_chainer:
            asyncio.create_task(self._send_urgent_alert(proposal))

        return True

    def _validate_proposal(self, proposal: Proposal) -> bool:
        """Validate proposal has all required information"""
        required_fields = [
            proposal.description,
            proposal.type,
            proposal.impact_score > 0,
            proposal.risk_level is not None,
            len(proposal.sources) > 0,
        ]

        return all(required_fields)

    def _is_urgent(self, proposal: Proposal) -> bool:
        """Determine if proposal needs urgent attention"""
        return (proposal.impact_score > 0.8 and proposal.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]) or (
            proposal.type == ProposalType.SECURITY and proposal.impact_score > 0.6
        )

    async def _send_urgent_alert(self, proposal: Proposal):
        """Send urgent alert through alert chainer"""
        if not self.alert_chainer:
            return

        alert_message = {
            "type": "urgent_proposal",
            "proposal_id": proposal.id,
            "description": proposal.description,
            "impact": proposal.impact_score,
            "risk": proposal.risk_level.value,
            "sources_count": len(proposal.sources),
        }

        await self.alert_chainer.send_alert(
            level="critical",
            message=f"Urgent proposal requires review: {proposal.description}",
            data=alert_message,
        )

    def record_decision(
        self,
        proposal_id: str,
        verdict: VerdictType,
        rationale: str,
        review_time_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Decision:
        """Record human decision on a proposal"""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        # Create decision record
        decision = Decision(
            proposal_id=proposal_id,
            verdict=verdict,
            rationale=rationale,
            review_time_seconds=review_time_seconds,
            metadata=metadata or {},
        )

        # Update proposal
        proposal.human_verdict = verdict
        proposal.rationale = rationale
        proposal.decided_at = datetime.now()

        # Update status based on verdict
        if verdict == VerdictType.ACCEPT:
            proposal.status = ProposalStatus.ACCEPTED
        elif verdict == VerdictType.REJECT:
            proposal.status = ProposalStatus.REJECTED
        elif verdict == VerdictType.DEFER:
            proposal.status = ProposalStatus.DEFERRED

        # Remove from pending queue
        if proposal_id in self.pending_queue:
            self.pending_queue.remove(proposal_id)

        # Sign decision if signer available
        if self.event_signer:
            decision.signature = self._sign_decision(decision)

        # Trigger post-decision actions
        asyncio.create_task(self._handle_post_decision(proposal, decision))

        return decision

    async def _handle_post_decision(self, proposal: Proposal, decision: Decision):
        """Handle actions after a decision is made"""
        if decision.verdict == VerdictType.ACCEPT:
            # Queue for implementation
            proposal.metadata["queued_for_implementation"] = datetime.now().isoformat()

            # Alert implementation system if available
            if self.alert_chainer:
                await self.alert_chainer.send_alert(
                    level="info",
                    message=f"Proposal accepted for implementation: {proposal.description}",
                    data={
                        "proposal_id": proposal.id,
                        "type": proposal.type.value,
                        "impact": proposal.impact_score,
                    },
                )

    def get_pending_proposals(
        self,
        filters: dict[str, Any] | None = None,
        sort_by: str = "created_at",
        limit: int | None = None,
    ) -> list[Proposal]:
        """Get pending proposals with optional filtering"""
        pending = [self.proposals[pid] for pid in self.pending_queue if pid in self.proposals]

        # Apply filters
        if filters:
            if "type" in filters:
                pending = [p for p in pending if p.type == filters["type"]]
            if "risk_level" in filters:
                pending = [p for p in pending if p.risk_level == filters["risk_level"]]
            if "min_impact" in filters:
                pending = [p for p in pending if p.impact_score >= filters["min_impact"]]

        # Sort
        if sort_by == "impact":
            pending.sort(key=lambda p: p.impact_score, reverse=True)
        elif sort_by == "risk":
            risk_order = {
                RiskLevel.CRITICAL: 4,
                RiskLevel.HIGH: 3,
                RiskLevel.MEDIUM: 2,
                RiskLevel.LOW: 1,
            }
            pending.sort(key=lambda p: risk_order[p.risk_level], reverse=True)
        else:  # Default to created_at
            pending.sort(key=lambda p: p.created_at)

        # Apply limit
        if limit:
            pending = pending[:limit]

        return pending

    def get_proposal_by_id(self, proposal_id: str) -> Proposal | None:
        """Get a specific proposal by ID"""
        return self.proposals.get(proposal_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get proposal statistics"""
        stats = {
            "total_proposals": len(self.proposals),
            "pending": len(self.pending_queue),
            "by_status": defaultdict(int),
            "by_type": defaultdict(int),
            "by_risk": defaultdict(int),
            "average_impact": 0.0,
            "urgent_count": 0,
        }

        if self.proposals:
            total_impact = 0.0

            for proposal in self.proposals.values():
                stats["by_status"][proposal.status.value] += 1
                stats["by_type"][proposal.type.value] += 1
                stats["by_risk"][proposal.risk_level.value] += 1
                total_impact += proposal.impact_score

                if self._is_urgent(proposal):
                    stats["urgent_count"] += 1

            stats["average_impact"] = total_impact / len(self.proposals)

        return dict(stats)

    def _sign_proposal(self, proposal: Proposal) -> str:
        """Sign proposal data"""
        if not self.event_signer:
            return ""

        data_to_sign = json.dumps(
            {
                "id": proposal.id,
                "type": proposal.type.value,
                "description": proposal.description,
                "impact_score": proposal.impact_score,
                "risk_level": proposal.risk_level.value,
                "created_at": proposal.created_at.isoformat(),
            },
            sort_keys=True,
        )

        return self.event_signer.sign(data_to_sign)

    def _sign_decision(self, decision: Decision) -> str:
        """Sign decision data"""
        if not self.event_signer:
            return ""

        data_to_sign = json.dumps(
            {
                "proposal_id": decision.proposal_id,
                "verdict": decision.verdict.value,
                "rationale": decision.rationale,
                "timestamp": decision.timestamp.isoformat(),
                "decided_by": decision.decided_by,
            },
            sort_keys=True,
        )

        return self.event_signer.sign(data_to_sign)

    def mark_implemented(self, proposal_id: str, implementation_time: float):
        """Mark a proposal as implemented"""
        if proposal_id not in self.proposals:
            return

        proposal = self.proposals[proposal_id]
        proposal.status = ProposalStatus.IMPLEMENTED
        proposal.implemented_at = datetime.now()
        proposal.implementation_time = implementation_time

    def mark_failed(self, proposal_id: str, reason: str):
        """Mark a proposal implementation as failed"""
        if proposal_id not in self.proposals:
            return

        proposal = self.proposals[proposal_id]
        proposal.status = ProposalStatus.FAILED
        proposal.metadata["failure_reason"] = reason
        proposal.metadata["failed_at"] = datetime.now().isoformat()
