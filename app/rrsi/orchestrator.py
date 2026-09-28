# app/rrsi/orchestrator.py
"""
RRSI Unified Orchestrator.
Main entry point coordinating:
- Proposal validation (Negative Knowledge & Area Stagnation)
- Selection & Leakage Screening
- Statistical & Cost-Aware Evaluation
- Fine-grained Error Taxonomy Tracking
- Ledger Persistence & Pruning Suggestions
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from app.rrsi.taxonomy import (
    ProposalHypothesis,
    PipelineArea,
    EvaluationMetrics,
    DecisionRecord,
    ExperimentStatus
)
from app.rrsi.memory import NegativeKnowledgeRegistry
from app.rrsi.screening import LeakageScreening, PruningAnalyzer
from app.rrsi.evaluator import StatisticalEvaluator
from app.rrsi.ledger import ExperimentLedger


class RRSIOrchestrator:
    """
    Coordinates the Reflective Recursive Self-Improvement (RRSI) lifecycle.
    """

    def __init__(
        self,
        memory_path: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
        stagnation_threshold: int = 3,
        cost_lambda: float = 0.05,
        tolerance: float = 1.0
    ):
        self.memory = NegativeKnowledgeRegistry(storage_path=memory_path, stagnation_threshold=stagnation_threshold)
        self.ledger = ExperimentLedger(ledger_path=ledger_path)
        self.evaluator = StatisticalEvaluator(cost_lambda=cost_lambda, tolerance=tolerance)

    def submit_proposal(self, hypothesis: ProposalHypothesis) -> Tuple[bool, List[str], Optional[PipelineArea]]:
        """
        Step 1 (Proposal Side):
        Evaluates the hypothesis against negative memory to ensure:
        - It does not repeat previously proven ineffective ideas.
        - It is not stuck micro-tweaking a stagnant component.
        """
        is_viable, warnings, rec_area = self.memory.check_proposal_viability(hypothesis)
        return is_viable, warnings, rec_area

    def screen_code_changes(self, code_diff: str) -> Tuple[bool, List[str]]:
        """
        Step 2 (Selection Side - Pre-Flight):
        Audits candidate code for hardcoded tokens, vendor memorization, or static coordinates.
        """
        return LeakageScreening.audit_code_changes(code_diff)

    def evaluate_and_decide(
        self,
        hypothesis: ProposalHypothesis,
        baseline_dev: EvaluationMetrics,
        candidate_dev: EvaluationMetrics,
        baseline_selection: Optional[EvaluationMetrics] = None,
        candidate_selection: Optional[EvaluationMetrics] = None,
        template_groups: Optional[Dict[str, str]] = None,
        git_base_ref: Optional[str] = None
    ) -> DecisionRecord:
        """
        Step 3 & 4 (Selection Side - Statistical & Cost-Aware Decision):
        Compares baseline vs candidate, applies paired bootstrap CI, cost utility,
        and logs error taxonomy deltas.
        """
        decision = self.evaluator.decide_acceptance(
            hypothesis=hypothesis,
            baseline_dev=baseline_dev,
            candidate_dev=candidate_dev,
            baseline_selection=baseline_selection,
            candidate_selection=candidate_selection,
            template_groups=template_groups,
            git_base_ref=git_base_ref
        )

        # Record in memory and persistent ledger
        self.memory.record_decision(hypothesis, decision)
        self.ledger.append(decision)

        return decision

    def get_stagnation_report(self) -> Dict[str, Any]:
        """Provides a breakdown of consecutive failures per pipeline area to guide exploration."""
        report = {}
        for area in PipelineArea:
            failures = self.memory.get_consecutive_failures_in_area(area)
            report[area.value] = {
                "consecutive_failures": failures,
                "is_stagnant": failures >= self.memory.stagnation_threshold,
                "recommendation": "Shift to alternative area" if failures >= self.memory.stagnation_threshold else "Active"
            }
        return report

    def format_decision_report(self, decision: DecisionRecord) -> str:
        """Generates auditable Markdown summary of the decision."""
        return self.ledger.generate_markdown_summary(decision)
