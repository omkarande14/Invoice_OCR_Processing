# app/rrsi/__init__.py
"""
Reflective Recursive Self-Improvement (RRSI) Package for Invoice OCR.
Provides:
- Error Taxonomy (ErrorCategory, PipelineArea, FailureDetail)
- Proposal Memory & Negative Knowledge Registry (NegativeKnowledgeRegistry)
- Selection & Leakage Screening (LeakageScreening, PruningAnalyzer)
- Statistical & Cost-Aware Evaluator (StatisticalEvaluator)
- Auditable Experiment Ledger (ExperimentLedger)
- Unified Orchestrator (RRSIOrchestrator)
"""

from app.rrsi.taxonomy import (
    ErrorCategory,
    PipelineArea,
    ExperimentStatus,
    FailureDetail,
    ProposalHypothesis,
    EvaluationMetrics,
    DecisionRecord
)
from app.rrsi.memory import NegativeKnowledgeRegistry
from app.rrsi.screening import LeakageScreening, PruningAnalyzer
from app.rrsi.evaluator import StatisticalEvaluator
from app.rrsi.ledger import ExperimentLedger
from app.rrsi.orchestrator import RRSIOrchestrator

__all__ = [
    "ErrorCategory",
    "PipelineArea",
    "ExperimentStatus",
    "FailureDetail",
    "ProposalHypothesis",
    "EvaluationMetrics",
    "DecisionRecord",
    "NegativeKnowledgeRegistry",
    "LeakageScreening",
    "PruningAnalyzer",
    "StatisticalEvaluator",
    "ExperimentLedger",
    "RRSIOrchestrator"
]
