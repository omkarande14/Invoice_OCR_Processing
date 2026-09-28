# app/rrsi/taxonomy.py
"""
RRSI Error Taxonomy and Data Structures.
1. Decoupled Extraction Error Classes (invoice_number_*, date_*, table_*, etc.) from Process Outcomes.
2. Non-double-counting FailureDetail with primary_class + secondary_tags.
3. FieldMetrics with Precision, Recall, F1, Accuracy, and Wilson Score Intervals.
4. Per-document outcome representation for paired statistical testing.
5. Selection Gate decision records and proposal hypotheses.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import time


# =============================================================================
# 1. Extraction Error Classes (Pipeline Performance Taxonomy)
# =============================================================================
class ExtractionErrorClass(str, Enum):
    """Fine-grained error classes for OCR extraction discrepancies."""
    # Invoice Number
    INVOICE_NUMBER_MISSING = "invoice_number_missing"
    INVOICE_NUMBER_MISMATCH = "invoice_number_mismatch"

    # Dates
    DATE_MISSING = "date_missing"
    DATE_MISMATCH = "date_mismatch"
    DATE_LOCALE_SWAP = "date_locale_swap"

    # Table Structure & Cells
    TABLE_ROW_MERGE = "table_row_merge"
    TABLE_ROW_SPLIT = "table_row_split"
    TABLE_COLUMN_SHIFT = "table_column_shift"
    TABLE_MISSING_ROW = "table_missing_row"
    TABLE_EXTRA_ROW = "table_extra_row"
    TABLE_CELL_MISMATCH = "table_cell_mismatch"

    # Currency & Amounts
    AMOUNT_PARSE = "amount_parse"
    CURRENCY_MISSING = "currency_missing"

    # Parties & Legal Identifiers
    GSTIN_INVALID = "gstin_invalid"
    GSTIN_MISMATCH = "gstin_mismatch"
    PARTY_SWAP = "party_swap"
    PARTY_NAME_MISMATCH = "party_name_mismatch"

    # OCR Engine & Arithmetic
    OCR_NOISE = "ocr_noise"
    TOTALS_ARITHMETIC = "totals_arithmetic"
    TOTALS_ARITHMETIC_ROUNDOFF = "totals_arithmetic_roundoff"
    DOCUMENT_INCONSISTENT = "document_inconsistent"

    # System & Pipeline
    HALLUCINATION = "hallucination"
    PIPELINE_ERROR = "pipeline_error"
    OTHER = "other"


# Backwards compatibility alias for existing code
ErrorCategory = ExtractionErrorClass


# =============================================================================
# 2. Process Outcomes (Selection Gate Decision Taxonomy - Never mixed with extraction)
# =============================================================================
class ProcessOutcome(str, Enum):
    """Decision outcomes produced by the selection gate."""
    ACCEPTED = "accepted"
    LEAKAGE = "leakage"
    INSIGNIFICANT = "insignificant"
    COST_REJECTED = "cost_rejected"
    REGRESSION_DETECTED = "regression_detected"
    GENERALIZATION_FAILURE = "generalization_failure"
    NO_CHANGE = "no_change"
    PRUNED = "pruned"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PipelineArea(str, Enum):
    """Subsystem areas of the Invoice OCR pipeline for targeted experimentation and area shifting."""
    INPUT_PREPROCESSING = "input_preprocessing"
    OCR_ENGINE = "ocr_engine"
    ENTITY_EXTRACTION = "entity_extraction"
    TABLE_EXTRACTION = "table_extraction"
    ARITHMETIC_VERIFIER = "arithmetic_verifier"
    DUAL_ENGINE_CONSENSUS = "dual_engine_consensus"
    SCHEMA_MAPPING = "schema_mapping"


class ExperimentStatus(str, Enum):
    """Status of an RRSI proposal or experiment."""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    PRUNED = "pruned"


# =============================================================================
# 3. Data Structures for Failures & Field Metrics
# =============================================================================
@dataclass
class FailureDetail:
    """
    Individual failure record.
    Enforces no double-counting: one primary_class, plus optional secondary_tags.
    """
    primary_class: ExtractionErrorClass
    field_name: str
    expected: Any
    actual: Any
    file_id: str
    severity: str = "medium"  # "critical", "high", "medium", "low"
    secondary_tags: List[str] = field(default_factory=list)
    context_message: str = ""

    @property
    def category(self) -> ExtractionErrorClass:
        """Alias for backward compatibility."""
        return self.primary_class

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_class": self.primary_class.value,
            "category": self.primary_class.value,
            "field_name": self.field_name,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "file_id": self.file_id,
            "severity": self.severity,
            "secondary_tags": self.secondary_tags,
            "context_message": self.context_message
        }


@dataclass
class FieldMetrics:
    """Detailed statistical metrics for a single extracted field."""
    field_name: str
    total_expected: int = 0
    correct: int = 0
    hallucinations: int = 0
    precision: float = 1.0
    recall: float = 1.0
    f1: float = 1.0
    accuracy: float = 1.0
    wilson_ci: Tuple[float, float] = (0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentEvaluationResult:
    """Per-document outcomes required for paired bootstrap and McNemar testing."""
    doc_id: str
    field_outcomes: Dict[str, bool] = field(default_factory=dict)
    field_errors: Dict[str, Optional[str]] = field(default_factory=dict)
    table_result: Optional[Any] = None
    validator_result: Optional[Any] = None
    arithmetic_status: str = "consistent"
    is_end_to_end_correct: bool = True
    failures: List[FailureDetail] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "field_outcomes": self.field_outcomes,
            "field_errors": self.field_errors,
            "arithmetic_status": self.arithmetic_status,
            "is_end_to_end_correct": self.is_end_to_end_correct,
            "failures": [f.to_dict() for f in self.failures]
        }


@dataclass
class ProposalHypothesis:
    """Candidate change submitted to the RRSI proposal engine."""
    proposal_id: str
    area: PipelineArea
    title: str
    description: str
    rationale: str
    code_diff_summary: str = ""
    tags: List[str] = field(default_factory=list)
    provenance: str = "real"  # "real" or "demo"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "area": self.area.value,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "code_diff_summary": self.code_diff_summary,
            "tags": self.tags,
            "provenance": self.provenance,
            "timestamp": self.timestamp
        }


@dataclass
class EvaluationMetrics:
    """Quantitative performance metrics measured on test datasets."""
    macro_accuracy: float = 0.0
    macro_f1: float = 0.0
    per_field_metrics: Dict[str, FieldMetrics] = field(default_factory=dict)
    table_row_precision: float = 1.0
    table_row_recall: float = 1.0
    table_row_f1: float = 1.0
    table_cell_precision: float = 1.0
    table_cell_recall: float = 1.0
    table_cell_f1: float = 1.0
    end_to_end_accuracy: float = 0.0
    arithmetic_consistency_pct: float = 0.0
    document_inconsistent_pct: float = 0.0
    total_failures: int = 0
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    runtime_median_seconds: float = 0.0
    runtime_p95_seconds: float = 0.0
    vision_fallback_calls: int = 0
    sample_count: int = 0
    doc_results: Dict[str, DocumentEvaluationResult] = field(default_factory=dict)

    # Backward compatibility properties
    @property
    def overall_accuracy_pct(self) -> float:
        return self.macro_accuracy

    @property
    def field_accuracy_pct(self) -> float:
        return self.macro_accuracy

    @property
    def table_f1_score(self) -> float:
        return self.table_cell_f1

    @property
    def math_consistency_pct(self) -> float:
        return self.arithmetic_consistency_pct

    @property
    def runtime_seconds(self) -> float:
        return self.runtime_median_seconds

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["overall_accuracy_pct"] = self.macro_accuracy
        res["field_accuracy_pct"] = self.macro_accuracy
        res["table_f1_score"] = self.table_cell_f1
        res["math_consistency_pct"] = self.arithmetic_consistency_pct
        res["runtime_seconds"] = self.runtime_median_seconds
        return res


@dataclass
class DecisionRecord:
    """Final auditable decision record produced by the RRSI Selection Gate."""
    proposal_id: str
    area: str
    status: ExperimentStatus
    process_outcome: ProcessOutcome
    measurement_status: str = "measured"  # "measured" | "not_measured"
    integrity_exempt: bool = False  # True for leak/overfit removals
    baseline_accuracy: Optional[float] = None
    candidate_accuracy: Optional[float] = None
    delta_accuracy: Optional[float] = None
    paired_ci_lower: Optional[float] = None
    paired_ci_upper: Optional[float] = None
    is_statistically_significant: bool = False
    leakage_screen_passed: bool = True
    leakage_findings: List[str] = field(default_factory=list)
    cost_aware_accepted: bool = False
    latency_delta_seconds: Optional[float] = None
    vision_fallback_calls: int = 0
    case_regressions: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    pruning_recommendations: List[str] = field(default_factory=list)
    failure_deltas: Dict[str, int] = field(default_factory=dict)
    provenance: str = "real"
    eval_run_id: Optional[str] = None
    eval_results_sha256: Optional[str] = None
    baseline_commit: Optional[str] = None
    candidate_commit: Optional[str] = None
    git_commit_hash: str = ""
    git_diff_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if isinstance(self.status, ExperimentStatus):
            res["status"] = self.status.value
        if isinstance(self.process_outcome, ProcessOutcome):
            res["process_outcome"] = self.process_outcome.value
        return res
