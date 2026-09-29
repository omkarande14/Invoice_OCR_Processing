# tests/test_rrsi_framework.py
"""
Comprehensive Unit Tests for the RRSI Framework:
- Proposal Memory & Negative Knowledge Registry
- Area Stagnation & Shift Recommendations
- Leakage Screening (hardcoded tokens, coordinates, holdout retention)
- Statistical Evaluator (noise floor calculation, cost-aware acceptance)
- Granular Error Taxonomy Classification
- Experiment Ledger Persistence & Auditing
"""

import tempfile
import pytest
import hashlib
from pathlib import Path

from app.rrsi import (
    ErrorCategory,
    PipelineArea,
    ExperimentStatus,
    FailureDetail,
    ProposalHypothesis,
    EvaluationMetrics,
    DecisionRecord,
    NegativeKnowledgeRegistry,
    LeakageScreening,
    PruningAnalyzer,
    StatisticalEvaluator,
    ExperimentLedger,
    RRSIOrchestrator
)
from app.rrsi.taxonomy import ProcessOutcome, ExtractionErrorClass


@pytest.fixture
def temp_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "proposal_memory.json"


@pytest.fixture
def temp_ledger():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "experiment_ledger.json"


def _create_dummy_decision(
    proposal_id="PROP-001",
    area=PipelineArea.ENTITY_EXTRACTION.value,
    status=ExperimentStatus.REJECTED,
    rejection_reasons=None
):
    return DecisionRecord(
        proposal_id=proposal_id,
        area=area,
        status=status,
        process_outcome=ProcessOutcome.INSIGNIFICANT if status == ExperimentStatus.REJECTED else ProcessOutcome.ACCEPTED,
        baseline_accuracy=85.0,
        candidate_accuracy=85.0,
        delta_accuracy=0.0,
        paired_ci_lower=0.0,
        paired_ci_upper=0.0,
        is_statistically_significant=False,
        leakage_screen_passed=True,
        leakage_findings=[],
        cost_aware_accepted=False,
        latency_delta_seconds=0.0,
        vision_fallback_calls=0,
        case_regressions=[],
        rejection_reasons=rejection_reasons or ["NO ACCURACY GAIN: Candidate did not improve baseline."],
        pruning_recommendations=[],
        failure_deltas={}
    )


# -----------------------------------------------------------------------------
# 1. Proposal Side: Memory, Negative Knowledge & Area Shifting
# -----------------------------------------------------------------------------
def test_negative_knowledge_blocks_repeated_failed_proposals(temp_memory):
    registry = NegativeKnowledgeRegistry(storage_path=temp_memory, stagnation_threshold=3)

    # 1. Record a failed hypothesis
    failed_hypo = ProposalHypothesis(
        proposal_id="PROP-001",
        area=PipelineArea.ENTITY_EXTRACTION,
        title="Tweak invoice date regex to support slash dot combinations",
        description="Add regex pattern for dd.mm.yyyy and slash formats in date parser",
        rationale="Fix date extraction test failures",
        tags=["regex", "date", "slash"]
    )
    decision = _create_dummy_decision("PROP-001", PipelineArea.ENTITY_EXTRACTION.value, ExperimentStatus.REJECTED)
    registry.record_decision(failed_hypo, decision)

    # 2. Candidate proposes a very similar idea in the same area
    candidate_hypo = ProposalHypothesis(
        proposal_id="PROP-002",
        area=PipelineArea.ENTITY_EXTRACTION,
        title="Tweak invoice date regex pattern for dot slash formats",
        description="Add regex pattern for dd.mm.yyyy date parsing",
        rationale="Fix date extraction failures",
        tags=["regex", "date", "slash"]
    )

    is_viable, warnings, _ = registry.check_proposal_viability(candidate_hypo)
    assert not is_viable
    assert any("PROPOSAL DUPLICATION WARNING" in w for w in warnings)


def test_area_stagnation_triggers_area_shift(temp_memory):
    registry = NegativeKnowledgeRegistry(storage_path=temp_memory, stagnation_threshold=3)

    # Record 3 consecutive failed attempts in ENTITY_EXTRACTION
    for i in range(1, 4):
        hypo = ProposalHypothesis(
            proposal_id=f"PROP-00{i}",
            area=PipelineArea.ENTITY_EXTRACTION,
            title=f"Unique attempt {i} in regex entity extraction",
            description=f"Description {i}",
            rationale=f"Rationale {i}"
        )
        decision = _create_dummy_decision(f"PROP-00{i}", PipelineArea.ENTITY_EXTRACTION.value, ExperimentStatus.REJECTED)
        registry.record_decision(hypo, decision)

    # Attempt a 4th proposal in the same component
    new_hypo = ProposalHypothesis(
        proposal_id="PROP-004",
        area=PipelineArea.ENTITY_EXTRACTION,
        title="Another entity extraction attempt",
        description="More micro tweaks",
        rationale="Hoping for improvement"
    )

    is_viable, warnings, rec_area = registry.check_proposal_viability(new_hypo)
    assert not is_viable
    assert any("STAGNATION DETECTED" in w for w in warnings)
    assert rec_area is not None
    assert rec_area != PipelineArea.ENTITY_EXTRACTION


# -----------------------------------------------------------------------------
# 2. Selection Side: Leakage Screening & Pruning
# -----------------------------------------------------------------------------
def test_leakage_screening_detects_hardcoded_sample_tokens():
    clean_code = "def parse_inv(text): return re.findall(r'INV-[0-9]+', text)"
    passed, findings = LeakageScreening.audit_code_changes(clean_code)
    assert passed
    assert len(findings) == 0

    # Synthetic forbidden token passed via runtime forbidden hashes
    salt = "TEST_SALT_MOCK_"
    token = "SYNTH_LEAKY_TOKEN_01"
    norm = "synthleakytoken01"
    h = hashlib.sha256((salt + norm).encode("utf-8")).hexdigest()

    leaky_code = f"if '{token}' in text: return '{token}'"
    passed, findings = LeakageScreening.audit_code_changes(
        leaky_code,
        salt=salt,
        forbidden_hashes={h}
    )
    assert not passed
    assert any("DATA LEAKAGE DETECTED" in f for f in findings)


def test_leakage_screening_detects_coordinate_overfitting():
    overfitted_code = "if ymin >= 145 and ymax <= 320: return crop[120:850]"
    passed, findings = LeakageScreening.audit_code_changes(overfitted_code)
    assert not passed
    assert any("COORDINATE OVERFITTING DETECTED" in f for f in findings)


def test_generalization_split_screening():
    # Healthy generalization
    passed, _ = LeakageScreening.compare_dev_vs_selection_gain(
        dev_gain=0.035, selection_gain=0.028, selection_ci_lower=0.015
    )
    assert passed

    # Regression / CI lower bound <= 0
    passed, msg = LeakageScreening.compare_dev_vs_selection_gain(
        dev_gain=0.040, selection_gain=-0.012, selection_ci_lower=-0.025
    )
    assert not passed
    assert "GENERALIZATION REJECTED" in msg

    # Disproportionate overfit (dev high, selection gain gap > max_gap)
    passed, msg = LeakageScreening.compare_dev_vs_selection_gain(
        dev_gain=0.080, selection_gain=0.005, selection_ci_lower=0.001, max_gap=0.05
    )
    assert not passed
    assert "Dev-selection gap" in msg


def test_pruning_analyzer_identifies_rule_ablation_candidates():
    rule_counts = {"rule_legacy_tax": 0, "rule_modern_gst": 100}
    # Below corpus size -> insufficient evidence
    res_insufficient = PruningAnalyzer.audit_rule_ablation_candidates(
        rule_counts, corpus_doc_count=10, corpus_template_count=2
    )
    assert res_insufficient["status"] == "INSUFFICIENT_EVIDENCE"

    # Sufficient corpus -> identifies zero-hit candidates
    res_sufficient = PruningAnalyzer.audit_rule_ablation_candidates(
        rule_counts, corpus_doc_count=50, corpus_template_count=15
    )
    assert res_sufficient["status"] == "CANDIDATES_IDENTIFIED"
    assert "rule_legacy_tax" in res_sufficient["candidates"]


# -----------------------------------------------------------------------------
# 3. Statistical Evaluator: Bootstrap CI, Cost-Awareness & Error Taxonomy
# -----------------------------------------------------------------------------
def test_selection_gate_requires_selection_metrics():
    evaluator = StatisticalEvaluator()
    hypo = ProposalHypothesis(
        proposal_id="PROP-009",
        area=PipelineArea.ENTITY_EXTRACTION,
        title="Check gate enforcement",
        description="Gate must fail-closed if selection split is omitted",
        rationale="Integrity check"
    )
    dev_m = evaluator.evaluate_predictions(
        [{"invoice_id": "1", "invoice_number": "A"}],
        [{"invoice_id": "1", "invoice_number": "A"}]
    )[0]

    with pytest.raises(ValueError, match="Selection-split metrics are required"):
        evaluator.decide_acceptance(
            hypothesis=hypo,
            baseline_dev=dev_m,
            candidate_dev=dev_m,
            baseline_selection=None,
            candidate_selection=None
        )


def test_evaluator_rejects_candidate_within_noise_floor():
    evaluator = StatisticalEvaluator()
    hypo = ProposalHypothesis(
        proposal_id="PROP-010",
        area=PipelineArea.TABLE_EXTRACTION,
        title="Minor column spacing tweak",
        description="Shift column border by 2px",
        rationale="Experimental"
    )

    # Identical predictions on selection split -> delta is 0 -> CI includes 0 -> Rejected
    gt = [{"invoice_id": str(i), "invoice_number": f"INV-{i}"} for i in range(10)]
    base_preds = [{"invoice_id": str(i), "invoice_number": f"INV-{i}"} for i in range(10)]
    cand_preds = [{"invoice_id": str(i), "invoice_number": f"INV-{i}"} for i in range(10)]

    base_m, _ = evaluator.evaluate_predictions(base_preds, gt, runtime_seconds=2.0)
    cand_m, _ = evaluator.evaluate_predictions(cand_preds, gt, runtime_seconds=2.0)

    decision = evaluator.decide_acceptance(
        hypothesis=hypo,
        baseline_dev=base_m,
        candidate_dev=cand_m,
        baseline_selection=base_m,
        candidate_selection=cand_m
    )
    assert decision.status == ExperimentStatus.REJECTED
    assert not decision.is_statistically_significant
    assert any("NO ACCURACY GAIN" in r or "NOT STATISTICALLY SIGNIFICANT" in r for r in decision.rejection_reasons)


def test_cost_aware_acceptance_rejects_latency_blowup():
    evaluator = StatisticalEvaluator(cost_lambda=0.50)
    hypo = ProposalHypothesis(
        proposal_id="PROP-011",
        area=PipelineArea.OCR_ENGINE,
        title="Triple pass multi-scale OCR rendering",
        description="Render image at 100, 200, and 400 DPI and merge results",
        rationale="Higher resolution"
    )

    # Base: 80% accuracy, 1.0s runtime
    gt = [{"invoice_id": str(i), "invoice_number": f"INV-{i}"} for i in range(10)]
    base_preds = [{"invoice_id": str(i), "invoice_number": f"INV-{i}" if i < 8 else "WRONG"} for i in range(10)]
    # Cand: 90% accuracy, but runtime balloons to 50.0s (+4900%)
    cand_preds = [{"invoice_id": str(i), "invoice_number": f"INV-{i}" if i < 9 else "WRONG"} for i in range(10)]

    base_m, _ = evaluator.evaluate_predictions(base_preds, gt, runtime_seconds=1.0)
    cand_m, _ = evaluator.evaluate_predictions(cand_preds, gt, runtime_seconds=50.0)

    decision = evaluator.decide_acceptance(
        hypothesis=hypo,
        baseline_dev=base_m,
        candidate_dev=cand_m,
        baseline_selection=base_m,
        candidate_selection=cand_m
    )
    assert decision.status == ExperimentStatus.REJECTED
    assert not decision.cost_aware_accepted
    assert any("COST UTILITY NEGATIVE" in r for r in decision.rejection_reasons)


def test_fine_grained_error_taxonomy_classification():
    evaluator = StatisticalEvaluator()

    ground_truth = [{
        "invoice_id": "1",
        "invoice_number": "INV-100",
        "invoice_date": "2026-05-12",
        "customer_name": "Acme Corp",
        "seller_gstin": "27AAPFU0939F1ZV",
        "invoice_amount": "1000.00"
    }]

    prediction = [{
        "invoice_id": "1",
        "invoice_number": "INV-999",       # INVOICE_NUMBER_MISMATCH
        "invoice_date": "2026-05-13",      # DATE_MISMATCH
        "customer_name": "Different Corp", # PARTY_NAME_MISMATCH
        "seller_gstin": "27AAPFU0939F1ZW", # GSTIN_INVALID
        "invoice_amount": "800.00"         # AMOUNT_PARSE
    }]

    metrics, failures = evaluator.evaluate_predictions(prediction, ground_truth)
    assert any(f.primary_class == ExtractionErrorClass.INVOICE_NUMBER_MISMATCH for f in failures)
    assert any(f.primary_class == ExtractionErrorClass.DATE_MISMATCH for f in failures)
    assert any(f.primary_class == ExtractionErrorClass.PARTY_NAME_MISMATCH for f in failures)
    assert any(f.primary_class == ExtractionErrorClass.GSTIN_INVALID for f in failures)
    assert any(f.primary_class == ExtractionErrorClass.AMOUNT_PARSE for f in failures)
    assert len(failures) >= 5


# -----------------------------------------------------------------------------
# 4. Orchestrator & Ledger Integration
# -----------------------------------------------------------------------------
def test_rrsi_orchestrator_end_to_end(temp_memory, temp_ledger):
    orchestrator = RRSIOrchestrator(
        memory_path=temp_memory,
        ledger_path=temp_ledger
    )

    hypo = ProposalHypothesis(
        proposal_id="PROP-E2E-001",
        area=PipelineArea.TABLE_EXTRACTION,
        title="Adaptive horizontal line clustering for tables",
        description="Group rows dynamically using morphological kernel analysis",
        rationale="Resolves merged multi-item rows"
    )

    is_viable, warnings, _ = orchestrator.submit_proposal(hypo)
    assert is_viable

    gt = [{"invoice_id": str(i), "invoice_number": f"INV-{i}"} for i in range(10)]
    base_preds = [{"invoice_id": str(i), "invoice_number": f"INV-{i}" if i < 4 else "WRONG"} for i in range(10)]
    cand_preds = [{"invoice_id": str(i), "invoice_number": f"INV-{i}"} for i in range(10)] # 100% correct

    base_dev, _ = orchestrator.evaluator.evaluate_predictions(base_preds, gt, runtime_seconds=2.0)
    cand_dev, _ = orchestrator.evaluator.evaluate_predictions(cand_preds, gt, runtime_seconds=2.1)
    base_sel, _ = orchestrator.evaluator.evaluate_predictions(base_preds, gt, runtime_seconds=2.0)
    cand_sel, _ = orchestrator.evaluator.evaluate_predictions(cand_preds, gt, runtime_seconds=2.1)

    decision = orchestrator.evaluate_and_decide(
        hypothesis=hypo,
        baseline_dev=base_dev,
        candidate_dev=cand_dev,
        baseline_selection=base_sel,
        candidate_selection=cand_sel
    )

    assert decision.status == ExperimentStatus.ACCEPTED
    assert decision.is_statistically_significant

    # Verify ledger was persisted
    assert len(orchestrator.ledger.records) == 1
    assert orchestrator.ledger.records[0]["proposal_id"] == "PROP-E2E-001"

    # Verify memory was updated with accepted approach
    assert len(orchestrator.memory.accepted_approaches) == 1
