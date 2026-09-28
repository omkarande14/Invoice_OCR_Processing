# tests/test_evaluator_refactor.py
import pytest
from decimal import Decimal
from app.rrsi.evaluator import StatisticalEvaluator
from app.rrsi.taxonomy import (
    ExtractionErrorClass,
    ProcessOutcome,
    ProposalHypothesis,
    PipelineArea,
    ExperimentStatus
)

def test_alignment_by_doc_id_not_zip():
    evaluator = StatisticalEvaluator()
    # Shuffled order between predictions and ground truths
    gt = [
        {"invoice_id": "doc_A", "invoice_number": "INV-A", "invoice_amount": "100.00"},
        {"invoice_id": "doc_B", "invoice_number": "INV-B", "invoice_amount": "200.00"}
    ]
    preds = [
        {"invoice_id": "doc_B", "invoice_number": "INV-B", "invoice_amount": "200.00"},
        {"invoice_id": "doc_A", "invoice_number": "INV-A", "invoice_amount": "100.00"}
    ]

    metrics, failures = evaluator.evaluate_predictions(preds, gt)
    # Since alignment is by doc_id, all fields should match 100%
    assert metrics.macro_accuracy == 100.0
    assert len(failures) == 0

def test_missing_prediction_handled_as_pipeline_error():
    evaluator = StatisticalEvaluator()
    gt = [
        {"invoice_id": "doc_A", "invoice_number": "INV-A", "invoice_amount": "100.00"},
        {"invoice_id": "doc_B", "invoice_number": "INV-B", "invoice_amount": "200.00"}
    ]
    # Prediction for doc_B is completely missing
    preds = [
        {"invoice_id": "doc_A", "invoice_number": "INV-A", "invoice_amount": "100.00"}
    ]

    metrics, failures = evaluator.evaluate_predictions(preds, gt)
    assert "doc_B" in metrics.doc_results
    doc_b_res = metrics.doc_results["doc_B"]
    assert doc_b_res.is_end_to_end_correct is False
    assert any(f.primary_class == ExtractionErrorClass.PIPELINE_ERROR for f in failures if f.file_id == "doc_B")

def test_precision_counts_hallucination_when_gt_is_null():
    evaluator = StatisticalEvaluator()
    # Ground truth has no PO_number
    gt = [{"invoice_id": "doc_A", "invoice_number": "INV-A", "PO_number": None}]
    # Prediction extracted a hallucinated PO_number
    preds = [{"invoice_id": "doc_A", "invoice_number": "INV-A", "PO_number": "PO-9999"}]

    metrics, failures = evaluator.evaluate_predictions(preds, gt)
    po_metrics = metrics.per_field_metrics.get("PO_number")
    assert po_metrics is not None
    assert po_metrics.hallucinations == 1
    assert any(f.primary_class == ExtractionErrorClass.HALLUCINATION for f in failures)

def test_per_document_outcomes_structure():
    evaluator = StatisticalEvaluator()
    gt = [{"invoice_id": "doc_A", "invoice_number": "INV-A", "invoice_date": "2026-08-20"}]
    preds = [{"invoice_id": "doc_A", "invoice_number": "INV-A", "invoice_date": "2026-08-21"}]

    metrics, failures = evaluator.evaluate_predictions(preds, gt)
    assert "doc_A" in metrics.doc_results
    doc_res = metrics.doc_results["doc_A"]
    assert doc_res.field_outcomes["invoice_number"] is True
    assert doc_res.field_outcomes["invoice_date"] is False
    assert doc_res.field_errors["invoice_date"] == "date_mismatch"

def test_selection_gate_requires_selection_split():
    evaluator = StatisticalEvaluator()
    hypo = ProposalHypothesis(
        proposal_id="PROP-101",
        area=PipelineArea.ENTITY_EXTRACTION,
        title="Test gate",
        description="Testing selection gate",
        rationale="Unit test"
    )
    dev_base = evaluator.evaluate_predictions([{"invoice_id": "1", "invoice_number": "A"}], [{"invoice_id": "1", "invoice_number": "A"}])[0]
    dev_cand = evaluator.evaluate_predictions([{"invoice_id": "1", "invoice_number": "A"}], [{"invoice_id": "1", "invoice_number": "A"}])[0]

    # Omitting selection split should raise ValueError (fail closed)
    with pytest.raises(ValueError, match="Selection-split metrics are required"):
        evaluator.decide_acceptance(
            hypothesis=hypo,
            baseline_dev=dev_base,
            candidate_dev=dev_cand,
            baseline_selection=None,
            candidate_selection=None
        )
