import json
import hashlib
import pytest
from pathlib import Path

from app.rrsi.ledger import ExperimentLedger
from app.rrsi.taxonomy import DecisionRecord, ExperimentStatus, ProcessOutcome


@pytest.fixture
def temp_ledger(tmp_path):
    ledger_file = tmp_path / "test_ledger.json"
    return ExperimentLedger(ledger_path=ledger_file)


@pytest.fixture
def dummy_results_file(tmp_path):
    res_path = tmp_path / "run_20260928_eval.json"
    data = {
        "run_id": "run_20260928_eval",
        "doc_results": {
            "INV_001": {"accuracy": 1.0, "field_outcomes": {"vendor_name": True}}
        },
        "per_field_metrics": {
            "vendor_name": {"precision": 1.0, "recall": 1.0}
        }
    }
    content_bytes = json.dumps(data, indent=2).encode("utf-8")
    res_path.write_bytes(content_bytes)
    sha256 = hashlib.sha256(content_bytes).hexdigest()
    return res_path, sha256


def test_ledger_rejects_hand_supplied_metrics_without_eval_run(temp_ledger):
    record = DecisionRecord(
        proposal_id="PROP-HAND-TYPED",
        area="table_extraction",
        status=ExperimentStatus.ACCEPTED,
        process_outcome=ProcessOutcome.ACCEPTED,
        measurement_status="measured",
        baseline_accuracy=95.0,
        candidate_accuracy=96.0,
        delta_accuracy=1.0,
        eval_run_id=None,
        eval_results_sha256=None,
        baseline_commit=None,
        candidate_commit=None,
        git_diff_hash=None
    )
    with pytest.raises(ValueError, match="requires a valid eval_run_id"):
        temp_ledger.append(record)


def test_ledger_rejects_mismatched_sha256(temp_ledger, dummy_results_file):
    res_path, true_sha = dummy_results_file
    record = DecisionRecord(
        proposal_id="PROP-BAD-SHA",
        area="table_extraction",
        status=ExperimentStatus.ACCEPTED,
        process_outcome=ProcessOutcome.ACCEPTED,
        measurement_status="measured",
        baseline_accuracy=95.0,
        candidate_accuracy=96.0,
        delta_accuracy=1.0,
        eval_run_id=str(res_path),
        eval_results_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        baseline_commit="8c9c281",
        candidate_commit="aff9aad",
        git_diff_hash="abcdef123456"
    )
    with pytest.raises(ValueError, match="eval_results_sha256 mismatch"):
        temp_ledger.append(record)


def test_ledger_rejects_missing_commit_hashes(temp_ledger, dummy_results_file):
    res_path, true_sha = dummy_results_file
    record = DecisionRecord(
        proposal_id="PROP-NO-COMMIT",
        area="table_extraction",
        status=ExperimentStatus.ACCEPTED,
        process_outcome=ProcessOutcome.ACCEPTED,
        measurement_status="measured",
        baseline_accuracy=95.0,
        candidate_accuracy=96.0,
        delta_accuracy=1.0,
        eval_run_id=str(res_path),
        eval_results_sha256=true_sha,
        baseline_commit="",
        candidate_commit="",
        git_diff_hash=""
    )
    with pytest.raises(ValueError, match="baseline_commit"):
        temp_ledger.append(record)


def test_ledger_accepts_verified_measured_entry(temp_ledger, dummy_results_file):
    res_path, true_sha = dummy_results_file
    record = DecisionRecord(
        proposal_id="PROP-VERIFIED",
        area="table_extraction",
        status=ExperimentStatus.ACCEPTED,
        process_outcome=ProcessOutcome.ACCEPTED,
        measurement_status="measured",
        baseline_accuracy=95.0,
        candidate_accuracy=96.0,
        delta_accuracy=1.0,
        eval_run_id=str(res_path),
        eval_results_sha256=true_sha,
        baseline_commit="8c9c281",
        candidate_commit="aff9aad",
        git_diff_hash="d3b07384d113edec49eaa6238ad5ff00"
    )
    temp_ledger.append(record)
    assert len(temp_ledger.records) == 1
    assert temp_ledger.records[0]["proposal_id"] == "PROP-VERIFIED"


def test_ledger_accepts_not_measured_integrity_exempt_entry(temp_ledger):
    record = DecisionRecord(
        proposal_id="REMOVE-HARDCODES",
        area="entity_extraction",
        status=ExperimentStatus.ACCEPTED,
        process_outcome=ProcessOutcome.ACCEPTED,
        measurement_status="not_measured",
        integrity_exempt=True,
        baseline_accuracy=None,
        candidate_accuracy=None,
        delta_accuracy=None,
        paired_ci_lower=None,
        paired_ci_upper=None,
        latency_delta_seconds=None,
        pruning_recommendations=["Pruned static hardcoded coordinates and specific entity regexes."]
    )
    temp_ledger.append(record)
    assert len(temp_ledger.records) == 1
    rec = temp_ledger.records[0]
    assert rec["measurement_status"] == "not_measured"
    assert rec["integrity_exempt"] is True
    assert rec["baseline_accuracy"] is None


def test_ledger_rejects_not_measured_with_numeric_metrics(temp_ledger):
    record = DecisionRecord(
        proposal_id="REMOVE-HARDCODES",
        area="entity_extraction",
        status=ExperimentStatus.ACCEPTED,
        process_outcome=ProcessOutcome.ACCEPTED,
        measurement_status="not_measured",
        integrity_exempt=True,
        baseline_accuracy=95.0,  # Fabricated numeric value! Must be rejected
        candidate_accuracy=None,
        delta_accuracy=None
    )
    with pytest.raises(ValueError, match="cannot contain hand-supplied numeric metrics"):
        temp_ledger.append(record)


def test_ledger_rejects_not_measured_without_integrity_exempt(temp_ledger):
    record = DecisionRecord(
        proposal_id="RANDOM-UNMEASURED",
        area="entity_extraction",
        status=ExperimentStatus.REJECTED,
        process_outcome=ProcessOutcome.INSIGNIFICANT,
        measurement_status="not_measured",
        integrity_exempt=False,
        baseline_accuracy=None
    )
    with pytest.raises(ValueError, match="must have integrity_exempt=True"):
        temp_ledger.append(record)
