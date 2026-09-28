import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.rrsi.taxonomy import DecisionRecord, ExperimentStatus


class ExperimentLedger:
    """
    Maintains an immutable record of all tested changes, acceptance decisions,
    and fine-grained failure deltas.
    """

    def __init__(self, ledger_path: Optional[Path] = None):
        if ledger_path is None:
            self.ledger_path = Path("data/rrsi/experiment_ledger.json")
        else:
            self.ledger_path = Path(ledger_path)

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"[ExperimentLedger] Warning loading ledger: {e}")
                self.records = []

    def save(self):
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def append(self, decision: DecisionRecord, results_dir: Optional[Path] = None):
        """
        Appends a new decision record to the auditable ledger.
        Enforces strict cryptographic and empirical provenance:
        - Hand-supplied numeric metric numbers without a verified evaluation run file are rejected.
        - Measured runs require eval_run_id pointing to a valid results file whose sha256 matches eval_results_sha256,
          plus baseline_commit, candidate_commit, and git_diff_hash.
        - not_measured runs must have integrity_exempt=True and all numeric metrics set to None.
        """
        if decision.measurement_status == "not_measured":
            if not decision.integrity_exempt:
                raise ValueError("Unmeasured decision records must have integrity_exempt=True.")
            numeric_fields = [
                decision.baseline_accuracy, decision.candidate_accuracy,
                decision.delta_accuracy, decision.paired_ci_lower,
                decision.paired_ci_upper, decision.latency_delta_seconds
            ]
            if any(f is not None for f in numeric_fields):
                raise ValueError("Unmeasured decision records cannot contain hand-supplied numeric metrics. All numeric metrics must be None.")
        elif decision.measurement_status == "measured":
            if not decision.eval_run_id:
                raise ValueError("Measured decision record requires a valid eval_run_id pointing to saved evaluation results.")
            if not decision.eval_results_sha256:
                raise ValueError("Measured decision record requires eval_results_sha256.")
            if not decision.baseline_commit:
                raise ValueError("Measured decision record requires baseline_commit.")
            if not decision.candidate_commit:
                raise ValueError("Measured decision record requires candidate_commit.")
            if not decision.git_diff_hash:
                raise ValueError("Measured decision record requires a non-empty git_diff_hash.")

            # Validate results file path and checksum
            results_path = Path(decision.eval_run_id)
            if not results_path.exists() and results_dir:
                results_path = results_dir / decision.eval_run_id
            if not results_path.exists() and not results_path.is_absolute():
                for search_dir in [Path("data_private/evaluations"), Path("reports/evaluations")]:
                    cand = search_dir / decision.eval_run_id
                    if cand.exists():
                        results_path = cand
                        break
                    if cand.with_suffix(".json").exists():
                        results_path = cand.with_suffix(".json")
                        break

            if not results_path.exists():
                raise FileNotFoundError(f"Evaluation results file '{decision.eval_run_id}' not found.")

            content_bytes = results_path.read_bytes()
            computed_sha = hashlib.sha256(content_bytes).hexdigest()
            if computed_sha.lower() != decision.eval_results_sha256.lower():
                raise ValueError(
                    f"Integrity violation: eval_results_sha256 mismatch for '{decision.eval_run_id}'. "
                    f"Expected {decision.eval_results_sha256}, got {computed_sha}."
                )

            # Ensure the results file actually contains per-doc / per-field outcomes structure
            try:
                run_data = json.loads(content_bytes.decode("utf-8"))
                if not (isinstance(run_data, dict) and any(k in run_data for k in ["doc_results", "field_outcomes", "per_field_metrics", "invoices"])):
                    raise ValueError(f"Evaluation results file '{decision.eval_run_id}' missing per-doc / per-field outcomes structure.")
            except json.JSONDecodeError:
                raise ValueError(f"Evaluation results file '{decision.eval_run_id}' is not valid JSON.")
        else:
            raise ValueError(f"Unknown measurement_status '{decision.measurement_status}'. Must be 'measured' or 'not_measured'.")

        self.records.append(decision.to_dict())
        self.save()

    def get_latest_accepted(self) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent accepted proposal that formed the active baseline."""
        for rec in reversed(self.records):
            if rec.get("status") == ExperimentStatus.ACCEPTED.value:
                return rec
        return None

    def generate_markdown_summary(self, decision: DecisionRecord) -> str:
        """Generates a structured, auditable Markdown report for a candidate decision."""
        status_val = decision.status.value if isinstance(decision.status, ExperimentStatus) else decision.status
        status_badge = "[ACCEPTED]" if status_val == ExperimentStatus.ACCEPTED.value else "[REJECTED]"

        if decision.measurement_status == "not_measured":
            b_acc = "Not Measured (Integrity Exempt)"
            c_acc = "Not Measured (Integrity Exempt)"
            dt_str = "N/A"
        else:
            delta_val = decision.delta_accuracy if decision.delta_accuracy is not None else 0.0
            delta_sign = "+" if delta_val > 0 else ""
            b_acc = f"{decision.baseline_accuracy:.2f}%" if decision.baseline_accuracy is not None else "N/A"
            c_acc = f"{decision.candidate_accuracy:.2f}% ({delta_sign}{delta_val:.2f}%)" if decision.candidate_accuracy is not None else "N/A"
            dt_str = f"{decision.latency_delta_seconds:+.2f}s" if decision.latency_delta_seconds is not None else "N/A"

        lines = [
            f"### RRSI Evaluation Report: `{decision.proposal_id}` ({status_badge})",
            f"- **Target Area**: `{decision.area}`",
            f"- **Measurement Status**: `{decision.measurement_status}` (Integrity Exempt: {decision.integrity_exempt})",
            f"- **Baseline Accuracy**: `{b_acc}`",
            f"- **Candidate Accuracy**: `{c_acc}`",
            f"- **Statistically Significant**: `{'YES' if decision.is_statistically_significant else 'NO'}`",
            f"- **Leakage Screen**: `{'PASSED' if decision.leakage_screen_passed else 'FAILED'}`",
            f"- **Cost-Aware Acceptance**: `{'PASSED' if decision.cost_aware_accepted else 'FAILED'}` (dt: {dt_str})",
            ""
        ]

        if decision.rejection_reasons:
            lines.append("#### Rejection Reasons:")
            for r in decision.rejection_reasons:
                lines.append(f"- [!] {r}")
            lines.append("")

        if decision.leakage_findings:
            lines.append("#### Leakage & Overfitting Findings:")
            for lf in decision.leakage_findings:
                lines.append(f"- [LEAKAGE] {lf}")
            lines.append("")

        if decision.pruning_recommendations:
            lines.append("#### Pruning Opportunities Identified:")
            for pr in decision.pruning_recommendations:
                lines.append(f"- [PRUNE] {pr}")
            lines.append("")

        if decision.failure_deltas:
            lines.append("#### Failure Deltas by Error Category:")
            lines.append("| Error Category | Delta vs Baseline | Status |")
            lines.append("| :--- | :---: | :---: |")
            for cat, delta in decision.failure_deltas.items():
                if delta < 0:
                    icon = f"[REDUCED] by {abs(delta)}"
                elif delta > 0:
                    icon = f"[INCREASED] by {delta}"
                else:
                    icon = "[UNCHANGED] (0)"
                lines.append(f"| `{cat}` | {delta:+d} | {icon} |")

        return "\n".join(lines)
