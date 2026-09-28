# app/rrsi/ledger.py
"""
Auditable Experiment Ledger.
Persists all RRSI experiment decisions, error breakdowns, and metric deltas to data/rrsi/experiment_ledger.json.
Provides human-readable summaries and regression tracking.
"""

import json
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

    def append(self, decision: DecisionRecord):
        """Appends a new decision record to the auditable ledger."""
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
        status_badge = "[ACCEPTED]" if decision.status == ExperimentStatus.ACCEPTED else "[REJECTED]"
        delta_sign = "+" if decision.delta_accuracy > 0 else ""

        lines = [
            f"### RRSI Evaluation Report: `{decision.proposal_id}` ({status_badge})",
            f"- **Target Area**: `{decision.area}`",
            f"- **Baseline Accuracy**: `{decision.baseline_accuracy:.2f}%`",
            f"- **Candidate Accuracy**: `{decision.candidate_accuracy:.2f}%` ({delta_sign}{decision.delta_accuracy:.2f}%)",
            f"- **Statistical Noise Floor**: `+/-{decision.noise_floor:.2f}%`",
            f"- **Statistically Significant**: `{'YES' if decision.is_statistically_significant else 'NO (Indistinguishable from luck/noise)'}`",
            f"- **Leakage Screen**: `{'PASSED' if decision.leakage_screen_passed else 'FAILED (Memorization/Cheating detected)'}`",
            f"- **Cost-Aware Acceptance**: `{'PASSED' if decision.cost_aware_accepted else 'FAILED (Latency/Cost budget exceeded)'}` (dt: {decision.latency_delta_seconds:+.2f}s)",
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
