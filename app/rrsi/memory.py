# app/rrsi/memory.py
"""
Proposal Memory & Negative Knowledge Registry.
1. Remembers previously failed approaches to prevent repeating ineffective changes.
2. Detects component stagnation and forces/recommends Area Shifts.
3. Persists historical hypotheses and outcomes in data/rrsi/proposal_memory.json.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.rrsi.taxonomy import (
    ProposalHypothesis,
    PipelineArea,
    ExperimentStatus,
    DecisionRecord
)


class NegativeKnowledgeRegistry:
    """
    Maintains memory of tested ideas, rejected approaches, and failure modes.
    Prevents cyclic/repetitive testing and guides search toward unexplored areas.
    """

    def __init__(self, storage_path: Optional[Path] = None, stagnation_threshold: int = 3):
        if storage_path is None:
            self.storage_path = Path("data/rrsi/proposal_memory.json")
        else:
            self.storage_path = Path(storage_path)

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.stagnation_threshold = stagnation_threshold

        self.history: List[Dict[str, Any]] = []
        self.failed_approaches: List[Dict[str, Any]] = []
        self.accepted_approaches: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    self.failed_approaches = data.get("failed_approaches", [])
                    self.accepted_approaches = data.get("accepted_approaches", [])
            except Exception as e:
                print(f"[NegativeKnowledgeRegistry] Warning loading memory: {e}")
                self.history = []
                self.failed_approaches = []
                self.accepted_approaches = []

    def save(self):
        data = {
            "history": self.history,
            "failed_approaches": self.failed_approaches,
            "accepted_approaches": self.accepted_approaches
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_decision(self, hypothesis: ProposalHypothesis, decision: DecisionRecord):
        """Records a completed proposal and its outcome in memory."""
        entry = {
            "proposal_id": hypothesis.proposal_id,
            "area": hypothesis.area.value,
            "title": hypothesis.title,
            "description": hypothesis.description,
            "rationale": hypothesis.rationale,
            "tags": hypothesis.tags,
            "code_diff_summary": hypothesis.code_diff_summary,
            "status": decision.status.value,
            "delta_accuracy": decision.delta_accuracy,
            "rejection_reasons": decision.rejection_reasons,
            "provenance": decision.provenance,
            "timestamp": decision.timestamp
        }
        self.history.append(entry)

        if decision.status == ExperimentStatus.ACCEPTED:
            self.accepted_approaches.append(entry)
        elif decision.status == ExperimentStatus.REJECTED:
            self.failed_approaches.append(entry)

        self.save()

    def check_proposal_viability(self, candidate: ProposalHypothesis) -> Tuple[bool, List[str], Optional[PipelineArea]]:
        """
        Validates whether a new candidate idea is viable or violates negative knowledge rules:
        1. Checks for duplication or high similarity with previously failed approaches.
        2. Checks for area stagnation (too many consecutive failed micro-tweaks in the same component).

        Returns:
            (is_viable: bool, warnings_or_reasons: List[str], recommended_area: Optional[PipelineArea])
        """
        reasons = []

        # 1. Similarity Check with known failed approaches
        candidate_tokens = self._tokenize(f"{candidate.title} {candidate.description} {candidate.rationale} {' '.join(candidate.tags)}")

        for failed in self.failed_approaches:
            if failed.get("area") == candidate.area.value:
                failed_tokens = self._tokenize(f"{failed.get('title', '')} {failed.get('description', '')} {failed.get('rationale', '')} {' '.join(failed.get('tags', []))}")
                jaccard = self._jaccard_similarity(candidate_tokens, failed_tokens)

                if jaccard >= 0.55:
                    reasons.append(
                        f"PROPOSAL DUPLICATION WARNING: Highly similar to previously failed approach '{failed.get('title')}' "
                        f"(Proposal ID: {failed.get('proposal_id')}, Jaccard Similarity: {jaccard:.2f}). "
                        f"Past Failure Reason: {', '.join(failed.get('rejection_reasons', ['Unknown']))}."
                    )

        # 2. Area Stagnation Check
        consecutive_failures = self.get_consecutive_failures_in_area(candidate.area)
        recommended_area = None

        if consecutive_failures >= self.stagnation_threshold:
            recommended_area = self.recommend_alternative_area(candidate.area)
            reasons.append(
                f"STAGNATION DETECTED: Pipeline area '{candidate.area.value}' has {consecutive_failures} consecutive "
                f"failures without net improvement. RRSI rule: Discontinue micro-tweaks to this component and "
                f"pivot to an orthogonal area (Recommended Area: '{recommended_area.value}')."
            )

        is_viable = len(reasons) == 0
        return is_viable, reasons, recommended_area

    def get_consecutive_failures_in_area(self, area: PipelineArea) -> int:
        """Counts how many recent proposals in this area failed consecutively."""
        count = 0
        for entry in reversed(self.history):
            if entry.get("area") == area.value:
                if entry.get("status") == ExperimentStatus.REJECTED.value:
                    count += 1
                elif entry.get("status") == ExperimentStatus.ACCEPTED.value:
                    break
        return count

    def recommend_alternative_area(self, stagnant_area: PipelineArea) -> PipelineArea:
        """
        Selects an orthogonal, underexplored pipeline area when a component stagnates.
        """
        orthogonal_map = {
            PipelineArea.ENTITY_EXTRACTION: [PipelineArea.DUAL_ENGINE_CONSENSUS, PipelineArea.OCR_ENGINE, PipelineArea.INPUT_PREPROCESSING],
            PipelineArea.TABLE_EXTRACTION: [PipelineArea.ARITHMETIC_VERIFIER, PipelineArea.DUAL_ENGINE_CONSENSUS, PipelineArea.INPUT_PREPROCESSING],
            PipelineArea.INPUT_PREPROCESSING: [PipelineArea.OCR_ENGINE, PipelineArea.ENTITY_EXTRACTION],
            PipelineArea.OCR_ENGINE: [PipelineArea.DUAL_ENGINE_CONSENSUS, PipelineArea.ARITHMETIC_VERIFIER],
            PipelineArea.ARITHMETIC_VERIFIER: [PipelineArea.TABLE_EXTRACTION, PipelineArea.SCHEMA_MAPPING],
            PipelineArea.DUAL_ENGINE_CONSENSUS: [PipelineArea.ENTITY_EXTRACTION, PipelineArea.TABLE_EXTRACTION],
            PipelineArea.SCHEMA_MAPPING: [PipelineArea.ENTITY_EXTRACTION, PipelineArea.ARITHMETIC_VERIFIER]
        }

        candidates = orthogonal_map.get(stagnant_area, list(PipelineArea))
        # Choose the area with the least recent failures or least attempts
        area_counts = {a: sum(1 for h in self.history if h.get("area") == a.value) for a in candidates}
        return min(area_counts, key=area_counts.get)

    @staticmethod
    def _tokenize(text: str) -> set:
        words = re.findall(r"[a-z0-9_]{3,}", text.lower())
        stopwords = {"this", "that", "with", "from", "have", "been", "using", "into", "more", "such", "each", "were"}
        return {w for w in words if w not in stopwords}

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0
