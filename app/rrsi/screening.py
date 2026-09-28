# app/rrsi/screening.py
"""
Selection-Side Verification: Leakage Screening & Pruning Analyzer.
1. Leakage Screening: Audits proposed code changes to detect test-data memorization,
   hardcoded coordinates, specific vendor strings, or fragile heuristics.
2. Pruning Analyzer: Identifies dead, redundant, or superseded heuristics in the OCR pipeline.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple


class LeakageScreening:
    """
    Prevents data leakage and overfitting. Ensures changes generalize to unseen invoices
    rather than memorizing the test dataset.
    """

    # Known sample-specific test tokens (invoice numbers / synthetic tokens)
    SUSPICIOUS_SAMPLE_TOKENS = {
        "wmp/b/04531", "bms/26-27/196", "cc-1415", "170588", "c/3930"
    }

    GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
    PHONE_REGEX = re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b|\b0\d{2,4}[\-\s]?\d{6,8}\b")
    SYNTHETIC_WHITELIST = {
        "27AAPFU0939F1ZV", "27AAPFU0939F1ZW", "AAPFU0939F", "020-00000000", "9999999999"
    }

    # Patterns indicating coordinate hardcoding (e.g. ymin == 145, box[1] > 840)
    HARDCODED_COORDINATE_PATTERNS = [
        r"\b(?:box|bbox|ymin|ymax|xmin|xmax)\s*(?:==|>=|<=|>|<)\s*\d{3,4}\b",
        r"\[\s*\d{3,4}\s*,\s*\d{3,4}\s*,\s*\d{3,4}\s*,\s*\d{3,4}\s*\]",
        r"\b(?:crop|slice)\s*\[\s*\d{3,4}\s*:\s*\d{3,4}\s*\]"
    ]

    @classmethod
    def _load_forbidden_hashes(cls) -> Tuple[str, set]:
        cfg_path = Path("config/forbidden_hashes.json")
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("salt", ""), set(data.get("hashes", []))
            except Exception:
                pass
        return "", set()

    @classmethod
    def audit_code_changes(cls, code_or_diff: str) -> Tuple[bool, List[str]]:
        """
        Scans code diff or candidate implementation for leakage and memorization.
        Returns:
            (passed: bool, findings: List[str])
        """
        findings = []
        lower_code = code_or_diff.lower()

        # 1. Check for specific test tokens / sample memorization
        for token in cls.SUSPICIOUS_SAMPLE_TOKENS:
            if re.search(r"['\"]" + re.escape(token) + r"['\"]", lower_code):
                findings.append(
                    f"DATA LEAKAGE DETECTED: Code contains hardcoded test token '{token}'. "
                    f"Production heuristics must be generalized, not tailored to specific sample entities."
                )

        # 1b. Check for literal GSTINs and Indian phone numbers in code
        for match in cls.GSTIN_REGEX.findall(code_or_diff):
            if match not in cls.SYNTHETIC_WHITELIST:
                findings.append(
                    f"DATA LEAKAGE DETECTED: Code contains hardcoded GSTIN '{match}'."
                )

        for match in cls.PHONE_REGEX.findall(code_or_diff):
            if match not in cls.SYNTHETIC_WHITELIST:
                findings.append(
                    f"DATA LEAKAGE DETECTED: Code contains hardcoded phone number '{match}'."
                )

        # 1c. Check for salted forbidden entity names
        salt, forbidden_hashes = cls._load_forbidden_hashes()
        if salt and forbidden_hashes:
            tokens = [t.lower().strip(" \t\r\n'\",;:.()[]{}") for t in code_or_diff.split() if len(t.strip()) >= 4]
            for token in tokens:
                thash = hashlib.sha256((salt + token).encode("utf-8")).hexdigest()
                if thash in forbidden_hashes:
                    findings.append("DATA LEAKAGE DETECTED: Code contains sensitive entity name matching forbidden hash.")
                    break

        # 2. Check for absolute coordinate hardcoding
        for pattern in cls.HARDCODED_COORDINATE_PATTERNS:
            matches = re.findall(pattern, code_or_diff)
            if matches:
                findings.append(
                    f"COORDINATE OVERFITTING WARNING: Found hardcoded pixel coordinates: {matches[:3]}. "
                    f"Heuristics must use relative ratios (0.0 to 1.0) or dynamic spatial clustering."
                )

        passed = len(findings) == 0
        return passed, findings

    @classmethod
    def audit_split_generalization(
        cls,
        calibration_delta: float,
        holdout_delta: float,
        min_holdout_retention: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Validates that performance gains on the calibration/training set generalize to the holdout set.
        Rejects candidates that overfit to calibration data while regressing or stagnating on holdout data.
        """
        if holdout_delta < 0.0:
            return False, f"GENERALIZATION FAILURE: Candidate regressed on holdout set by {holdout_delta:.2f}% despite gain on calibration set ({calibration_delta:.2f}%)."

        if calibration_delta > 2.0 and holdout_delta < (calibration_delta * min_holdout_retention):
            return False, (
                f"LEAKAGE/OVERFITTING DETECTED: Disproportionate calibration gain (+{calibration_delta:.2f}%) "
                f"vs holdout retention (+{holdout_delta:.2f}%). Fails the {min_holdout_retention*100:.0f}% generalization retention threshold."
            )

        return True, "Split generalization verified."


class PruningAnalyzer:
    """
    Identifies redundant, dead, or superseded heuristics in the OCR pipeline.
    Recommends removal of clutter to keep pipeline lean, maintainable, and fast.
    """

    @staticmethod
    def analyze_regex_redundancy(patterns: List[str]) -> List[str]:
        """Detects whether any regex pattern is completely shadowed or duplicated by another."""
        recommendations = []
        cleaned = [p.strip() for p in patterns if p.strip()]

        for i, p1 in enumerate(cleaned):
            for j, p2 in enumerate(cleaned):
                if i != j and p1 == p2:
                    recommendations.append(f"PRUNING OPPORTUNITY: Exact duplicate regex pattern '{p1}' found at index {i} and {j}.")
                elif i != j and len(p1) < len(p2) and p1 in p2 and p1.startswith(r".*") and p1.endswith(r".*"):
                    recommendations.append(f"PRUNING OPPORTUNITY: Pattern '{p2}' may be shadowed by broader pattern '{p1}'.")

        return list(set(recommendations))

    @staticmethod
    def audit_keyword_list(keywords: set, sample_occurrences: Dict[str, int]) -> List[str]:
        """Identifies keywords that never activate across any real-world invoice samples."""
        recommendations = []
        for kw in sorted(keywords):
            count = sample_occurrences.get(kw.lower(), 0)
            if count == 0:
                recommendations.append(f"PRUNING CANDIDATE: Keyword '{kw}' has 0 activations across dataset. Consider deprecating or archiving.")
        return recommendations
