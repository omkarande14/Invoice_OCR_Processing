# tests/test_screening_ast.py
"""
Tests for RRSI Leakage Screening (B1, B2, B3, B4, B5):
- Adversarial cases caught:
  1. Partial vendor name / 8-gram matching
  2. Suffix variation
  3. Concatenated literal ('WMP/B/' + '04531')
  4. Regex with wildcards / re.compile argument
  5. Bare number matching forbidden hash
  6. box[1] > 840 coordinate hardcoding
- Benign cases NOT flagged:
  1. [255, 255, 255, 255]
  2. cv2.resize(img, (1000, 1000))
  3. Whitelisted kernel sizes (3, 5, 7)
- B4: Paired-bootstrap dev vs selection gain comparison
- B5: PruningAnalyzer hit counter evidence threshold & ablation evaluation
"""

import pytest
import hashlib
from app.rrsi.screening import LeakageScreening, PruningAnalyzer


class TestScreeningAST:
    SALT = "TEST_SALT_2026_"

    @pytest.fixture
    def forbidden_store(self):
        # Synthetic sample ground truth
        mock_gt = [{
            "invoice_number": "SYNTH-04531",
            "seller_name": "ACME INDUSTRIAL SUPPLIES LTD",
            "seller_gstin": "27AAPFU0939F1ZV",
            "total_amount": "54421.00",
            "items": [
                {"description": "PRECISION LATHE CHUCK", "amount": "25000.00"}
            ]
        }]
        store = LeakageScreening.build_runtime_forbidden_store(mock_gt, salt=self.SALT)
        return store

    def test_caught_partial_vendor_name_ngram(self, forbidden_store):
        """Adversarial Case 1: 8-gram of vendor name ('acmeindu') caught via char-8-gram hash."""
        code = "if 'acmeindu' in raw_text: return True"
        passed, findings = LeakageScreening.audit_code_changes(
            code,
            salt=forbidden_store["salt"],
            forbidden_hashes=set(forbidden_store["hashes"]),
            forbidden_ngrams=set(forbidden_store["ngrams"])
        )
        assert not passed
        assert any("DATA LEAKAGE DETECTED" in f for f in findings)

    def test_caught_concatenated_literal(self, forbidden_store):
        """Adversarial Case 3: Constant-folded concatenation 'SYNTH-' + '04531' caught."""
        code = "inv_target = 'SYNTH-' + '04531'"
        passed, findings = LeakageScreening.audit_code_changes(
            code,
            salt=forbidden_store["salt"],
            forbidden_hashes=set(forbidden_store["hashes"]),
            forbidden_ngrams=set(forbidden_store["ngrams"])
        )
        assert not passed
        assert any("DATA LEAKAGE DETECTED" in f for f in findings)

    def test_caught_fstring_part(self, forbidden_store):
        """Adversarial Case: f-string part containing sensitive token."""
        code = "pattern = f'SYNTH-04531_{suffix}'"
        passed, findings = LeakageScreening.audit_code_changes(
            code,
            salt=forbidden_store["salt"],
            forbidden_hashes=set(forbidden_store["hashes"]),
            forbidden_ngrams=set(forbidden_store["ngrams"])
        )
        assert not passed
        assert any("DATA LEAKAGE DETECTED" in f for f in findings)

    def test_caught_regex_compile_forbidden(self, forbidden_store):
        """Adversarial Case 4: Regex argument containing sensitive token."""
        code = "rgx = re.compile(r'ACME INDUSTRIAL SUPPLIES.*')"
        passed, findings = LeakageScreening.audit_code_changes(
            code,
            salt=forbidden_store["salt"],
            forbidden_hashes=set(forbidden_store["hashes"]),
            forbidden_ngrams=set(forbidden_store["ngrams"])
        )
        assert not passed
        assert any("DATA LEAKAGE DETECTED" in f for f in findings)

    def test_caught_bare_number(self, forbidden_store):
        """Adversarial Case 5: Bare amount number 54421.00 from forbidden set."""
        code = "if abs(amt - 54421.00) < 0.01: pass"
        passed, findings = LeakageScreening.audit_code_changes(
            code,
            salt=forbidden_store["salt"],
            forbidden_hashes=set(forbidden_store["hashes"]),
            forbidden_ngrams=set(forbidden_store["ngrams"])
        )
        assert not passed
        assert any("DATA LEAKAGE DETECTED" in f for f in findings)

    def test_caught_box_coordinate_comparison(self):
        """Adversarial Case 6: Missed case 'box[1] > 840' must be caught!"""
        code = "if box[1] > 840:\n    return 'footer'"
        passed, findings = LeakageScreening.audit_code_changes(code)
        assert not passed
        assert any("COORDINATE OVERFITTING DETECTED" in f for f in findings)

    def test_caught_vendor_conditional_branch(self):
        """Vendor-conditional gating on specific string literal >= 6 chars."""
        code = "if vendor_name == 'UniqueCustomVendor':\n    parse_custom_layout()"
        passed, findings = LeakageScreening.audit_code_changes(code)
        assert not passed
        assert any("VENDOR-CONDITIONAL BRANCH DETECTED" in f for f in findings)

    def test_benign_color_mask_not_flagged(self):
        """Benign Case 1: [255, 255, 255, 255] color mask should NOT be flagged."""
        code = "mask = [255, 255, 255, 255]"
        passed, findings = LeakageScreening.audit_code_changes(code)
        assert passed, f"Benign color mask was falsely flagged: {findings}"

    def test_benign_cv2_resize_not_flagged(self):
        """Benign Case 2: cv2.resize(img, (1000, 1000)) should NOT be flagged."""
        code = "resized = cv2.resize(img, (1000, 1000))"
        passed, findings = LeakageScreening.audit_code_changes(code)
        assert passed, f"Benign cv2.resize was falsely flagged: {findings}"

    def test_benign_relative_coordinates_not_flagged(self):
        """Benign Case: Ratio-based coordinate logic (e.g. ymin > height * 0.35)."""
        code = "if ymin > page_height * 0.35:\n    return 'lower_band'"
        passed, findings = LeakageScreening.audit_code_changes(code)
        assert passed, f"Relative ratio was falsely flagged: {findings}"

    def test_diff_filtering_scans_only_added_lines(self):
        """Diff parser tests that removed (-) or context lines are not flagged."""
        diff = """
diff --git a/app/ocr/pipeline.py b/app/ocr/pipeline.py
--- a/app/ocr/pipeline.py
+++ b/app/ocr/pipeline.py
@@ -10,3 +10,3 @@
-if box[1] > 840:
+if ymin > page_height * 0.5:
     return True
"""
        passed, findings = LeakageScreening.audit_code_changes(diff)
        assert passed, f"Diff added line was clean but flagged: {findings}"

    def test_b4_dev_vs_selection_gain_comparison(self):
        """B4: Selection CI lower bound <= 0 or wide gap rejected."""
        # Case A: Selection CI lower bound <= 0 -> Reject
        passed_a, msg_a = LeakageScreening.compare_dev_vs_selection_gain(
            dev_gain=0.04, selection_gain=0.02, selection_ci_lower=-0.005
        )
        assert not passed_a
        assert "CI lower bound" in msg_a

        # Case B: Gap > 0.05 -> Reject
        passed_b, msg_b = LeakageScreening.compare_dev_vs_selection_gain(
            dev_gain=0.10, selection_gain=0.03, selection_ci_lower=0.01, max_gap=0.05
        )
        assert not passed_b
        assert "Dev-selection gap" in msg_b

        # Case C: Valid generalization (CI > 0, gap <= 0.05) -> Accept
        passed_c, msg_c = LeakageScreening.compare_dev_vs_selection_gain(
            dev_gain=0.04, selection_gain=0.03, selection_ci_lower=0.008, max_gap=0.05
        )
        assert passed_c

    def test_b5_pruning_analyzer_insufficient_evidence(self):
        """B5: Below 40 docs or 10 templates, returns INSUFFICIENT_EVIDENCE."""
        rule_counts = {"rule_gst_footer": 0, "rule_hsn_prefix": 15}
        res = PruningAnalyzer.audit_rule_ablation_candidates(
            rule_hit_counts=rule_counts,
            corpus_doc_count=25,
            corpus_template_count=8,
            min_docs=40,
            min_templates=10
        )
        assert res["status"] == "INSUFFICIENT_EVIDENCE"
        assert len(res["candidates"]) == 0

    def test_b5_pruning_analyzer_candidates_identified(self):
        """B5: When corpus is sufficient, zero-hit rules identified as ablation candidates."""
        rule_counts = {"rule_gst_footer": 0, "rule_hsn_prefix": 15}
        res = PruningAnalyzer.audit_rule_ablation_candidates(
            rule_hit_counts=rule_counts,
            corpus_doc_count=45,
            corpus_template_count=12,
            min_docs=40,
            min_templates=10
        )
        assert res["status"] == "CANDIDATES_IDENTIFIED"
        assert res["candidates"] == ["rule_gst_footer"]

    def test_b5_ablation_acceptance_evaluation(self):
        """B5: Ablation requires within-noise score change and cost drop."""
        # 1. Score dropped beyond noise -> Reject
        passed_1, _ = PruningAnalyzer.evaluate_ablation_acceptance(
            delta_accuracy=-2.5, noise_sd=0.5, cost_delta=-0.3
        )
        assert not passed_1

        # 2. Cost did not drop -> Reject
        passed_2, _ = PruningAnalyzer.evaluate_ablation_acceptance(
            delta_accuracy=0.1, noise_sd=0.5, cost_delta=0.0
        )
        assert not passed_2

        # 3. Within noise and cost dropped -> Accept
        passed_3, _ = PruningAnalyzer.evaluate_ablation_acceptance(
            delta_accuracy=-0.2, noise_sd=0.5, cost_delta=-0.4
        )
        assert passed_3
