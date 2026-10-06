# tests/test_data_access_and_manifest.py
"""
Tests for DataAccessGate (Phase 1.4):
- Gating locked_test split with RRSI_FINAL_EVAL=1
- Refusing locked_test when RRSI_FINAL_EVAL is not set
- Manifest integrity validation:
  1. Split overlap detection
  2. Template straddling detection
  3. Contaminated invoice detection in selection/test
  4. Copy / page split consistency
- Real repository manifest verification
"""

import os
import json
import pytest
from pathlib import Path
from app.rrsi.data_access import DataAccessGate


class TestDataAccessGate:

    @pytest.fixture
    def mock_manifest_file(self, tmp_path):
        manifest_data = {
            "version": "1.0",
            "manifest": [
                {
                    "doc_id": "INV_DEV_01",
                    "template_group": "T01",
                    "split": "dev",
                    "contaminated": True,
                    "files": ["dev_01.png"]
                },
                {
                    "doc_id": "INV_SEL_01",
                    "template_group": "T02",
                    "split": "selection",
                    "contaminated": False,
                    "files": ["sel_01.png"]
                },
                {
                    "doc_id": "INV_TEST_01",
                    "template_group": "T03",
                    "split": "locked_test",
                    "contaminated": False,
                    "files": ["test_01.png"]
                }
            ]
        }
        man_path = tmp_path / "mock_manifest.json"
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)
        return man_path

    def test_locked_test_refused_without_env_var(self, mock_manifest_file, monkeypatch):
        monkeypatch.delenv("RRSI_FINAL_EVAL", raising=False)
        gate = DataAccessGate(mock_manifest_file)

        # dev and selection work normally
        dev_invoices = gate.get_invoices_for_split("dev")
        assert len(dev_invoices) == 1
        assert dev_invoices[0]["doc_id"] == "INV_DEV_01"

        sel_invoices = gate.get_invoices_for_split("selection")
        assert len(sel_invoices) == 1
        assert sel_invoices[0]["doc_id"] == "INV_SEL_01"

        # locked_test MUST raise PermissionError
        with pytest.raises(PermissionError, match="ACCESS DENIED: 'locked_test' split is locked"):
            gate.get_invoices_for_split("locked_test")

    def test_locked_test_allowed_with_env_var(self, mock_manifest_file, monkeypatch):
        monkeypatch.setenv("RRSI_FINAL_EVAL", "1")
        gate = DataAccessGate(mock_manifest_file)

        test_invoices = gate.get_invoices_for_split("locked_test")
        assert len(test_invoices) == 1
        assert test_invoices[0]["doc_id"] == "INV_TEST_01"

    def test_template_straddling_detected(self, tmp_path):
        bad_manifest = {
            "manifest": [
                {"doc_id": "INV_1", "template_group": "T01", "split": "dev", "files": ["1.png"]},
                {"doc_id": "INV_2", "template_group": "T01", "split": "selection", "files": ["2.png"]}
            ]
        }
        p = tmp_path / "straddle.json"
        p.write_text(json.dumps(bad_manifest), encoding="utf-8")

        gate = DataAccessGate(p)
        is_valid, errors = gate.verify_manifest_integrity()
        assert not is_valid
        assert any("Template straddling" in e for e in errors)

    def test_split_overlap_detected(self, tmp_path):
        bad_manifest = {
            "manifest": [
                {"doc_id": "INV_1", "template_group": "T01", "split": "dev", "files": ["1.png"]},
                {"doc_id": "INV_1", "template_group": "T01", "split": "selection", "files": ["1.png"]}
            ]
        }
        p = tmp_path / "overlap.json"
        p.write_text(json.dumps(bad_manifest), encoding="utf-8")

        gate = DataAccessGate(p)
        is_valid, errors = gate.verify_manifest_integrity()
        assert not is_valid
        assert any("Split overlap" in e for e in errors)

    def test_contaminated_invoice_in_selection_detected(self, tmp_path):
        bad_manifest = {
            "manifest": [
                {
                    "doc_id": "INV_BAD",
                    "template_group": "T02",
                    "split": "selection",
                    "contaminated": True,
                    "files": ["bad.png"]
                }
            ]
        }
        p = tmp_path / "contam.json"
        p.write_text(json.dumps(bad_manifest), encoding="utf-8")

        gate = DataAccessGate(p)
        is_valid, errors = gate.verify_manifest_integrity()
        assert not is_valid
        assert any("Contaminated invoice" in e for e in errors)

        # Also get_invoices_for_split should reject it
        with pytest.raises(ValueError, match="INTEGRITY VIOLATION"):
            gate.get_invoices_for_split("selection")

    def test_real_workspace_manifest_integrity(self):
        """Verifies the actual tracked manifest in the repository."""
        manifest_path = Path("data/manifest.json")
        if not manifest_path.exists():
            pytest.skip("data/manifest.json not found")

        gate = DataAccessGate(manifest_path)
        is_valid, errors = gate.verify_manifest_integrity()
        assert is_valid, f"Repository manifest integrity errors: {errors}"
