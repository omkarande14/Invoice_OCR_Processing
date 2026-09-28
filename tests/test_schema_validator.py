# tests/test_schema_validator.py

import json
from pathlib import Path
import pytest
from app.ocr.schema_validator import SchemaValidator, HEADER_KEYS_SPEC, ITEM_KEYS_SPEC

def test_schema_validator_spec_lengths():
    assert len(HEADER_KEYS_SPEC) == 24  # 23 header keys + 'items'
    assert len(ITEM_KEYS_SPEC) == 26    # 26 line item keys
    assert HEADER_KEYS_SPEC[-1] == "items"

def test_schema_validator_on_synthetic_valid_and_invalid():
    validator = SchemaValidator()

    # Perfect template object
    valid_obj = {k: None for k in HEADER_KEYS_SPEC}
    valid_obj["items"] = [{k: None for k in ITEM_KEYS_SPEC}]

    res = validator.validate(valid_obj)
    assert res["valid"] is True
    assert res["status"] == "PASS"
    assert len(res["errors"]) == 0

    # Missing header key
    invalid_header = valid_obj.copy()
    del invalid_header["customer_gstin"]
    res_inv = validator.validate(invalid_header)
    assert res_inv["valid"] is False
    assert res_inv["status"] == "FAIL"
    assert "customer_gstin" in res_inv["header_conformance"]["missing"]

    # Missing item key
    invalid_item = {k: None for k in HEADER_KEYS_SPEC}
    item_dict = {k: None for k in ITEM_KEYS_SPEC}
    del item_dict["hsn_sac"]
    invalid_item["items"] = [item_dict]
    res_item_inv = validator.validate(invalid_item)
    assert res_item_inv["valid"] is False
    assert any("hsn_sac" in e for e in res_item_inv["errors"])

def test_schema_validator_on_all_official_outputs():
    validator = SchemaValidator()
    files = [
        "data/ocr_results/invoice_4531_official.json",
        "data/ocr_results/kundan_official.json",
        "data/ocr_results/196_pune_official.json",
        "data/ocr_results/cc1415_official.json"
    ]

    existing_files = [Path(f) for f in files if Path(f).exists()]
    if not existing_files:
        pytest.skip("Private OCR official output results not present on fresh clone.")

    for f_path in existing_files:
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = validator.validate(data)
        assert report["valid"] is True, f"Failed structural validation on {f_path}: {report['errors']}"
