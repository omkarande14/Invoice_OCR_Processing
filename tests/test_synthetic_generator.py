# tests/test_synthetic_generator.py
"""
Unit tests for SyntheticInvoiceGenerator (Phase 1.6):
- Mathematical integrity: Taxable + GST + Freight + RoundOff == Total
- Deterministic reproducibility given seed
- Correct schema attributes
- PIL image rendering and dimensions
"""

import pytest
from decimal import Decimal
from app.rrsi.synthetic_generator import SyntheticInvoiceGenerator
from app.validators import InvoiceArithmeticValidator, ValidationStatus


class TestSyntheticGenerator:

    def test_arithmetic_consistency(self):
        gen = SyntheticInvoiceGenerator(seed=123)
        validator = InvoiceArithmeticValidator()

        for i in range(1, 10):
            data = gen.generate_invoice_data(invoice_idx=i)
            # Reconcile via official validator
            val_res = validator.validate(data)
            assert val_res.status == ValidationStatus.PASS, f"Synthetic invoice {i} failed arithmetic check: {val_res.error_message}"
            assert val_res.is_valid is True

    def test_seed_determinism(self):
        gen1 = SyntheticInvoiceGenerator(seed=999)
        gen2 = SyntheticInvoiceGenerator(seed=999)

        d1 = gen1.generate_invoice_data(invoice_idx=1)
        d2 = gen2.generate_invoice_data(invoice_idx=1)

        assert d1["invoice_number"] == d2["invoice_number"]
        assert d1["total_amount"] == d2["total_amount"]
        assert len(d1["items"]) == len(d2["items"])

    def test_image_rendering(self, tmp_path):
        gen = SyntheticInvoiceGenerator(seed=42)
        img, data = gen.generate_sample(invoice_idx=5, output_dir=tmp_path, add_noise=True)

        assert img.size == (1200, 1600)
        assert (tmp_path / "synth_0005.png").exists()
        assert (tmp_path / "synth_0005_gt.json").exists()
