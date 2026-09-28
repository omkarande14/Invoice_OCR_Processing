# RRSI Phase 0 Technical Audit & Verification Report

**Date**: 2026-09-28  
**Component**: Reflective Recursive Self-Improvement (RRSI) Framework  
**Status**: PHASE 0 COMPLETE — GATE AUDITED

---

## 1. Audit Target 0.1 — Live Pipeline Execution vs. Synthetic Static Comparison

### The Issue Identified
Prior to this audit, `run_rrsi_evaluation.py --audit` did **not** execute the real extraction pipeline on source images. Instead, it:
1. Attempted to read pre-baked, static JSON files located in `data/ocr_results/`.
2. If no pre-baked file was found, it fell back to:
   ```python
   if not matched:
       predictions.append(gt)  # Evaluated Ground Truth against itself!
   ```
3. Loaded `data/ground_truth/pdf_ground_truth_4531.json`, which was a duplicate schema representation with non-standard keys (`vendor`, `buyer`, `line_items`), causing silent evaluation misses.

### Deconstruction of the "86.67%" Overall Accuracy
The previously reported **86.67%** was an arithmetic illusion resulting from:
$$\text{Overall Accuracy} = 0.40 \times \text{Field Accuracy} + 0.40 \times (\text{Table F1} \times 100) + 0.20 \times \text{Math Accuracy}$$

Because ground-truth files were compared against static outputs or themselves:
- Field Accuracy = $100.00\%$
- Table F1-Score = $1.000$ ($100.00\%$)
- Math Accuracy = $33.33\%$ (1 of 3 files passed the naive equation)
$$\text{Overall Accuracy} = 0.40(100) + 0.40(100) + 0.20(33.33) = 40 + 40 + 6.67 = 86.67\%$$

### The Fix
`run_rrsi_evaluation.py` has been refactored so that `--audit`:
1. Maps each canonical ground truth annotation to its real source image:
   - `invoice_4531_expected.json` $\rightarrow$ `data/test_uploads/staging/deskewed/invoice_4531 (1)_page_1_oriented_deskewed.png`
   - `kundan_expected.json` $\rightarrow$ `data/test_uploads/staging/deskewed/264882_SB_DEENANATH MEDICAL STORES_20_08_2026_page_1_oriented_deskewed_customer_copy.png`
2. Instantiates `InvoicePipeline(use_gpu=False)` and executes `pipeline.process_invoice(img_path)` live on each raw image.
3. Evaluates fresh predictions directly against ground-truth annotations and records true execution latency.

### Live Baseline Results (Real Pipeline on Real Source Images)
```text
Total Invoices Processed: 2
Overall Accuracy Score:   72.56%
Field Accuracy:           94.74%
Table F1-Score:           0.867
Arithmetic Consistency:   0.00%
Total Execution Time:     48.18s (24.09s / invoice on CPU)
Calculated Noise Floor:   +/-30.92% (due to small N=2 sample size)
```

The true baseline accuracy is **72.56%**, not 86.67%. The live run immediately exposed real pipeline extraction errors that were previously masked.

---

## 2. Audit Target 0.2 — Diagnosis of `totals_arithmetic_error` Cases

In the initial audit, 2 out of 3 invoices failed arithmetic checks. Both cases were investigated mathematically to classify whether the failure was an **evaluator flaw**, a **pipeline extraction error**, or a **source-document inconsistency**.

### Case 1: `invoice_4531` (`WMP/B/04531`)
* **Physical Invoice Data**:
  - Item 1 (ALBUCEL): Qty 60 @ Rs. 3,700.00 = Gross Rs. 222,000.00. Trade discount 2% (-Rs. 4,440.00) $\rightarrow$ Taxable Rs. 217,560.00. GST 5% = Rs. 10,878.00.
  - Item 2 (TELIPSA): Qty 120 @ Rs. 198.00 = Gross Rs. 23,760.00. Trade discount 5% (-Rs. 1,188.00) $\rightarrow$ Taxable Rs. 22,572.00. GST 5% = Rs. 1,128.60.
  - Total Taxable Value: $\text{Rs. } 217,560.00 + \text{Rs. } 22,572.00 = \text{Rs. } 240,132.00$
  - Total GST: $\text{Rs. } 10,878.00 + \text{Rs. } 1,128.60 = \text{Rs. } 12,006.60$
  - Subtotal + GST: $\text{Rs. } 240,132.00 + \text{Rs. } 12,006.60 = \text{Rs. } 252,138.60$
  - Round-Off: $+\text{Rs. } 0.40$
  - Invoice Grand Total: $\mathbf{\text{Rs. } 252,139.00}$
* **Diagnosis**:
  - The source invoice is **100% mathematically consistent**.
  - The evaluator failed because it naively checked $\sum \text{Gross Items} + \text{GST} = \text{Total}$ ($\text{Rs. } 245,760.00 + \text{Rs. } 12,006.60 = \text{Rs. } 257,766.60 \neq \text{Rs. } 252,139.00$), ignoring line-item discounts and round-off.
* **Classification**: **Evaluator Defect (Missing Discount & Round-Off Handling)**.

### Case 2: `kundan` (`KP26264882`)
* **Physical Invoice Data**:
  - 14 Line items, each with individual trade discounts (e.g. 4.00%).
  - Gross line items sum: Rs. 53,542.60.
  - Taxable discounted sum: Rs. 51,400.90.
  - Total GST: Rs. 3,020.08.
  - Taxable + GST: $\text{Rs. } 51,400.90 + \text{Rs. } 3,020.08 = \text{Rs. } 54,420.98$.
  - Round-Off: $+\text{Rs. } 0.02$.
  - Invoice Grand Total: $\mathbf{\text{Rs. } 54,421.00}$.
* **Pipeline Extraction Output**:
  - Pipeline extracted `total_gst_amount = 2396.94` (missed GST tax buckets on CPU line extraction) instead of `3020.08`.
  - Pipeline missed 1 table line item (`ATOCOR 40 (15 TAB)`).
* **Diagnosis**:
  - The source invoice is **100% mathematically consistent**.
  - The pipeline produced a **real extraction error** on GST amount (Rs. 2,396.94 vs Rs. 3,020.08) and missed a line item.
  - The evaluator also lacked round-off and discount awareness.
* **Classification**: **Pipeline Extraction Error + Evaluator Missing Discount Handling**.

### Evaluator Mathematical Hardening
`StatisticalEvaluator` in `app/rrsi/evaluator.py` has been updated with:
1. **Configurable Tolerance** (`tolerance: float = 1.0`, adjustable per currency/precision).
2. **Round-Off Awareness**: Checks both $\text{Taxable} + \text{GST} + \text{Roundoff}$ and $\text{Taxable} + \text{GST} - \text{Roundoff}$ against Grand Total.
3. **Item Discount & Taxable Value Derivation**: Derives line-item taxable values using `total_price * (1 - Discount/100)`.
4. **Error Decoupling**:
   - `document_inconsistent`: Triggered when ground-truth arithmetic itself fails (printed document numbers do not add up).
   - `totals_arithmetic_error`: Triggered when the pipeline extracts values that do not reconcile, while the ground truth is consistent.

---

## 3. Audit Target 0.3 — Environment Isolation & Provenance Tracking

### The Contamination Identified
Running `python run_rrsi_evaluation.py --demo` previously wrote directly to:
- `data/rrsi/proposal_memory.json`
- `data/rrsi/experiment_ledger.json`

This injected a fake proposal (`PROP-2026-001`) into the production registry and caused false stagnation counters in `entity_extraction` and `table_extraction`.

### Remediations Applied
1. **Ledger & Memory Purge**:
   `data/rrsi/proposal_memory.json` and `data/rrsi/experiment_ledger.json` were reset to clean, empty states.
2. **Demo Isolation via Tempfile**:
   `run_rrsi_evaluation.py --demo` now executes inside a `tempfile.TemporaryDirectory()`, instantiating isolated registry and ledger instances that are discarded after the demonstration. `data/rrsi/` is never modified during demo runs or tests.
3. **Provenance Tracking**:
   Added a mandatory `provenance` field (`"real"` vs `"demo"`) across:
   - `ProposalHypothesis` (`app/rrsi/taxonomy.py`)
   - `DecisionRecord` (`app/rrsi/taxonomy.py`)
   - Ledger entries (`app/rrsi/ledger.py`)
   - Proposal memory records (`app/rrsi/memory.py`)

---

## 4. Audit Target 0.4 — Inventory of Hardcoded Heuristics, Coordinates & Vendor Strings in `app/`

A comprehensive static audit was performed across all source files in `app/` to inventory hardcoded constants, vendor names, and coordinate checks.

| Location | Hardcoded Value / Pattern | Context / Risk | Generalization Hazard |
| :--- | :--- | :--- | :--- |
| `app/ocr/invoice_pipeline.py:1001` | `"[REDACTED_BUYER_PAN]" in buyer_gstin` | Hardcoded PAN of specific buyer | Fails when entity is seller or on any other entity |
| `app/ocr/invoice_pipeline.py:1010` | `parties["buyer_name"] = "[REDACTED_BUYER_NAME]"` | Static override of buyer name when prefix detected | Hardcodes specific buyer entity into generic OCR pipeline |
| `app/ocr/invoice_pipeline.py:1011` | `buyer_gstin == "[REDACTED_BUYER_GSTIN]"` | Hardcoded 15-char GSTIN string | Overfits to single buyer entity |
| `app/ocr/self_healing.py:198` | `xmin > 550 or ymin > 300` | Fixed pixel coordinate thresholds for buyer GSTIN recovery | Breaks on 300 DPI, mobile scans (4096px), or varying layouts |
| `app/ocr/self_healing.py:201` | `xmin <= 550` | Fixed pixel coordinate threshold for seller GSTIN recovery | Fails on left-aligned buyer layouts or multi-column invoices |
| `app/ocr/self_healing.py:207` | `"[REDACTED_BUYER_PAN]" in buyer["gstin"]` | Hardcoded PAN check to disambiguate identical buyer/seller GSTIN | Fails on any customer other than hardcoded entity |
| `app/ocr/table_extractor.py:64-66` | `"[REDACTED_VENDOR]", "strives for", "strives", "custgihe", "po/phrmcy"` | Hardcoded vendor slogans and customer tags in `TABLE_STOP_KEYWORDS` | Fragile heuristic: vendor-specific slogans used as table boundary stops |
| `app/ocr/table_extractor.py:234-246` | `first_col_xmin > 220.0`, `first_col_xmin > 350.0`, `xmin: 50.0, xmin: 220.0` | Fixed pixel constants to synthesize missing description columns | Hardcoded to specific template geometry; breaks on other dimensions |
| `app/ocr/invoice_pipeline.py:105, 153, 188, 255` | `o_xmin - xmax <= 400`, `<= 350`, `<= 300`, `o_ymin - ymax <= 60` | Fixed pixel proximity thresholds for spatial key-value association | Fails on high-resolution camera captures where word spacing exceeds 400px |
| `app/ui/workbench.py:282` | `if "4531" in selected_file_path.name` | Hardcoded filename match for workbench ground-truth comparison | UI demo cheat that must not exist in production extraction |

---

## 5. Phase 0 Gate Conclusion

1. **Comparison Authenticity**: Confirmed that prior 86.67% was an artifact of static JSON self-comparison. The live pipeline execution baseline on real images has been established at **72.56% Overall Accuracy** (94.74% Field Accuracy, 0.867 Table F1, 0.0% Math Consistency).
2. **Arithmetic Diagnosis**: Concluded both test invoices (`invoice_4531` and `kundan`) are internally consistent. Failures were caused by naive evaluator discount/roundoff omissions, compounded by a live pipeline GST extraction error on Kundan. Configurable tolerance and round-off logic have been implemented.
3. **Isolation Enforced**: Demo and test runs are strictly sandboxed via `tempfile`, persistent registries have been purged, and `provenance` metadata is enforced.
4. **Hardcoding Documented**: Complete inventory of 10 hardcoded points cataloged above, serving as targets for Phase 5 leakage elimination and Phase 7 refactoring.

**Ready for Phase 1 (Dataset Manifest, Template Grouping & Anonymization).**
