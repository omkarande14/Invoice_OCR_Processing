# Invoice OCR System

An automated Invoice Processing and OCR pipeline designed for high-accuracy data extraction from real-world invoice PDFs and scanned images.

---

## 1. Architecture & Pipeline Stages

1. **Input Validation (Milestone 1)** (`app/input/`):
   - Validates existence, formats (`.pdf`, `.jpg`, `.jpeg`, `.png`), corruption, PDF encryption, and page counts.
2. **Image Conversion (Milestone 2)** (`app/preprocessing/image_converter.py`):
   - Renders valid PDF pages and images into standard 200 DPI grayscale PNG files (`<filename>_page_<N>.png`).
3. **Exact Visual Duplicate Detection (Milestone 3a)** (`app/preprocessing/duplicate_detector.py`):
   - Computes SHA-256 hashes of decoded pixel data (`img.tobytes()`).
   - The **only reliable duplicate signal** prior to OCR. Identifies redundant exact copies without modifying original input.
4. **Template Similarity Detection (Milestone 3b)** (`app/preprocessing/template_similarity_detector.py`):
   - Uses perceptual hashing (`pHash`) to cluster images sharing visual layouts/templates.
   - **Critical Architectural Decision:** Evaluated and confirmed that perceptual hashing detects *shared vendor template layouts*, NOT duplicate document content. Different invoices from the same vendor share near-identical layouts (pHash distance 0–4) despite having different invoice numbers, line items, and amounts.
   - **Non-destructive:** Results are strictly informational for downstream template-aware extraction (Milestones 4–6) and must **never** be used to skip or delete files. True duplicate detection for different scans of the same document is deferred until post-OCR content comparison.
5. **Orientation Correction (Milestone 3c)** (`app/preprocessing/orientation_corrector.py`):
   - Uses Tesseract OSD to detect orientation (0°, 90°, 180°, 270°) and restores all images to standard 0° upright in `data/oriented/`.
6. **Deskew Correction (Milestone 3d)** (`app/preprocessing/deskew_corrector.py`):
   - Uses OpenCV Hough Line Transform to detect and correct small tilts ($0.5^\circ$ to $15.0^\circ$) into `data/deskewed/`, preserving pristine sharpness for straight files ($< 0.5^\circ$).
7. **Normalization Evaluation (Milestone 3e)**:
   - Empirically audited contrast/sharpness across the dataset; speculative binarization safely deferred to protect thin character strokes.
8. **OCR Engine Selection & GPU Acceleration (Milestone 4a)** (`app/ocr/`):
   - GPU-accelerated PaddleOCR configured with CUDA 12 and cuDNN v8 targeting NVIDIA GeForce RTX 3050 (6GB VRAM).
   - Massive real-world speedup: high-resolution camera scans (4096 × 3072) dropped from 275s on CPU to ~16s on GPU; standard 200 DPI PDF invoices run in ~3–5 seconds.
9. **Dual-Engine Benchmarking & Consensus Strategy (Milestone 4b)** (`app/ocr/dual_engine_benchmarker.py`):
   - Evaluated PaddleOCR (deep-learning layout/table focus) against Tesseract (independent patch validation) across 3 diverse invoice types.
   - Achieved 100% consensus agreement on clean invoices; PaddleOCR successfully extracted critical data (94% confidence) on mobile camera scans where Tesseract degraded to 67%.
10. **Table & Line-Item Extraction (Milestone 4c)** (`app/ocr/table_extractor.py`):
    - Spatial geometric column mapping and adaptive vertical row clustering.
    - Accurately parses Item Description, HSN/SAC Code, Quantity, Unit, Rate, and Taxable Amount.
    - Automatic row-level arithmetic validation ($\text{Quantity} \times \text{Rate} = \text{Amount}$).
11. **Arithmetic Verifier & End-to-End Pipeline (Milestone 4d)** (`app/ocr/verifier.py`, `app/ocr/invoice_pipeline.py`):
    - Multi-stage mathematical reconciliation:
      - Row Item Check: $\text{Qty} \times \text{Rate} = \text{Amount}$
      - Line Item Sum: $\sum \text{Line Items} = \text{Subtotal}$
      - Tax Reconciliation: $\text{Subtotal} + \text{CGST} + \text{SGST} + \text{IGST} \pm \text{Roundoff} = \text{Grand Total}$
    - 15-character Indian GSTIN format and state code verification.
    - Cross-engine consensus scoring and automated JSON export into `data/ocr_results/`.
12. **Reflective Recursive Self-Improvement (RRSI Framework)** (`app/rrsi/`):
    - **Proposal Side**: Negative Knowledge Registry (`proposal_memory.json`) prevents repeating failed approaches; Area Stagnation monitors consecutive failures and enforces pivoting to orthogonal pipeline areas.
    - **Selection Side**: Leakage Screening audits candidate code for sample memorization and coordinate overfitting; Noise Floor Guard ($\pm \text{SE}$) eliminates lucky fluctuations; Cost-Aware Acceptance rejects latency blowups.
    - **Error Taxonomy & Ledger**: Tracks fine-grained failures (`table_data_mismatch`, `date_extraction_mismatch`, `invoice_number_error`, `party_mismatch`, `totals_arithmetic_error`) in an auditable ledger (`experiment_ledger.json`).

---

## 2. Running the System

```bash
# Run all automated tests (including RRSI unit tests)
python -m pytest tests/ -v

# Run RRSI Baseline Error Taxonomy Audit
python run_rrsi_evaluation.py --audit

# Run RRSI Proposal & Selection Demonstration
python run_rrsi_evaluation.py --demo

# Run Streamlit Web Workbench (Testing & Verification UI)
streamlit run app/ui/workbench.py

# Run complete end-to-end OCR and verification pipeline
python run_invoice_pipeline.py
```

