# app/ui/workbench.py

import json
import os
import sys
from pathlib import Path

# Ensure project root is the primary entry in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Remove app/ui from sys.path to prevent namespace conflicts
ui_dir = str(Path(__file__).resolve().parent)
while ui_dir in sys.path:
    sys.path.remove(ui_dir)

# Configure GPU DLL paths so Paddle can find cuDNN v8 on Windows
from app.ocr.dual_engine_benchmarker import configure_gpu_dll_paths
configure_gpu_dll_paths()

import streamlit as st
from PIL import Image

import pandas as pd

from app.ocr.invoice_pipeline import InvoicePipeline
from app.ocr.stage_runner import StageByStageRunner
from app.ocr.schema_mapper import SchemaMapper

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Invoice OCR — Testing & Verification Workbench",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cached GPU Pipeline Instance (st.cache_resource)
# -----------------------------------------------------------------------------
# User constraint: use st.cache_resource so PaddleOCR/Tesseract stay in VRAM
# and are never re-initialized on UI re-runs.
@st.cache_resource(show_spinner="Initializing GPU-accelerated OCR Engines (PaddleOCR + Tesseract)...")
def get_cached_pipeline() -> InvoicePipeline:
    return InvoicePipeline(use_gpu=True)

pipeline = get_cached_pipeline()
runner = StageByStageRunner(pipeline=pipeline, staging_dir=PROJECT_ROOT / "data" / "test_uploads" / "staging")

# -----------------------------------------------------------------------------
# Staging Directories Setup (Strict Safety: Never touches data/input/)
# -----------------------------------------------------------------------------
UPLOAD_DIR = PROJECT_ROOT / "data" / "test_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🧾 Invoice OCR — Testing & Verification Workbench")
st.caption("Dual-Engine OCR (Paddle GPU + Tesseract CPU) with Arithmetic Verification & Provenance Tracking")

# -----------------------------------------------------------------------------
# Sidebar: Upload & Source Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Select or Upload Invoice")
    source_mode = st.radio(
        "Input Source:",
        ["Upload New Invoice", "Choose from Test Dataset"],
        index=0
    )

    selected_file_path = None

    if source_mode == "Upload New Invoice":
        uploaded_file = st.file_uploader(
            "Upload an invoice file (PDF, PNG, JPG):",
            type=["pdf", "png", "jpg", "jpeg"],
            help="File is saved to data/test_uploads/ and verified using Milestone 1 rules."
        )
        if uploaded_file is not None:
            # Save file safely to data/test_uploads/
            save_path = UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_file_path = save_path
            st.success(f"Uploaded: `{uploaded_file.name}`")
    else:
        # Existing sample files from dataset
        deskewed_dir = PROJECT_ROOT / "data" / "deskewed"
        sample_files = []
        if deskewed_dir.exists():
            sample_files = sorted([f.name for f in deskewed_dir.glob("*.png")])
        
        if sample_files:
            choice = st.selectbox("Sample Invoice from Dataset:", sample_files, index=0)
            selected_file_path = deskewed_dir / choice
        else:
            st.info("No sample files found in data/deskewed/.")

    st.markdown("---")
    st.markdown("### System Diagnostics")
    device_name = "NVIDIA GeForce RTX 3050 (6GB)" if pipeline.benchmarker.use_gpu else "CPU Mode"
    st.info(f"**OCR Device**: {device_name}\n\n**Framework**: PaddleOCR + Tesseract")

# -----------------------------------------------------------------------------
# Main Screen: Stage-by-Stage Runner
# -----------------------------------------------------------------------------
if selected_file_path is None:
    st.info("👈 Please upload a PDF/image or select a sample invoice from the sidebar to begin.")
else:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("🚀 Run Pipeline End-to-End", type="primary", use_container_width=True)
    with col_info:
        st.write(f"Target file: `{selected_file_path.name}` ({selected_file_path.stat().st_size / 1024:.1f} KB)")

    # Execute pipeline when triggered or if cached in session state
    if run_clicked:
        st.markdown("### 2. Stage-by-Stage Execution")
        
        with st.status("Executing Invoice Pipeline...", expanded=True) as status_box:
            
            def on_progress(stage_name: str, stage_status: str, details: dict):
                stage_labels = {
                    "1_validation": "Stage 1: Input Validation (Milestone 1)",
                    "2_conversion": "Stage 2: Format Conversion (200 DPI Grayscale PNG)",
                    "3_similarity": "Stage 3: Deduplication & Perceptual Hashing",
                    "4_orientation": "Stage 4: Orientation Correction (Tesseract OSD)",
                    "5_deskew": "Stage 5: Deskewing (Hough Line Transform)",
                    "6_ocr": "Stage 6: Dual-Engine OCR (GPU Paddle + CPU Tesseract)",
                    "7_extraction": "Stage 7: Metadata & Table Line-Item Extraction",
                    "8_verification": "Stage 8: Arithmetic Verification & Consensus Audit"
                }
                label = stage_labels.get(stage_name, stage_name)
                
                if stage_status == "running":
                    st.write(f"⏳ **{label}**: {details.get('message', 'Processing...')}")
                elif stage_status == "completed":
                    dur = details.get("duration_seconds", 0.0)
                    extra = []
                    if "format" in details:
                        extra.append(f"Format: {details['format']}, Pages: {details['page_count']}")
                    if "detected_skew_deg" in details:
                        extra.append(f"Skew: {details['detected_skew_deg']}°")
                    if "paddle_gpu_runtime" in details:
                        extra.append(f"GPU OCR: {details['paddle_gpu_runtime']}s")
                    if "line_items_count" in details:
                        extra.append(f"Line Items: {details['line_items_count']}")
                    if "arithmetic_status" in details:
                        extra.append(f"Math: {details['arithmetic_status']}")
                    
                    extra_str = f" ({', '.join(extra)})" if extra else ""
                    st.write(f"✅ **{label}** — {dur}s{extra_str}")
                elif stage_status == "failed":
                    st.write(f"❌ **{label}** — FAILED: {details.get('reason', 'Unknown error')}")

            # Run all stages
            result = runner.run_stages(selected_file_path, progress_callback=on_progress)

            if result["success"]:
                total_dur = result["total_duration_seconds"]
                status_box.update(
                    label=f"Pipeline Completed Successfully in {total_dur}s!",
                    state="complete",
                    expanded=False
                )
                st.session_state["latest_result"] = result
            else:
                status_box.update(
                    label=f"Pipeline Failed at {result.get('failed_stage', 'Unknown Stage')}",
                    state="error",
                    expanded=True
                )
                st.error(f"Execution failed: {result.get('error')}")

    # Display results if present in session state
    if "latest_result" in st.session_state:
        res = st.session_state["latest_result"]
        data = res["final_json"]
        audit = res.get("audit_json", {})
        fin = audit.get("financial_summary", {})
        ver = audit.get("verification", {})
        perf = audit.get("pipeline_performance", {})

        st.markdown("---")
        st.markdown("### 3. Extracted Results & Verification")

        # Top Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Invoice Amount", f"₹ {data.get('invoice_amount', '0.00')}")
        with m2:
            st.metric("Total GST", f"₹ {data.get('total_gst_amount', '0.00')}")
        with m3:
            st.metric("Total Quantity", f"{data.get('total_quantity', 0)}")
        with m4:
            st.metric("Provenance", fin.get("grand_total_source", "verified"))
        with m5:
            st.metric("Dual-Engine Consensus", f"{ver.get('consensus_score_pct', 0):.0f}% ({ver.get('dual_engine_consensus', 'N/A')})")

        # Verification Status Banner
        if ver.get("arithmetic_status") == "MATHEMATICALLY_VERIFIED":
            st.success(f"✅ **Arithmetic Verification Passed**: Line items match subtotal, and subtotal + taxes match grand total.")
        elif ver.get("arithmetic_status") == "SUBTOTAL_VERIFIED":
            st.info(f"ℹ️ **Subtotal Verified**: Line items sum matches subtotal.")
        else:
            st.warning(f"⚠️ **Arithmetic Verification Status**: {ver.get('arithmetic_status')}")

        # Side-by-side View: Document Image + Extracted JSON
        img_col, json_col = st.columns([1, 1])

        with img_col:
            st.markdown("#### Processed Document Image")
            img_path = res.get("page1_image_path")
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True, caption=f"{Path(img_path).name}")
            else:
                st.info("No processed preview image available.")

        with json_col:
            st.markdown("#### Official JSON Deliverable (Team Lead Schema)")
            tabs = st.tabs(["Pretty JSON", "Line Items Table", "Metadata", "Sidecar Diagnostics"])
            
            with tabs[0]:
                st.json(data)
            
            with tabs[1]:
                line_items = data.get("items", []) or data.get("line_items", [])
                if line_items:
                    st.dataframe(line_items, use_container_width=True)
                else:
                    st.info("No line items extracted from this invoice.")

            with tabs[2]:
                st.write("**Seller Name**:", data.get("seller_name"))
                st.write("**Seller GSTIN**:", data.get("seller_gstin"))
                st.write("**Customer Name**:", data.get("customer_name"))
                st.write("**Customer GSTIN**:", data.get("customer_gstin"))
                st.write("**Invoice No**:", data.get("invoice_number"))
                st.write("**Invoice Date**:", data.get("invoice_date"))
                st.write("**Due Date**:", data.get("due_date"))
                st.write("**PO Number**:", data.get("PO_number"))
                st.write("**Currency**:", data.get("currency_code"))

            with tabs[3]:
                st.caption("Verification, consensus, and pipeline performance sidecar audit:")
                st.json(audit)

        # ---------------------------------------------------------------------
        # Section 4: Ground Truth Comparison & Accuracy Benchmark
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 4. Ground Truth Comparison & Accuracy Benchmark")
        st.caption("Compare extracted output against expected ground truth using automated schema mapping.")

        gt_dir = PROJECT_ROOT / "data" / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        available_gt_files = sorted([f.name for f in gt_dir.glob("*.json")])

        gt_col1, gt_col2 = st.columns([1, 2])
        with gt_col1:
            gt_source_choice = st.radio(
                "Ground Truth Source:",
                ["Select from data/ground_truth/", "Upload Ground Truth JSON"],
                index=0
            )

        ground_truth_data = None
        ground_truth_name = None

        with gt_col2:
            if gt_source_choice == "Select from data/ground_truth/":
                if available_gt_files:
                    default_idx = 0
                    stem = selected_file_path.stem.lower()
                    for i, fname in enumerate(available_gt_files):
                        f_low = fname.lower()
                        if stem in f_low or any(part in f_low for part in stem.split("_") if len(part) >= 4):
                            default_idx = i
                            break
                    selected_gt_name = st.selectbox("Ground Truth File:", available_gt_files, index=default_idx)
                    gt_file_path = gt_dir / selected_gt_name
                    with open(gt_file_path, "r", encoding="utf-8") as f:
                        ground_truth_data = json.load(f)
                    ground_truth_name = selected_gt_name
                else:
                    st.info("No ground-truth files found in `data/ground_truth/`.")
            else:
                uploaded_gt = st.file_uploader("Upload Expected Ground Truth (.json):", type=["json"])
                if uploaded_gt is not None:
                    try:
                        ground_truth_data = json.load(uploaded_gt)
                        ground_truth_name = uploaded_gt.name
                        st.success(f"Loaded ground truth: `{uploaded_gt.name}`")
                    except Exception as e:
                        st.error(f"Failed to parse JSON: {e}")

        if ground_truth_data is not None:
            mapper = SchemaMapper()
            audit_report = mapper.compare(ground_truth=ground_truth_data, system_output=data)
            metrics = audit_report["metrics"]

            st.markdown(f"#### 📊 Benchmark Scorecards (`{ground_truth_name}`)")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Overall Accuracy", f"{metrics['overall_accuracy_pct']}%", f"{metrics['overall_matches']} fields")
            with sc2:
                st.metric("Metadata Text Accuracy", f"{metrics['text_accuracy_pct']}%", f"{metrics['text_matches']} fields")
            with sc3:
                st.metric("Financial Numeric Accuracy", f"{metrics['numeric_accuracy_pct']}%", f"{metrics['numeric_matches']} fields")
            with sc4:
                st.metric("Line Items Accuracy", f"{metrics['line_item_accuracy_pct']}%", f"{metrics['line_item_matches']} fields")

            cmp_tab1, cmp_tab2, cmp_tab3 = st.tabs([
                "📋 Metadata & Totals Audit",
                "📦 Line Items Audit",
                "🔍 Raw JSON Diff"
            ])

            with cmp_tab1:
                scalar_rows = []
                for s in audit_report["scalar_fields"]:
                    status_badge = "✅ MATCH" if s["match"] else "❌ MISMATCH"
                    scalar_rows.append({
                        "Field": s["field"],
                        "Type": s["type"].upper(),
                        "Expected (Ground Truth)": str(s["expected"]) if s["expected"] is not None else "null",
                        "Extracted (OCR Pipeline)": str(s["extracted"]) if s["extracted"] is not None else "null",
                        "Status": status_badge
                    })
                df_scalar = pd.DataFrame(scalar_rows)
                st.dataframe(df_scalar, use_container_width=True, hide_index=True)

            with cmp_tab2:
                st.caption("Side-by-side line item evaluation with automated schema mapping and known typo handling.")
                for item_res in audit_report["line_items"]:
                    st.markdown(f"##### Line Item #{item_res['item_index']}")
                    item_rows = []
                    for f in item_res["fields"]:
                        if f.get("is_exempted"):
                            status_badge = "ℹ️ EXEMPTED (Known GT Typo)"
                        elif f["match"]:
                            status_badge = "✅ MATCH"
                        else:
                            status_badge = "❌ MISMATCH"

                        item_rows.append({
                            "Field Name": f["field"],
                            "Canonical Target": f["target_key"],
                            "Type": f["type"].upper(),
                            "Expected Value": str(f["expected"]) if f["expected"] is not None else "null",
                            "Extracted Value": str(f["extracted"]) if f["extracted"] is not None else "null",
                            "Match Status": status_badge
                        })
                    df_item = pd.DataFrame(item_rows)
                    st.dataframe(df_item, use_container_width=True, hide_index=True)

            with cmp_tab3:
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.markdown("**Expected Ground Truth JSON**")
                    st.json(ground_truth_data)
                with d_col2:
                    st.markdown("**Extracted Pipeline JSON**")
                    st.json(data)

            st.download_button(
                label="📥 Download Audit Report (JSON)",
                data=json.dumps(audit_report, indent=2),
                file_name=f"benchmark_report_{selected_file_path.stem}.json",
                mime="application/json"
            )
