# streamlit_app.py
"""
Entrypoint for the Invoice OCR Testing & Verification Workbench.
Run with:
    streamlit run streamlit_app.py
"""

import html
import json
import os
import sys
from pathlib import Path

# Ensure project root is the primary entry in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Configure GPU DLL paths so Paddle can find cuDNN v8 on Windows
from app.ocr.dual_engine_benchmarker import configure_gpu_dll_paths
configure_gpu_dll_paths()

import streamlit as st
from PIL import Image

from app.ocr.invoice_pipeline import InvoicePipeline
from app.ocr.stage_runner import StageByStageRunner
from app.ocr.vision_fallback import GeminiVisionFallback

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Invoice OCR",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cached Pipeline Instance (st.cache_resource)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing OCR Engines (PaddleOCR + Tesseract)...")
def get_cached_pipeline() -> InvoicePipeline:
    # High-stability mode for demo (eliminates CUDA memory aborts on large/multi-page files)
    return InvoicePipeline(use_gpu=False)

pipeline = get_cached_pipeline()
runner = StageByStageRunner(pipeline=pipeline, staging_dir=PROJECT_ROOT / "data" / "test_uploads" / "staging")

# -----------------------------------------------------------------------------
# Staging Directories Setup
# -----------------------------------------------------------------------------
UPLOAD_DIR = PROJECT_ROOT / "data" / "test_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Header: Clean & Simple for Demo
# -----------------------------------------------------------------------------
st.title("Invoice OCR")

# -----------------------------------------------------------------------------
# Sidebar: Upload & Source Selection (Kept Exactly as is)
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
    st.header("2. AI Intelligence & Fallback")
    st.success("⚡ **Self-Healing Engine**: Active\n\n*(Ledger Invariants & Cross-Field Deduction)*")

    enable_vision = st.checkbox(
        "Enable Vision-AI (Gemini Flash) Fallback",
        value=True,
        help="Automatically queries Gemini Flash if local OCR misses critical fields or encounters complex layout anomalies."
    )
    custom_gemini_key = st.text_input(
        "Gemini API Key (Optional):",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Paste your Gemini API key to enable zero-shot Vision-AI fallback on unseen invoice layouts."
    )

    st.markdown("---")
    st.markdown("### System Diagnostics")
    st.info("**OCR Engine**: PaddleOCR + Tesseract\n\n**Mode**: High-Stability Production Mode\n\n**Status**: 🟢 Ready")
    if st.button("🔄 Clear Cache & Reload Pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# Main Screen: Stage-by-Stage Runner & Side-by-Side Results
# -----------------------------------------------------------------------------
if selected_file_path is None:
    st.info("👈 Please upload a PDF/image or select a sample invoice from the sidebar to begin.")
else:
    # If a new or different file is selected, clear any stale results from the previous file
    current_file_key = str(selected_file_path)
    if st.session_state.get("processed_file_key") != current_file_key:
        st.session_state.pop("latest_result", None)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("🚀 Run Pipeline End-to-End", type="primary", use_container_width=True)
    with col_info:
        st.write(f"Target file: `{selected_file_path.name}` ({selected_file_path.stat().st_size / 1024:.1f} KB)")

    # Execute pipeline when triggered
    if run_clicked:
        st.markdown("### Stage-by-Stage Execution")
        
        with st.status("Executing Invoice Pipeline...", expanded=True) as status_box:
            
            def on_progress(stage_name: str, stage_status: str, details: dict):
                stage_labels = {
                    "1_validation": "Stage 1: Input Validation (Milestone 1)",
                    "2_conversion": "Stage 2: Format Conversion (300 DPI PNG)",
                    "3_similarity": "Stage 3: Deduplication & Perceptual Hashing",
                    "4_orientation": "Stage 4: Orientation Correction (Tesseract OSD)",
                    "5_deskew": "Stage 5: Deskewing & Dual-Copy Selection",
                    "6_ocr": "Stage 6: Dual-Engine OCR (GPU Paddle + CPU Tesseract)",
                    "7_extraction": "Stage 7: Metadata & Table Line-Item Extraction",
                    "8_verification": "Stage 8: Arithmetic Verification & Schema Mapping"
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

            # Run all stages with fault tolerance
            active_fallback = GeminiVisionFallback(api_key=custom_gemini_key) if (enable_vision and custom_gemini_key) else None
            runner.vision_fallback = active_fallback
            runner.enable_vision_fallback = enable_vision

            try:
                result = runner.run_stages(selected_file_path, progress_callback=on_progress)
            except Exception as e:
                st.warning(f"Initial run encountered an issue ({e}). Retrying with safe CPU execution...")
                try:
                    cpu_pipeline = InvoicePipeline(use_gpu=False)
                    cpu_runner = StageByStageRunner(
                        pipeline=cpu_pipeline,
                        staging_dir=PROJECT_ROOT / "data" / "test_uploads" / "staging",
                        vision_fallback=active_fallback,
                        enable_vision_fallback=enable_vision
                    )
                    result = cpu_runner.run_stages(selected_file_path, progress_callback=on_progress)
                except Exception as fatal_e:
                    result = {
                        "success": False,
                        "failed_stage": "Pipeline Execution",
                        "error": str(fatal_e),
                        "total_duration_seconds": 0.0
                    }

            if result.get("success", False):
                total_dur = result.get("total_duration_seconds", 0.0)
                status_box.update(
                    label=f"Pipeline Completed Successfully in {total_dur}s!",
                    state="complete",
                    expanded=False
                )
                st.session_state["latest_result"] = result
                st.session_state["processed_file_key"] = current_file_key
            else:
                status_box.update(
                    label=f"Pipeline Failed at {result.get('failed_stage', 'Unknown Stage')}",
                    state="error",
                    expanded=True
                )
                st.error(f"Execution failed: {result.get('error')}")
                st.session_state.pop("latest_result", None)
                st.session_state.pop("processed_file_key", None)

    # Display side-by-side results if present in session state
    if "latest_result" in st.session_state:
        res = st.session_state["latest_result"]
        data = res.get("final_json", {})
        
        st.markdown("---")

        # Side-by-side View: Document Preview (Left) + Formatted JSON (Right)
        img_col, json_col = st.columns([1, 1], gap="medium")

        with img_col:
            st.markdown("### 📄 Document Preview")
            img_path = res.get("page1_image_path")
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True, caption=f"Processed Image: {Path(img_path).name}")
            else:
                st.info("No processed preview image available.")

        with json_col:
            st.markdown("### 🧾 Extracted JSON Output")
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            escaped_json = html.escape(json_str)

            # Copy button + Download toolbar
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                st.components.v1.html(f"""
                <div style="display: flex; align-items: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; height: 38px;">
                    <button id="copy-json-btn" style="
                        background-color: #2563eb;
                        color: #ffffff;
                        border: none;
                        padding: 7px 16px;
                        font-size: 13px;
                        font-weight: 600;
                        border-radius: 6px;
                        cursor: pointer;
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        outline: none;
                        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
                    " onmouseover="this.style.backgroundColor='#1d4ed8'" onmouseout="this.style.backgroundColor='#2563eb'">
                        📋 Copy JSON
                    </button>
                    <span id="copy-success-badge" style="
                        display: none;
                        margin-left: 10px;
                        color: #15803d;
                        font-size: 13px;
                        font-weight: 600;
                    ">✅ Copied to clipboard!</span>
                </div>
                <textarea id="hidden-json-payload" style="position: absolute; left: -9999px; top: -9999px;">{escaped_json}</textarea>
                <script>
                document.getElementById('copy-json-btn').addEventListener('click', function() {{
                    const text = document.getElementById('hidden-json-payload').value;
                    function showSuccess() {{
                        const badge = document.getElementById('copy-success-badge');
                        badge.style.display = 'inline';
                        setTimeout(function() {{ badge.style.display = 'none'; }}, 3000);
                    }}
                    if (navigator.clipboard && window.isSecureContext) {{
                        navigator.clipboard.writeText(text).then(showSuccess).catch(function() {{
                            fallbackCopy();
                        }});
                    }} else {{
                        fallbackCopy();
                    }}
                    function fallbackCopy() {{
                        const area = document.getElementById('hidden-json-payload');
                        area.select();
                        document.execCommand('copy');
                        showSuccess();
                    }}
                }});
                </script>
                """, height=44)

            with btn_col2:
                file_stem = selected_file_path.stem if selected_file_path else "extracted_invoice"
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{file_stem}_official.json",
                    mime="application/json",
                    use_container_width=True
                )

            # Code display block with Streamlit's built-in 1-click copy icon
            st.code(json_str, language="json")
