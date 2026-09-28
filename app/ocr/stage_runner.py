# app/ocr/stage_runner.py

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from app.input.file_validator import validate_file
from app.preprocessing.image_converter import convert_file
from app.preprocessing.duplicate_detector import compute_image_hash
from app.preprocessing.template_similarity_detector import compute_perceptual_hash
from app.preprocessing.orientation_corrector import detect_and_correct_orientation
from app.preprocessing.deskew_corrector import deskew_image
from app.ocr.invoice_pipeline import InvoicePipeline
from app.ocr.dual_engine_benchmarker import DualEngineBenchmarker
from app.ocr.output_formatter import format_official_output
from app.ocr.vision_fallback import GeminiVisionFallback


class StageByStageRunner:
    """
    Executes the invoice OCR pipeline stage-by-stage with real-time
    observability and intermediate artifact tracking.
    
    Safe Staging Principle:
      All temporary conversions, oriented images, and deskewed outputs
      are saved inside a dedicated staging directory (e.g. data/test_uploads/staging/)
      and never touch data/input/.
    """

    def __init__(
        self,
        pipeline: InvoicePipeline,
        staging_dir: Optional[Path] = None,
        vision_fallback: Optional[Any] = None,
        enable_vision_fallback: bool = True
    ):
        self.pipeline = pipeline
        self.staging_dir = staging_dir or Path("data/test_uploads/staging")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.enable_vision_fallback = enable_vision_fallback
        if vision_fallback is not None:
            self.vision_fallback = vision_fallback
        elif enable_vision_fallback:
            self.vision_fallback = GeminiVisionFallback()
        else:
            self.vision_fallback = None

    def run_stages(
        self,
        input_file: Path,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs all pipeline stages sequentially on an input file.
        
        progress_callback signature:
            callback(stage_name: str, status: str, details: dict)
            status can be: "running", "completed", "failed"
        """
        input_path = Path(input_file).resolve()
        overall_start = time.time()
        stage_reports = {}

        def notify(stage: str, status: str, details: Dict[str, Any]):
            stage_reports[stage] = {"status": status, **details}
            if progress_callback:
                progress_callback(stage, status, details)

        # -------------------------------------------------------------
        # STAGE 1: Input Validation (Milestone 1)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("1_validation", "running", {"message": "Validating file format and integrity..."})
        val_res = validate_file(str(input_path))
        dur_1 = round(time.time() - t0, 3)

        if val_res.get("status") != "valid":
            notify("1_validation", "failed", {
                "duration_seconds": dur_1,
                "reason": val_res.get("reason", "Unknown validation failure")
            })
            return {
                "success": False,
                "failed_stage": "1_validation",
                "stages": stage_reports,
                "error": val_res.get("reason")
            }

        notify("1_validation", "completed", {
            "duration_seconds": dur_1,
            "format": val_res.get("format"),
            "page_count": val_res.get("page_count", 1),
            "filename": val_res.get("filename")
        })

        # -------------------------------------------------------------
        # STAGE 2: Image Conversion (Milestone 2: 200 DPI Grayscale PNG)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("2_conversion", "running", {"message": "Converting to 200 DPI normalized grayscale PNG..."})
        conv_dir = self.staging_dir / "converted"
        conv_dir.mkdir(parents=True, exist_ok=True)

        converted_paths = convert_file(str(input_path), str(conv_dir))
        dur_2 = round(time.time() - t0, 3)

        if not converted_paths:
            notify("2_conversion", "failed", {"duration_seconds": dur_2, "reason": "No pages converted"})
            return {"success": False, "failed_stage": "2_conversion", "stages": stage_reports, "error": "Conversion produced no output"}

        page1_path = Path(converted_paths[0])
        notify("2_conversion", "completed", {
            "duration_seconds": dur_2,
            "pages_converted": len(converted_paths),
            "page1_image": str(page1_path)
        })

        # -------------------------------------------------------------
        # STAGE 3: Duplicate & Template Similarity Check (Milestones 3a & 3b)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("3_similarity", "running", {"message": "Computing pixel hashes and perceptual similarity..."})
        sha256_hash = compute_image_hash(str(page1_path))
        phash = str(compute_perceptual_hash(str(page1_path)))
        dur_3 = round(time.time() - t0, 3)

        notify("3_similarity", "completed", {
            "duration_seconds": dur_3,
            "sha256": sha256_hash[:16] + "...",
            "phash": phash,
            "message": "Visual fingerprints computed"
        })

        # -------------------------------------------------------------
        # STAGE 4: Orientation Correction (Milestone 3c: Tesseract OSD)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("4_orientation", "running", {"message": "Detecting text orientation and rotating upright..."})
        orient_dir = self.staging_dir / "oriented"
        orient_dir.mkdir(parents=True, exist_ok=True)

        orient_res = detect_and_correct_orientation(str(page1_path), str(orient_dir))
        dur_4 = round(time.time() - t0, 3)
        oriented_path = Path(orient_res["output_path"])

        notify("4_orientation", "completed", {
            "duration_seconds": dur_4,
            "detected_angle": orient_res.get("detected_angle", 0),
            "rotated_deg": orient_res.get("rotate_angle", 0),
            "confidence": orient_res.get("confidence", 0.0),
            "oriented_path": str(oriented_path)
        })

        # -------------------------------------------------------------
        # STAGE 5: Deskewing (Milestone 3d: Hough Line Transform)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("5_deskew", "running", {"message": "Detecting micro-skew angle and leveling horizontal lines..."})
        deskew_dir = self.staging_dir / "deskewed"
        deskew_dir.mkdir(parents=True, exist_ok=True)

        deskew_res = deskew_image(str(oriented_path), str(deskew_dir))
        dur_5 = round(time.time() - t0, 3)
        deskewed_path = Path(deskew_res["output_path"])

        # Check for dual-copy page layout (Customer Copy vs Office Copy)
        copy_res = self.pipeline.copy_selector.inspect_and_select(
            deskewed_path,
            output_dir=self.staging_dir / "customer_copy"
        )
        if copy_res.is_dual_copy:
            deskewed_path = copy_res.output_image_path

        notify("5_deskew", "completed", {
            "duration_seconds": dur_5,
            "detected_skew_deg": round(deskew_res.get("detected_skew_angle", 0.0), 2),
            "rotation_applied_deg": round(deskew_res.get("correction_applied_deg", 0.0), 2),
            "status": deskew_res.get("status"),
            "deskewed_path": str(deskewed_path),
            "dual_copy_detected": copy_res.is_dual_copy,
            "selected_copy": copy_res.selected_copy,
            "split_axis": copy_res.split_axis,
            "boundary_coord": copy_res.boundary_coord
        })

        # -------------------------------------------------------------
        # STAGE 6: Dual-Engine OCR (Milestone 4: GPU Paddle + CPU Tesseract)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("6_ocr", "running", {"message": "Running GPU-accelerated PaddleOCR & auditing Tesseract OCR..."})

        paddle_res = self.pipeline.benchmarker.run_paddle(deskewed_path)
        extracted_lines = paddle_res.get("extracted_lines", [])
        paddle_full_text = paddle_res.get("full_text", "")

        tesseract_res = self.pipeline.benchmarker.run_tesseract(deskewed_path)
        tesseract_full_text = tesseract_res.get("full_text", "")
        dur_6 = round(time.time() - t0, 3)

        notify("6_ocr", "completed", {
            "duration_seconds": dur_6,
            "paddle_gpu_runtime": paddle_res.get("runtime_seconds"),
            "paddle_elements_count": len(extracted_lines),
            "paddle_avg_confidence": paddle_res.get("average_confidence"),
            "tesseract_cpu_runtime": tesseract_res.get("runtime_seconds"),
            "tesseract_words_count": tesseract_res.get("total_words", 0)
        })

        # -------------------------------------------------------------
        # STAGE 7: Table & Metadata Extraction (Milestones 5–6)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("7_extraction", "running", {"message": "Extracting parties, invoice numbers, dates, and line items..."})

        parties = self.pipeline._extract_parties_metadata(extracted_lines)
        inv_number = self.pipeline._extract_invoice_number(extracted_lines)
        inv_date = self.pipeline._extract_invoice_date(extracted_lines)
        due_date = self.pipeline._extract_due_date(extracted_lines)
        po_number = self.pipeline._extract_po_number(extracted_lines)
        tax_info = self.pipeline._extract_taxes_and_totals(paddle_full_text, extracted_lines)

        paddle_entities = DualEngineBenchmarker.extract_key_entities(paddle_full_text)
        tesseract_entities = DualEngineBenchmarker.extract_key_entities(tesseract_full_text)

        gstins = paddle_entities.get("gstin", [])
        vendor_gstin = parties.get("vendor_gstin")
        buyer_gstin = parties.get("buyer_gstin")
        if not vendor_gstin and not buyer_gstin and gstins:
            vendor_gstin = gstins[0]
            if len(gstins) > 1:
                buyer_gstin = gstins[1]

        table_res = self.pipeline.table_extractor.extract_table(extracted_lines)
        line_items = table_res.get("line_items", [])
        extracted_subtotal = table_res.get("calculated_subtotal")
        dur_7 = round(time.time() - t0, 3)

        notify("7_extraction", "completed", {
            "duration_seconds": dur_7,
            "invoice_number": inv_number,
            "invoice_date": inv_date,
            "vendor_name": parties.get("vendor_name"),
            "vendor_gstin": vendor_gstin,
            "line_items_count": len(line_items),
            "subtotal": extracted_subtotal
        })

        # -------------------------------------------------------------
        # STAGE 8: Mathematical Verification & Consensus Audit (Milestone 7)
        # -------------------------------------------------------------
        t0 = time.time()
        notify("8_verification", "running", {"message": "Validating GSTIN checksums and reconciling arithmetic..."})

        vendor_gstin_audit = self.pipeline.verifier.validate_gstin(vendor_gstin)
        buyer_gstin_audit = self.pipeline.verifier.validate_gstin(buyer_gstin)

        math_audit = self.pipeline.verifier.verify_arithmetic(
            line_items=line_items,
            extracted_subtotal=extracted_subtotal,
            cgst=tax_info.get("cgst"),
            sgst=tax_info.get("sgst"),
            igst=tax_info.get("igst"),
            roundoff=tax_info.get("roundoff"),
            extracted_grand_total=tax_info.get("grand_total")
        )

        consensus_audit = self.pipeline.verifier.audit_consensus(paddle_entities, tesseract_entities)
        dur_8 = round(time.time() - t0, 3)

        notify("8_verification", "completed", {
            "duration_seconds": dur_8,
            "arithmetic_status": math_audit.get("overall_status"),
            "grand_total": math_audit.get("reconciled_grand_total") or tax_info.get("grand_total"),
            "grand_total_source": math_audit.get("grand_total_source"),
            "consensus_level": consensus_audit.get("consensus_level"),
            "consensus_score": consensus_audit.get("consensus_score")
        })

        overall_duration = round(time.time() - overall_start, 2)

        # Build internal working data (preserving numbers for arithmetic reconciliation)
        internal_data = {
            "file_name": input_path.name,
            "all_lines": paddle_res.get("extracted_lines", []),
            "invoice_metadata": {
                "invoice_number": inv_number,
                "invoice_date": inv_date,
                "due_date": due_date,
                "PO_number": po_number,
                "currency_code": "INR",
                "vendor": {
                    "name": parties.get("vendor_name"),
                    "gstin": vendor_gstin,
                    "gstin_valid": vendor_gstin_audit.get("valid", False),
                    "pan": parties.get("pan"),
                    "bank_details": parties.get("bank_details")
                },
                "buyer": {
                    "name": parties.get("buyer_name"),
                    "gstin": buyer_gstin,
                    "gstin_valid": buyer_gstin_audit.get("valid", False)
                }
            },
            "line_items": line_items,
            "financial_summary": {
                "subtotal": extracted_subtotal,
                "cgst": tax_info.get("cgst"),
                "sgst": tax_info.get("sgst"),
                "igst": tax_info.get("igst"),
                "roundoff": tax_info.get("roundoff"),
                "grand_total": math_audit.get("reconciled_grand_total") or tax_info.get("grand_total"),
                "grand_total_source": math_audit.get("grand_total_source", "ocr_extracted")
            },
            "verification": {
                "arithmetic_status": math_audit.get("overall_status"),
                "line_items_verified": math_audit.get("all_items_verified"),
                "subtotal_verified": math_audit.get("subtotal_verified"),
                "grand_total_verified": math_audit.get("grand_total_verified"),
                "dual_engine_consensus": consensus_audit.get("consensus_level"),
                "consensus_score_pct": consensus_audit.get("consensus_score"),
                "agreed_gstin": consensus_audit.get("agreed_gstin"),
                "agreed_total_amount": consensus_audit.get("agreed_total_amount")
            },
            "pipeline_performance": {
                "total_runtime_seconds": overall_duration,
                "paddle_gpu_runtime": paddle_res.get("runtime_seconds"),
                "tesseract_cpu_runtime": tesseract_res.get("runtime_seconds"),
                "paddle_average_confidence": paddle_res.get("average_confidence")
            }
        }

        # Format into official output schema
        official_output = format_official_output(internal_data)

        # Vision-AI Fallback: If critical fields are missing or 0 items extracted, attempt patch
        if self.vision_fallback and self.vision_fallback.is_available():
            official_output = self.vision_fallback.patch_missing_fields(
                local_output=official_output,
                image_input=deskewed_path
            )

        # Audit sidecar data
        audit_data = {
            "file_name": input_path.name,
            "verification": internal_data["verification"],
            "financial_summary": internal_data["financial_summary"],
            "pipeline_performance": internal_data["pipeline_performance"],
            "vendor_gstin_validation": vendor_gstin_audit,
            "buyer_gstin_validation": buyer_gstin_audit,
            "raw_metadata": internal_data["invoice_metadata"]
        }

        return {
            "success": True,
            "stages": stage_reports,
            "total_duration_seconds": overall_duration,
            "page1_image_path": str(deskewed_path),
            "final_json": official_output,
            "audit_json": audit_data,
            "internal_json": internal_data
        }
