import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import pytesseract
from pytesseract import Output

# Configure GPU memory management and DLL paths on Windows so Paddle can load cuDNN v8
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.4")

def configure_gpu_dll_paths() -> None:
    dll_dirs = [
        r"C:\Ai India Innovations\InvoiceOCR\venv\Lib\site-packages\nvidia\cudnn\bin",
        r"C:\Ai India Innovations\InvoiceOCR\venv\Lib\site-packages\nvidia\cublas\bin",
        r"C:\Ai India Innovations\InvoiceOCR\venv\Lib\site-packages\nvidia\cuda_nvrtc\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
    ]
    for d in dll_dirs:
        if os.path.exists(d):
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass

configure_gpu_dll_paths()

try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


from app.ocr.stage_cache import OCRStageCache


class DualEngineBenchmarker:
    """
    Benchmarks and compares PaddleOCR (deep learning, layout/table focus)
    against Tesseract OCR (traditional LSTM, isolated patch validation)
    across invoices.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        tesseract_cmd: Optional[str] = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        stage_cache: Optional[OCRStageCache] = None,
        enable_cache: bool = True
    ):
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.enable_cache = enable_cache
        self.stage_cache = stage_cache or (OCRStageCache() if enable_cache else None)

        self.use_gpu = use_gpu and PADDLE_AVAILABLE and (
            hasattr(paddle, "is_compiled_with_cuda") and paddle.is_compiled_with_cuda()
        )
        self.paddle_ocr = None
        if PADDLE_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=self.use_gpu,
                    gpu_mem=500,
                    det_limit_side_len=1280,
                    det_limit_type='max',
                    show_log=False
                )
            except Exception as exc:
                print(f"Warning: PaddleOCR init with use_gpu={self.use_gpu} failed: {exc}. Falling back to CPU.")
                self.use_gpu = False
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=False,
                    show_log=False
                )

    def run_paddle(self, image_path: Path) -> Dict[str, Any]:
        """Runs PaddleOCR on a single image and collects structured text and metrics."""
        if not self.paddle_ocr:
            return {"error": "PaddleOCR is not available"}

        img_sha = None
        if self.stage_cache and self.stage_cache.enabled:
            try:
                img_sha = OCRStageCache.compute_image_sha256(image_path)
                cached = self.stage_cache.get(img_sha, "PaddleOCR", "v1.0")
                if cached:
                    # Mark cached run
                    cached_out = dict(cached)
                    cached_out["device"] = "Cache (Instant)"
                    cached_out["runtime_seconds"] = 0.001
                    return cached_out
            except Exception:
                pass

        start_time = time.time()
        actual_device = "GPU" if self.use_gpu else "CPU"
        try:
            result = self.paddle_ocr.ocr(str(image_path), cls=False)
        except (Exception, BaseException) as exc:
            print(f"Warning: Primary GPU OCR error ({exc}). Automatically switching to CPU PaddleOCR...")
            self.use_gpu = False
            actual_device = "CPU (Fallback)"
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=False,
                    show_log=False
                )
                result = self.paddle_ocr.ocr(str(image_path), cls=False)
            except Exception as cpu_err:
                print(f"Warning: CPU PaddleOCR fallback failed ({cpu_err}). Returning empty OCR result.")
                result = [[]]

        duration = round(time.time() - start_time, 3)

        raw_lines = result[0] if result and result[0] else []
        extracted_lines = []
        confidences = []

        for item in raw_lines:
            box, (text, score) = item
            clean_text = text.strip()
            if clean_text:
                extracted_lines.append({
                    "text": clean_text,
                    "confidence": round(float(score), 4),
                    "box": box
                })
                confidences.append(float(score))

        full_text = "\n".join(item["text"] for item in extracted_lines)
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        out = {
            "engine": "PaddleOCR",
            "device": actual_device,
            "runtime_seconds": duration,
            "total_elements": len(extracted_lines),
            "average_confidence": avg_conf,
            "extracted_lines": extracted_lines,
            "full_text": full_text
        }
        if self.stage_cache and self.stage_cache.enabled and img_sha and extracted_lines:
            try:
                self.stage_cache.put(img_sha, "PaddleOCR", "v1.0", out)
            except Exception:
                pass
        return out

    def run_tesseract(self, image_path: Path) -> Dict[str, Any]:
        """Runs Tesseract OCR on a single image and collects text and word confidences."""
        start_time = time.time()
        img = Image.open(image_path)
        
        # Get detailed word-level data
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        full_text = pytesseract.image_to_string(img)
        duration = round(time.time() - start_time, 3)

        valid_confs = []
        words = []
        for i in range(len(data["text"])):
            w = data["text"][i].strip()
            conf = float(data["conf"][i])
            if w and conf >= 0:
                words.append({"word": w, "confidence": round(conf / 100.0, 4)})
                valid_confs.append(conf / 100.0)

        avg_conf = round(sum(valid_confs) / len(valid_confs), 4) if valid_confs else 0.0
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]

        return {
            "engine": "Tesseract",
            "device": "CPU",
            "runtime_seconds": duration,
            "total_words": len(words),
            "total_lines": len(lines),
            "average_confidence": avg_conf,
            "full_text": full_text
        }

    @staticmethod
    def extract_key_entities(text: str) -> Dict[str, Any]:
        """
        Extracts foundational invoice entities using pattern recognition:
        - GSTIN numbers (Indian standard 15-char alphanumeric)
        - Invoice Numbers
        - Dates
        - Grand Totals / Amounts
        """
        # GSTIN pattern: 2 digits + 5 alpha + 4 digits + 1 alpha + 1 alphanumeric + Z + 1 alphanumeric
        gstin_matches = re.findall(r"(?:GSTIN?[:\s]*)?([0-9]{2}[A-Z0-9]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z2][0-9A-Z]{1})", text, re.IGNORECASE)
        # Normalize to uppercase
        gstin_list = sorted(list(set(g.upper() for g in gstin_matches)))

        # Invoice number pattern
        inv_no_patterns = [
            r"(?:Invoice\s*(?:No\.?|Number|#)?[:\s-]*)([A-Za-z0-9\/\-]+)",
            r"(?:Inv\.?\s*(?:No\.?|#)?[:\s-]*)([A-Za-z0-9\/\-]+)",
            r"(?:Bill\s*(?:No\.?|Number|#)?[:\s-]*)([A-Za-z0-9\/\-]+)"
        ]
        invoice_numbers = []
        for pat in inv_no_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                m_clean = m.strip(" :-")
                if len(m_clean) >= 3 and not m_clean.lower().startswith(("date", "gst", "total")):
                    invoice_numbers.append(m_clean)
        invoice_numbers = list(dict.fromkeys(invoice_numbers))  # preserve order unique

        # Dates (DD/MM/YYYY, DD-MM-YYYY, DD-Mon-YY, etc.)
        date_matches = re.findall(
            r"\b\d{1,2}[-\/](?:\d{1,2}|[A-Za-z]{3})[-\/](?:\d{4}|\d{2})\b",
            text
        )
        dates = list(dict.fromkeys(date_matches))

        # Amounts / Totals (e.g., Grand Total: 252,139.00 or 252139.00)
        # Priority 1: Labeled Grand Total / Net Amount / Total Amount
        gt_matches = re.findall(
            r"(?:Grand\s*Total|Total\s*Amount|Amount\s*Payable|Net\s*(?:Amount|Amt\.?|Payable)|Invoice\s*Total|TO\s*PAY|TOPAY)[\s:₹Rs\.]*([0-9,]+\.[0-9]{2}|[0-9,]+)",
            text,
            re.IGNORECASE
        )
        totals = []
        for t in gt_matches:
            val_clean = t.replace(",", "").strip(".")
            try:
                val_float = float(val_clean)
                if val_float > 10.0:  # Skip trivial quantities or single digits
                    totals.append(f"{val_float:.2f}")
            except ValueError:
                pass

        # Priority 2: General total lines (excluding Total Qty, Total Items, Total Cases, etc.)
        if not totals:
            general_matches = re.findall(
                r"(?:Total(?!\s*(?:Qty|Quantity|Items|Cases|Boxes|Class|GST|Tax|Discount)))[\s:₹Rs\.]*([0-9,]+\.[0-9]{2}|[0-9,]+)",
                text,
                re.IGNORECASE
            )
            for t in general_matches:
                val_clean = t.replace(",", "").strip(".")
                try:
                    val_float = float(val_clean)
                    if val_float > 10.0:
                        totals.append(f"{val_float:.2f}")
                except ValueError:
                    pass

        # Priority 3: Any bottom-of-invoice monetary amount >= 100
        if not totals:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            bottom_text = " ".join(lines[-20:])
            bottom_nums = re.findall(r"\b([0-9,]+\.[0-9]{2})\b", bottom_text)
            for t in reversed(bottom_nums):
                val_clean = t.replace(",", "").strip(".")
                try:
                    val_float = float(val_clean)
                    if val_float >= 100.0:
                        totals.append(f"{val_float:.2f}")
                        break
                except ValueError:
                    pass

        totals = list(dict.fromkeys(totals))

        return {
            "gstin": gstin_list,
            "invoice_numbers": invoice_numbers[:3],
            "dates": dates[:3],
            "totals": totals[:3]
        }

    def compare_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Runs both PaddleOCR and Tesseract on the invoice image,
        evaluates entity extraction, and determines cross-engine consensus.
        """
        p_res = self.run_paddle(image_path)
        t_res = self.run_tesseract(image_path)

        p_entities = self.extract_key_entities(p_res.get("full_text", ""))
        t_entities = self.extract_key_entities(t_res.get("full_text", ""))

        # Check consensus on entities
        gstin_overlap = set(p_entities["gstin"]).intersection(set(t_entities["gstin"]))
        inv_overlap = set(p_entities["invoice_numbers"]).intersection(set(t_entities["invoice_numbers"]))
        total_overlap = set(p_entities["totals"]).intersection(set(t_entities["totals"]))

        consensus_summary = {
            "gstin_match": len(gstin_overlap) > 0,
            "gstin_agreed": list(gstin_overlap),
            "invoice_no_match": len(inv_overlap) > 0,
            "invoice_no_agreed": list(inv_overlap),
            "total_amount_match": len(total_overlap) > 0,
            "total_amount_agreed": list(total_overlap),
            "paddle_exclusive_gstin": list(set(p_entities["gstin"]) - set(t_entities["gstin"])),
            "tesseract_exclusive_gstin": list(set(t_entities["gstin"]) - set(p_entities["gstin"]))
        }

        # Calculate a consensus confidence score (0 to 100)
        agreements = sum([
            1 if consensus_summary["gstin_match"] else 0,
            1 if consensus_summary["invoice_no_match"] else 0,
            1 if consensus_summary["total_amount_match"] else 0
        ])
        consensus_score = round((agreements / 3.0) * 100, 1)

        return {
            "image_name": image_path.name,
            "paddle_metrics": {
                "runtime_seconds": p_res.get("runtime_seconds", 0.0),
                "total_elements": p_res.get("total_elements", 0),
                "average_confidence": p_res.get("average_confidence", 0.0),
                "entities": p_entities
            },
            "tesseract_metrics": {
                "runtime_seconds": t_res.get("runtime_seconds", 0.0),
                "total_words": t_res.get("total_words", 0),
                "average_confidence": t_res.get("average_confidence", 0.0),
                "entities": t_entities
            },
            "consensus": consensus_summary,
            "consensus_score": consensus_score
        }

    def benchmark_sample_set(
        self,
        images: List[Path],
        output_report: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Runs the comparison across a set of sample invoices and compiles a report."""
        results = []
        for img in images:
            if img.exists():
                print(f"Benchmarking dual engines on: {img.name}...")
                res = self.compare_image(img)
                results.append(res)

        # Compute summary averages
        avg_paddle_time = round(
            sum(r["paddle_metrics"]["runtime_seconds"] for r in results) / len(results), 2
        ) if results else 0.0
        avg_tess_time = round(
            sum(r["tesseract_metrics"]["runtime_seconds"] for r in results) / len(results), 2
        ) if results else 0.0
        avg_consensus = round(
            sum(r["consensus_score"] for r in results) / len(results), 1
        ) if results else 0.0

        summary_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images_tested": len(results),
            "average_paddle_gpu_runtime": avg_paddle_time,
            "average_tesseract_cpu_runtime": avg_tess_time,
            "average_consensus_agreement_pct": avg_consensus,
            "per_image_comparisons": results
        }

        if output_report:
            output_report.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(output_report, "w", encoding="utf-8") as f:
                json.dump(summary_report, f, indent=2)
            print(f"Benchmark report saved to: {output_report}")

        return summary_report
