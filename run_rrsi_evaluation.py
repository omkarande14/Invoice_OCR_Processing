# run_rrsi_evaluation.py
"""
RRSI Execution CLI & Benchmark Runner.
Runs the Reflective Recursive Self-Improvement evaluation workflow:
- Measures baseline on ground truth dataset
- Validates proposals against Negative Knowledge Memory
- Screens code for Data Leakage and static coordinate overfitting
- Enforces Statistical Noise Floor & Cost-Aware Acceptance
- Categorizes all failures by the fine-grained RRSI Error Taxonomy
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.rrsi import (
    RRSIOrchestrator,
    ProposalHypothesis,
    PipelineArea,
    EvaluationMetrics,
    ErrorCategory,
    DecisionRecord,
    ExperimentStatus
)
from app.rrsi.stats import wilson_interval


import tempfile

def load_ground_truths(gt_dir: Path):
    """Loads all ground truth files in data/ground_truth."""
    gts = []
    for f in gt_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    gts.extend(data)
                elif isinstance(data, dict):
                    gts.append(data)
        except Exception as e:
            print(f"Warning loading {f.name}: {e}")
    return gts


def find_image_for_gt(gt: dict, project_root: Path) -> Optional[Path]:
    """Finds the actual source image path corresponding to a ground truth dictionary."""
    doc_id = gt.get("doc_id")
    inv_num = gt.get("invoice_number") or gt.get("invoice_id")

    # 1. Check data_private/manifest_private.json
    man_path = project_root / "data_private" / "manifest_private.json"
    if man_path.exists():
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if doc_id and doc_id in manifest:
                files = manifest[doc_id].get("files", [])
                for candidate_dir in [
                    project_root / "data" / "deskewed",
                    project_root / "data" / "test_uploads" / "staging" / "deskewed"
                ]:
                    if files:
                        p = candidate_dir / files[0]
                        if p.exists():
                            return p
        except Exception:
            pass

    # 2. Check metadata files in data_private/ground_truth/
    meta_dir = project_root / "data_private" / "ground_truth"
    if meta_dir.exists():
        for meta_file in meta_dir.glob("*_metadata.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as fp:
                    meta = json.load(fp)
                    if meta.get("doc_id") in [inv_num, doc_id]:
                        fname = meta.get("filename")
                        if fname:
                            for candidate_dir in [
                                project_root / "data" / "deskewed",
                                project_root / "data" / "test_uploads" / "staging" / "deskewed"
                            ]:
                                p = candidate_dir / fname
                                if p.exists():
                                    return p
            except Exception:
                pass

    if not inv_num:
        return None

    # 3. Search in deskewed directories by full string and digit stems
    digits = [d for d in re.findall(r"\d{4,}", inv_num)]
    for search_dir in [
        project_root / "data" / "deskewed",
        project_root / "data" / "test_uploads" / "staging" / "deskewed"
    ]:
        if search_dir.exists():
            clean_no = inv_num.replace("/", "_").replace("-", "_")
            for img in search_dir.glob("*.png"):
                if clean_no in img.name or inv_num in img.name:
                    return img
                for d in digits:
                    if d in img.name:
                        return img

    return None


def run_benchmark_audit():
    """
    Runs a live benchmark audit by executing the real OCR pipeline (app/ocr/invoice_pipeline.py)
    on the source invoice images and comparing against ground truth annotations.
    """
    print("=" * 75)
    print("        RRSI FRAMEWORK - LIVE PIPELINE AUDIT (SOURCE IMAGES)")
    print("=" * 75)

    from app.ocr.invoice_pipeline import InvoicePipeline

    orchestrator = RRSIOrchestrator()
    private_gt = PROJECT_ROOT / "data_private" / "ground_truth"
    gt_dir = private_gt if private_gt.exists() else PROJECT_ROOT / "data" / "ground_truth"
    gts = load_ground_truths(gt_dir)

    print(f"Loaded {len(gts)} ground truth annotation files from: {gt_dir}")
    if not gts:
        print("[!] No ground truth files found. Ensure data/ground_truth contains JSON files.")
        return

    pipeline = InvoicePipeline(use_gpu=False)

    evaluated_gts = []
    predictions = []
    total_pipeline_time = 0.0

    for gt in gts:
        # Require canonical schema with top-level invoice_number and items list
        if not gt.get("invoice_number") or not gt.get("items"):
            continue

        img_path = find_image_for_gt(gt, PROJECT_ROOT)
        if not img_path or not img_path.exists():
            print(f"[!] Warning: No source image found for invoice '{gt.get('invoice_number')}'. Skipping.")
            continue

        print(f"Executing real pipeline on source image: {img_path.name} ...")
        t0 = time.time()
        pred = pipeline.process_invoice(img_path, return_diagnostics=False)
        dt = time.time() - t0
        total_pipeline_time += dt

        evaluated_gts.append(gt)
        predictions.append(pred)

    if not evaluated_gts:
        print("[!] No runnable ground-truth image pairs found.")
        return

    metrics, failures = orchestrator.evaluator.evaluate_predictions(
        predictions=predictions,
        ground_truths=evaluated_gts,
        runtime_seconds=total_pipeline_time
    )

    print("\n[+] Audit Metrics Summary (Live Pipeline Execution on Real Images):")
    print(f"    Total Invoices Processed: {metrics.sample_count}")
    print(f"    Overall Accuracy Score:   {metrics.overall_accuracy_pct:.2f}%")
    print(f"    Field Accuracy:           {metrics.field_accuracy_pct:.2f}%")
    print(f"    Table F1-Score:           {metrics.table_f1_score:.3f}")
    print(f"    Arithmetic Consistency:   {metrics.math_consistency_pct:.2f}%")
    print(f"    Total Execution Time:     {total_pipeline_time:.2f}s ({total_pipeline_time / metrics.sample_count:.2f}s / invoice)")
    ci_low, ci_high = wilson_interval(int(round(metrics.overall_accuracy_pct * metrics.sample_count / 100)), metrics.sample_count)
    print(f"    Wilson 95% CI:            [{ci_low*100:.1f}%, {ci_high*100:.1f}%] (N={metrics.sample_count})")

    print("\n[+] Fine-Grained Failure Breakdown by RRSI Taxonomy:")
    print("-" * 60)
    for cat in ErrorCategory:
        count = metrics.failure_breakdown.get(cat.value, 0)
        status_icon = "[OK] 0 Errors" if count == 0 else f"[FAIL] {count} Failures"
        print(f"    * {cat.value:<28} : {status_icon}")
    print("-" * 60)

    if failures:
        print("\n[!] Discovered Extraction Discrepancies:")
        for f in failures[:8]:
            print(f"    - [{f.category.value}] Field '{f.field_name}' on {f.file_id}: Expected '{f.expected}', Got '{f.actual}'")

    # Stagnation Report
    stagnation = orchestrator.get_stagnation_report()
    print("\n[+] Pipeline Area Stagnation Status (Proposal-Side Memory):")
    for area, status in stagnation.items():
        stag_icon = "[STAGNANT] Shift Area" if status["is_stagnant"] else "[ACTIVE]"
        print(f"    * {area:<25} : {status['consecutive_failures']} consecutive fails | {stag_icon}")

    print("\n" + "=" * 75)


def run_demo_proposal():
    """Demonstrates Proposal-Side screening and Selection-Side noise floor/leakage evaluation in strict isolation."""
    print("=" * 75)
    print("        DEMONSTRATION: PROPOSAL & SELECTION WORKFLOW (ISOLATED)")
    print("=" * 75)

    # Strictly isolate demo execution to a temporary directory so data/rrsi/ is never modified
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        orchestrator = RRSIOrchestrator(
            memory_path=tmp_path / "demo_memory.json",
            ledger_path=tmp_path / "demo_ledger.json"
        )

        leaky_hypothesis = ProposalHypothesis(
            proposal_id="PROP-2026-001",
            area=PipelineArea.ENTITY_EXTRACTION,
            title="Hardcode vendor check for Acme Healthcare",
            description="Specifically check for Acme Healthcare and WMP/B/04531 in line text",
            rationale="Fixes invoice WMP/04531 extraction test failure",
            code_diff_summary="if 'WMP/B/04531' in text: return 'WMP/B/04531'",
            tags=["regex", "hardcode", "vendor"],
            provenance="demo"
        )

        print(f"\n1. Submitting Candidate Proposal: {leaky_hypothesis.proposal_id} - '{leaky_hypothesis.title}'")
        is_viable, warnings, rec = orchestrator.submit_proposal(leaky_hypothesis)
        print(f"   Proposal Viability Check: {'PASSED' if is_viable else 'WARNINGS DETECTED'}")
        for w in warnings:
            print(f"   - {w}")

        code_diff = "if text == 'WMP/B/04531': return 'WMP/B/04531'"
        leak_passed, leak_findings = orchestrator.screen_code_changes(code_diff)
        print(f"   Leakage Screen: {'PASSED' if leak_passed else 'REJECTED (Leakage Detected)'}")
        for lf in leak_findings:
            print(f"   - [LEAKAGE] {lf}")

        b_metrics = EvaluationMetrics(overall_accuracy_pct=88.5, sample_count=10, runtime_seconds=2.1)
        c_metrics = EvaluationMetrics(overall_accuracy_pct=89.0, sample_count=10, runtime_seconds=2.2)

        decision = orchestrator.evaluate_and_decide(
            hypothesis=leaky_hypothesis,
            baseline_metrics=b_metrics,
            candidate_metrics=c_metrics,
            candidate_code=code_diff
        )

        print("\n2. Evaluator Decision Report:")
        print(orchestrator.format_decision_report(decision))
        print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoice OCR - RRSI Evaluation Tool")
    parser.add_argument("--audit", action="store_true", help="Run baseline audit on ground truth")
    parser.add_argument("--demo", action="store_true", help="Run demonstration of RRSI proposal & selection")
    args = parser.parse_args()

    if args.demo:
        run_demo_proposal()
    else:
        run_benchmark_audit()
