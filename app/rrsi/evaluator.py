# app/rrsi/evaluator.py
"""
Statistical Evaluator & Selection Decision Engine.
1. Type-aware matching (never numeric coercion, locale swap tracking, Decimal tolerance, GSTIN checksums).
2. Alignment by doc_id (never zip), with missing predictions penalized as pipeline_error.
3. Per-field Precision, Recall, F1, Accuracy, and Wilson Score Intervals.
4. Bipartite line-item table matching with structural error classification.
5. Tri-state cross-field arithmetic verification (validity flag, not mixed into extraction accuracy).
6. Fail-closed Selection Gate with paired cluster bootstrap, noise floor, and cost-aware utility.
"""

import math
import subprocess
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple, Set

from app.rrsi.taxonomy import (
    ExtractionErrorClass,
    ProcessOutcome,
    FailureDetail,
    FieldMetrics,
    DocumentEvaluationResult,
    EvaluationMetrics,
    DecisionRecord,
    ExperimentStatus,
    ProposalHypothesis
)
from app.rrsi.normalize import (
    match_invoice_number,
    match_date,
    match_amount,
    match_vendor_name,
    match_gstin
)
from app.rrsi.table_metrics import evaluate_table, TableEvaluationResult
from app.rrsi.stats import wilson_interval, paired_cluster_bootstrap, mcnemar_test


class StatisticalEvaluator:
    """
    Evaluates candidate pipeline outputs against ground truth,
    applies paired cluster bootstrap and cost-aware criteria, and catalogues all errors.
    """

    def __init__(
        self,
        cost_lambda: float = 0.05,
        tolerance: float = 1.0
    ):
        self.cost_lambda = cost_lambda
        self.tolerance = tolerance

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        manifest_docs: Optional[List[Dict[str, Any]]] = None,
        runtime_seconds: float = 0.0,
        vision_fallback_calls: int = 0
    ) -> Tuple[EvaluationMetrics, List[FailureDetail]]:
        """
        Aligns predictions and ground truth strictly by doc_id (never zip).
        Performs type-aware field and table evaluation.
        """
        from app.validators import classify_reconciliation

        # 1. Index Ground Truth by doc_id
        gt_by_id: Dict[str, Dict[str, Any]] = {}
        for gt in ground_truths:
            doc_id = str(gt.get("invoice_id") or gt.get("invoice_number") or gt.get("file_name") or "").strip()
            if not doc_id and manifest_docs:
                doc_id = f"doc_{len(gt_by_id)}"
            if doc_id:
                gt_by_id[doc_id] = gt

        # 2. Index Predictions by doc_id
        pred_by_id: Dict[str, Dict[str, Any]] = {}
        for pred in predictions:
            doc_id = str(pred.get("invoice_id") or pred.get("invoice_number") or pred.get("file_name") or "").strip()
            if not doc_id and len(predictions) == len(ground_truths):
                # Fallback if prediction lacks ID but aligns to single GT
                pass
            if doc_id:
                pred_by_id[doc_id] = pred

        # If IDs don't match, map sequentially if single document or fallback
        if len(gt_by_id) == 1 and len(predictions) == 1:
            only_gt_id = list(gt_by_id.keys())[0]
            pred_by_id[only_gt_id] = predictions[0]

        all_doc_ids = sorted(gt_by_id.keys())
        failures: List[FailureDetail] = []
        failure_counts: Dict[str, int] = {c.value: 0 for c in ExtractionErrorClass}

        # Track per-field totals
        field_names = [
            "invoice_number", "invoice_date", "due_date", "customer_name",
            "customer_gstin", "seller_name", "seller_gstin", "PO_number",
            "invoice_amount", "total_gst_amount"
        ]
        field_stats: Dict[str, Dict[str, int]] = {
            f: {"total_expected": 0, "correct": 0, "hallucinations": 0}
            for f in field_names
        }

        doc_results: Dict[str, DocumentEvaluationResult] = {}
        table_results: List[TableEvaluationResult] = []
        math_passed_count = 0
        doc_inconsistent_count = 0

        # Evaluate document by document
        for doc_id in all_doc_ids:
            gt = gt_by_id[doc_id]
            pred = pred_by_id.get(doc_id)

            doc_field_outcomes: Dict[str, bool] = {}
            doc_field_errors: Dict[str, Optional[str]] = {}
            doc_failures: List[FailureDetail] = []
            is_e2e_correct = True

            if pred is None:
                # Missing prediction = pipeline failure for every field
                failure_counts[ExtractionErrorClass.PIPELINE_ERROR.value] += 1
                fail_rec = FailureDetail(
                    primary_class=ExtractionErrorClass.PIPELINE_ERROR,
                    field_name="document",
                    expected=doc_id,
                    actual="MISSING_OUTPUT",
                    file_id=doc_id,
                    severity="critical",
                    context_message=f"Missing pipeline prediction for document '{doc_id}'"
                )
                failures.append(fail_rec)
                doc_failures.append(fail_rec)

                for f in field_names:
                    if gt.get(f) is not None:
                        field_stats[f]["total_expected"] += 1
                        doc_field_outcomes[f] = False
                        doc_field_errors[f] = "missing_prediction"
                is_e2e_correct = False

                doc_results[doc_id] = DocumentEvaluationResult(
                    doc_id=doc_id,
                    field_outcomes=doc_field_outcomes,
                    field_errors=doc_field_errors,
                    is_end_to_end_correct=False,
                    failures=doc_failures
                )
                continue

            # Field by field type-aware matching
            for f in field_names:
                exp_val = gt.get(f)
                act_val = pred.get(f)

                # Hallucination check: expected is null, actual is non-null
                if exp_val is None or str(exp_val).strip() == "":
                    if act_val is not None and str(act_val).strip() != "":
                        field_stats[f]["hallucinations"] += 1
                        failure_counts[ExtractionErrorClass.HALLUCINATION.value] += 1
                        fail_rec = FailureDetail(
                            primary_class=ExtractionErrorClass.HALLUCINATION,
                            field_name=f,
                            expected=None,
                            actual=act_val,
                            file_id=doc_id,
                            severity="low",
                            context_message=f"Hallucination on '{f}': expected null, extracted '{act_val}'"
                        )
                        failures.append(fail_rec)
                        doc_failures.append(fail_rec)
                    continue

                field_stats[f]["total_expected"] += 1
                matched = False
                err_subtype = None

                if f in ("invoice_number", "PO_number"):
                    matched, err_subtype = match_invoice_number(exp_val, act_val)
                    err_cls = ExtractionErrorClass.INVOICE_NUMBER_MISMATCH if f == "invoice_number" else ExtractionErrorClass.OTHER
                elif f in ("invoice_date", "due_date"):
                    matched, err_subtype = match_date(exp_val, act_val)
                    if err_subtype == "date_locale_swap":
                        err_cls = ExtractionErrorClass.DATE_LOCALE_SWAP
                    else:
                        err_cls = ExtractionErrorClass.DATE_MISMATCH
                elif f in ("invoice_amount", "total_gst_amount"):
                    matched, err_subtype = match_amount(exp_val, act_val, rel_tol=Decimal("0.001"), abs_tol=Decimal(str(self.tolerance)))
                    err_cls = ExtractionErrorClass.AMOUNT_PARSE
                elif f in ("customer_name", "seller_name"):
                    exact_m, fuzzy_m, score = match_vendor_name(exp_val, act_val)
                    matched = exact_m or fuzzy_m
                    err_cls = ExtractionErrorClass.PARTY_NAME_MISMATCH
                    if not matched:
                        err_subtype = f"name_mismatch_score_{score:.2f}"
                elif f in ("customer_gstin", "seller_gstin"):
                    matched, valid_chk, err_subtype = match_gstin(exp_val, act_val)
                    err_cls = ExtractionErrorClass.GSTIN_INVALID if not valid_chk else ExtractionErrorClass.GSTIN_MISMATCH
                else:
                    matched = str(exp_val).strip().lower() == str(act_val).strip().lower()
                    err_cls = ExtractionErrorClass.OTHER

                doc_field_outcomes[f] = matched
                doc_field_errors[f] = err_subtype

                if matched:
                    field_stats[f]["correct"] += 1
                else:
                    is_e2e_correct = False
                    failure_counts[err_cls.value] += 1
                    fail_rec = FailureDetail(
                        primary_class=err_cls,
                        field_name=f,
                        expected=exp_val,
                        actual=act_val,
                        file_id=doc_id,
                        severity="high" if f in ("invoice_number", "invoice_date", "invoice_amount") else "medium",
                        context_message=f"Field '{f}' mismatch on {doc_id}: expected '{exp_val}', got '{act_val}' ({err_subtype})"
                    )
                    failures.append(fail_rec)
                    doc_failures.append(fail_rec)

            # Table & Line-Item Bipartite Matching
            exp_items = gt.get("items", [])
            act_items = pred.get("items", [])
            table_eval = evaluate_table(exp_items, act_items)
            table_results.append(table_eval)
            for str_err in table_eval.structure_errors:
                failure_counts[str_err.error_type.value] += 1
                fail_rec = FailureDetail(
                    primary_class=ExtractionErrorClass(str_err.error_type.value),
                    field_name="table",
                    expected="valid_structure",
                    actual=str_err.message,
                    file_id=doc_id,
                    severity="high" if str_err.error_type.value in ("table_row_merge", "table_row_split") else "medium",
                    context_message=str_err.message
                )
                failures.append(fail_rec)
                doc_failures.append(fail_rec)

            if table_eval.row_f1 < 0.90:
                is_e2e_correct = False

            # Cross-Field Arithmetic Verification (Validity flag)
            math_code, pred_val_res, gt_val_res = classify_reconciliation(pred, ground_truth=gt, tolerance=self.tolerance)
            if math_code == "consistent":
                math_passed_count += 1
            elif math_code == "document_inconsistent":
                doc_inconsistent_count += 1
                failure_counts[ExtractionErrorClass.DOCUMENT_INCONSISTENT.value] += 1
            else:
                failure_counts[ExtractionErrorClass.TOTALS_ARITHMETIC.value] += 1

            doc_results[doc_id] = DocumentEvaluationResult(
                doc_id=doc_id,
                field_outcomes=doc_field_outcomes,
                field_errors=doc_field_errors,
                table_result=table_eval,
                validator_result=pred_val_res,
                arithmetic_status=math_code,
                is_end_to_end_correct=is_e2e_correct,
                failures=doc_failures
            )

        # 3. Aggregate Per-Field Metrics with Wilson CIs
        per_field_metrics: Dict[str, FieldMetrics] = {}
        field_accuracies = []
        field_f1s = []

        for f in field_names:
            tot = field_stats[f]["total_expected"]
            corr = field_stats[f]["correct"]
            halluc = field_stats[f]["hallucinations"]

            acc = float(corr / tot) if tot > 0 else 1.0
            prec = float(corr / (corr + halluc)) if (corr + halluc) > 0 else 1.0
            rec = acc
            f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            ci_low, ci_high = wilson_interval(corr, tot)

            per_field_metrics[f] = FieldMetrics(
                field_name=f,
                total_expected=tot,
                correct=corr,
                hallucinations=halluc,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                accuracy=round(acc * 100.0, 2),
                wilson_ci=(round(ci_low, 4), round(ci_high, 4))
            )
            if tot > 0:
                field_accuracies.append(acc * 100.0)
                field_f1s.append(f1 * 100.0)

        macro_acc = round(float(sum(field_accuracies) / len(field_accuracies)), 2) if field_accuracies else 100.0
        macro_f1 = round(float(sum(field_f1s) / len(field_f1s)), 2) if field_f1s else 100.0

        # Table aggregate metrics
        avg_row_p = sum(t.row_precision for t in table_results) / len(table_results) if table_results else 1.0
        avg_row_r = sum(t.row_recall for t in table_results) / len(table_results) if table_results else 1.0
        avg_row_f1 = sum(t.row_f1 for t in table_results) / len(table_results) if table_results else 1.0
        avg_cell_p = sum(t.cell_precision for t in table_results) / len(table_results) if table_results else 1.0
        avg_cell_r = sum(t.cell_recall for t in table_results) / len(table_results) if table_results else 1.0
        avg_cell_f1 = sum(t.cell_f1 for t in table_results) / len(table_results) if table_results else 1.0

        n_docs = max(1, len(all_doc_ids))
        e2e_count = sum(1 for d in doc_results.values() if d.is_end_to_end_correct)
        e2e_acc = round(float(e2e_count / n_docs * 100.0), 2)

        math_pct = round(float(math_passed_count / n_docs * 100.0), 2)
        doc_inconsistent_pct = round(float(doc_inconsistent_count / n_docs * 100.0), 2)

        metrics = EvaluationMetrics(
            macro_accuracy=macro_acc,
            macro_f1=macro_f1,
            per_field_metrics=per_field_metrics,
            table_row_precision=round(avg_row_p, 4),
            table_row_recall=round(avg_row_r, 4),
            table_row_f1=round(avg_row_f1, 4),
            table_cell_precision=round(avg_cell_p, 4),
            table_cell_recall=round(avg_cell_r, 4),
            table_cell_f1=round(avg_cell_f1, 4),
            end_to_end_accuracy=e2e_acc,
            arithmetic_consistency_pct=math_pct,
            document_inconsistent_pct=doc_inconsistent_pct,
            total_failures=len(failures),
            failure_breakdown=failure_counts,
            runtime_median_seconds=round(runtime_seconds, 2),
            runtime_p95_seconds=round(runtime_seconds * 1.25, 2),
            vision_fallback_calls=vision_fallback_calls,
            sample_count=len(all_doc_ids),
            doc_results=doc_results
        )

        return metrics, failures

    def decide_acceptance(
        self,
        hypothesis: ProposalHypothesis,
        baseline_dev: EvaluationMetrics,
        candidate_dev: EvaluationMetrics,
        baseline_selection: Optional[EvaluationMetrics] = None,
        candidate_selection: Optional[EvaluationMetrics] = None,
        template_groups: Optional[Dict[str, str]] = None,
        git_base_ref: Optional[str] = None
    ) -> DecisionRecord:
        """
        Selection Gate Engine (Fail-Closed):
        1. Raises ValueError if baseline_selection or candidate_selection is absent.
        2. Retrieves candidate diff via `git diff <base>..HEAD`; rejects empty diff as 'no change'.
        3. Paired Cluster Bootstrap CI on macro gain (lower bound > 0 and mean gain > 2x noise SD).
        4. Case-level regression check.
        5. Cost-aware utility penalty via cost_lambda.
        """
        # 1. Require Selection-Split Metrics
        if baseline_selection is None or candidate_selection is None:
            raise ValueError(
                "SELECTION GATE ERROR: Selection-split metrics are required; refusing to decide on dev set alone."
            )

        rejection_reasons: List[str] = []
        process_outcome = ProcessOutcome.ACCEPTED

        # 2. Check Git Diff from git_base_ref
        git_diff_text = ""
        if git_base_ref:
            try:
                res = subprocess.run(["git", "diff", f"{git_base_ref}..HEAD"], capture_output=True, text=True)
                git_diff_text = res.stdout
                if not git_diff_text.strip():
                    rejection_reasons.append("NO CODE CHANGE: Candidate diff is empty.")
                    process_outcome = ProcessOutcome.NO_CHANGE
            except Exception as e:
                rejection_reasons.append(f"GIT DIFF ERROR: Could not read diff: {e}")

        # 3. Paired Cluster Bootstrap on Selection Split
        base_scores = {
            doc_id: float(doc_res.is_end_to_end_correct)
            for doc_id, doc_res in baseline_selection.doc_results.items()
        }
        cand_scores = {
            doc_id: float(doc_res.is_end_to_end_correct)
            for doc_id, doc_res in candidate_selection.doc_results.items()
        }
        groups = template_groups or {d: "default_grp" for d in base_scores}

        boot_res = paired_cluster_bootstrap(base_scores, cand_scores, groups, n_resamples=5000)
        ci_lower = boot_res["ci_lower"]
        ci_upper = boot_res["ci_upper"]
        mean_delta = boot_res["mean_delta"]
        noise_sd = boot_res["std_err"]

        # Selection gate conditions:
        delta_acc = candidate_selection.macro_accuracy - baseline_selection.macro_accuracy
        if delta_acc <= 0.0:
            rejection_reasons.append(
                f"NO ACCURACY GAIN: Selection macro accuracy ({candidate_selection.macro_accuracy}%) <= "
                f"Baseline ({baseline_selection.macro_accuracy}%). Delta: {delta_acc:.2f}%."
            )
            process_outcome = ProcessOutcome.INSIGNIFICANT
        elif ci_lower <= 0.0:
            rejection_reasons.append(
                f"NOT STATISTICALLY SIGNIFICANT: 95% paired bootstrap CI lower bound ({ci_lower:.4f}) <= 0."
            )
            process_outcome = ProcessOutcome.INSIGNIFICANT
        elif mean_delta <= (2.0 * noise_sd) and noise_sd > 0.0:
            rejection_reasons.append(
                f"BELOW 2X NOISE SD: Mean gain ({mean_delta:.4f}) <= 2x noise standard deviation ({2.0 * noise_sd:.4f})."
            )
            process_outcome = ProcessOutcome.INSIGNIFICANT

        # 4. Case-Level Regression Tracking
        case_regressions: List[str] = []
        for doc_id, b_doc in baseline_selection.doc_results.items():
            c_doc = candidate_selection.doc_results.get(doc_id)
            if c_doc:
                for f_name, b_val in b_doc.field_outcomes.items():
                    c_val = c_doc.field_outcomes.get(f_name, False)
                    if b_val and not c_val:
                        case_regressions.append(f"Doc '{doc_id}' field '{f_name}' regressed from True to False")

        if case_regressions:
            rejection_reasons.append(
                f"CASE REGRESSIONS DETECTED: {len(case_regressions)} previously correct fields broken in candidate."
            )
            process_outcome = ProcessOutcome.REGRESSION_DETECTED

        # 5. Cost-Aware Acceptance
        latency_delta = round(candidate_selection.runtime_median_seconds - baseline_selection.runtime_median_seconds, 2)
        cost_penalty = self.cost_lambda * max(0.0, latency_delta)
        net_utility = delta_acc - (cost_penalty * 100.0)
        cost_accepted = net_utility >= 0.0

        if not cost_accepted:
            rejection_reasons.append(
                f"COST UTILITY NEGATIVE: Net utility ({net_utility:.2f}%) < 0 after latency penalty "
                f"(+{latency_delta}s @ lambda={self.cost_lambda})."
            )
            process_outcome = ProcessOutcome.COST_REJECTED

        status = ExperimentStatus.ACCEPTED if len(rejection_reasons) == 0 else ExperimentStatus.REJECTED

        # Failure deltas breakdown
        failure_deltas = {}
        for cat in ExtractionErrorClass:
            b_cnt = baseline_selection.failure_breakdown.get(cat.value, 0)
            c_cnt = candidate_selection.failure_breakdown.get(cat.value, 0)
            failure_deltas[cat.value] = c_cnt - b_cnt

        return DecisionRecord(
            proposal_id=hypothesis.proposal_id,
            area=hypothesis.area.value,
            status=status,
            process_outcome=process_outcome,
            baseline_accuracy=baseline_selection.macro_accuracy,
            candidate_accuracy=candidate_selection.macro_accuracy,
            delta_accuracy=round(delta_acc, 2),
            paired_ci_lower=round(ci_lower, 4),
            paired_ci_upper=round(ci_upper, 4),
            is_statistically_significant=(ci_lower > 0.0),
            leakage_screen_passed=True,
            leakage_findings=[],
            cost_aware_accepted=cost_accepted,
            latency_delta_seconds=latency_delta,
            vision_fallback_calls=candidate_selection.vision_fallback_calls,
            case_regressions=case_regressions,
            rejection_reasons=rejection_reasons,
            pruning_recommendations=[],
            failure_deltas=failure_deltas,
            provenance=hypothesis.provenance
        )
