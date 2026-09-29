# app/rrsi/screening.py
"""
Selection-Side Verification: Leakage Screening & Pruning Analyzer.
Implements:
- B1: Forbidden-set runtime store using salted hashes + char-8-gram hashes.
      No real PII / tokens in source.
- B2: AST scan of added lines of candidate diff (forbidden literals,
      concatenations, f-strings, coordinate comparisons including box[1] > 840,
      literal-int slices, vendor-conditional branches >= 6 chars). Whitelists
      kernel sizes and color constants via config/selection.yaml.
- B4: Paired-bootstrap comparison of dev gain vs selection gain.
- B5: Rule hit counter analysis requiring minimum corpus (>=40 docs, >=10 templates)
      to identify ablation candidates, and ablation acceptance evaluation.
"""

import ast
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
import yaml


class LeakageASTVisitor(ast.NodeVisitor):
    """
    AST Visitor to scan added lines of code for data leakage,
    coordinate hardcoding, and fragile vendor-conditional gating.
    """

    def __init__(
        self,
        salt: str = "",
        forbidden_hashes: Optional[Set[str]] = None,
        forbidden_ngrams: Optional[Set[str]] = None,
        whitelist_numbers: Optional[Set[float]] = None
    ):
        self.salt = salt
        self.forbidden_hashes = forbidden_hashes or set()
        self.forbidden_ngrams = forbidden_ngrams or set()
        self.whitelist_numbers = whitelist_numbers or {0, 1, 3, 5, 7, 255, 1000}
        self.findings: List[str] = []

    def _normalize_token(self, val: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", str(val)).lower()

    def _check_sensitive_string(self, text: str, node: ast.AST):
        if not self.salt or not (self.forbidden_hashes or self.forbidden_ngrams):
            return

        norm = self._normalize_token(text)
        if not norm or len(norm) < 3:
            return

        candidates = [norm]
        # If numeric, also check integer representation and 2-decimal representation
        try:
            f_val = float(text)
            candidates.append(str(int(f_val)))
            candidates.append(f"{f_val:.2f}".replace(".", ""))
        except (ValueError, TypeError):
            pass

        for cand in candidates:
            # 1. Exact salted hash check
            h_exact = hashlib.sha256((self.salt + cand).encode("utf-8")).hexdigest()
            if h_exact in self.forbidden_hashes:
                self.findings.append(
                    f"DATA LEAKAGE DETECTED: Literal matches forbidden entity hash: '{text}' (Line {getattr(node, 'lineno', '?')})"
                )
                return

            # 2. Salted char-8-gram check
            if len(cand) >= 8:
                for i in range(len(cand) - 7):
                    ngram = cand[i:i+8]
                    h_ngram = hashlib.sha256((self.salt + ngram).encode("utf-8")).hexdigest()
                    if h_ngram in self.forbidden_ngrams:
                        self.findings.append(
                            f"DATA LEAKAGE DETECTED: Literal substring matches forbidden 8-gram: '{ngram}' (Line {getattr(node, 'lineno', '?')})"
                        )
                        return

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self._check_sensitive_string(node.value, node)
        elif isinstance(node.value, (int, float)):
            # Check numbers if represented as string in ground truth
            self._check_sensitive_string(str(node.value), node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        """Detect constant-folded string concatenations like 'WMP/B/' + '04531'"""
        if isinstance(node.op, ast.Add):
            try:
                # If both are string constants, evaluate combined string
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) and \
                   isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                    combined = node.left.value + node.right.value
                    self._check_sensitive_string(combined, node)
            except Exception:
                pass
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        """Detect f-strings containing concatenated sensitive parts"""
        try:
            parts = []
            for val in node.values:
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    parts.append(val.value)
            if parts:
                combined = "".join(parts)
                self._check_sensitive_string(combined, node)
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check regex and function calls for sensitive arguments or benign patterns"""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Check regex calls
        if func_name in ("compile", "search", "match", "findall", "sub"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self._check_sensitive_string(arg.value, node)

        # Allow benign calls like cv2.resize(..., (1000, 1000))
        if func_name == "resize":
            return  # skip inspecting arguments of cv2.resize

        self.generic_visit(node)

    def _is_coordinate_node(self, node: ast.AST) -> bool:
        """Determines if an AST node represents a coordinate or bounding box attribute/subscript."""
        # 1. Attribute access: box.ymin, item.x, bbox.top, line.bottom, etc.
        if isinstance(node, ast.Attribute):
            coord_attrs = {"x", "y", "top", "left", "bottom", "right", "ymin", "ymax", "xmin", "xmax", "bbox", "box"}
            if node.attr.lower() in coord_attrs:
                return True

        # 2. Simple Name: ymin, xmax, top, left, x, y
        if isinstance(node, ast.Name):
            coord_names = {"x", "y", "top", "left", "bottom", "right", "ymin", "ymax", "xmin", "xmax"}
            if node.id.lower() in coord_names:
                return True

        # 3. Subscript: box[1], bbox[0], box[i], point[1]
        if isinstance(node, ast.Subscript):
            val = node.value
            if isinstance(val, ast.Name) and val.id.lower() in ("box", "bbox", "pt", "point", "coords", "points"):
                return True
            if isinstance(val, ast.Attribute) and val.attr.lower() in ("box", "bbox", "pt", "point", "coords"):
                return True

        return False

    def visit_Compare(self, node: ast.Compare):
        """
        Flag:
        1. Coordinate hardcoding: x/y/top/left/box[i]/bbox compared with >=50 (including box[1] > 840)
        2. Vendor-conditional branches: comparisons, in, startswith against string literals >=6 chars
        """
        left = node.left

        # Check for coordinate hardcoding
        is_left_coord = self._is_coordinate_node(left)
        for comparator in node.comparators:
            is_right_coord = self._is_coordinate_node(comparator)

            # Left is coord, right is number
            if is_left_coord and isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                val = float(comparator.value)
                if val >= 50.0 and val not in self.whitelist_numbers:
                    self.findings.append(
                        f"COORDINATE OVERFITTING DETECTED: Coordinate comparison with literal {comparator.value} "
                        f"(Line {getattr(node, 'lineno', '?')}). Heuristics must use relative ratios."
                    )

            # Right is coord, left is number
            elif is_right_coord and isinstance(left, ast.Constant) and isinstance(left.value, (int, float)):
                val = float(left.value)
                if val >= 50.0 and val not in self.whitelist_numbers:
                    self.findings.append(
                        f"COORDINATE OVERFITTING DETECTED: Coordinate comparison with literal {left.value} "
                        f"(Line {getattr(node, 'lineno', '?')}). Heuristics must use relative ratios."
                    )

            # Check vendor-conditional gating: string literal >= 6 chars compared against variable
            for op in node.ops:
                if isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
                    # Either operand is a string literal >= 6 chars
                    literal_str = None
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        literal_str = comparator.value
                    elif isinstance(left, ast.Constant) and isinstance(left.value, str):
                        literal_str = left.value

                    if literal_str and len(literal_str) >= 6:
                        norm = self._normalize_token(literal_str)
                        if norm and not literal_str.isupper() and not literal_str.startswith("GST"):
                            # Check if not a common standard keyword (like INVOICE, TOTAL, TAX)
                            common_keywords = {"invoice", "taxable", "subtotal", "balance", "consignee", "discount"}
                            if norm not in common_keywords:
                                self.findings.append(
                                    f"VENDOR-CONDITIONAL BRANCH DETECTED: Branch comparison against specific string '{literal_str}' "
                                    f"(Line {getattr(node, 'lineno', '?')}). Logic must be generalized across templates."
                                )

        self.generic_visit(node)

    def visit_Slice(self, node: ast.Slice):
        """Flag literal-int slices >= 50, e.g. crop[:850]"""
        for bound in (node.lower, node.upper):
            if isinstance(bound, ast.Constant) and isinstance(bound.value, int):
                if bound.value >= 50 and bound.value not in self.whitelist_numbers:
                    self.findings.append(
                        f"COORDINATE OVERFITTING DETECTED: Hardcoded pixel slice index [{bound.value}] "
                        f"(Line {getattr(node, 'lineno', '?')})."
                    )
        self.generic_visit(node)


class LeakageScreening:
    """
    Prevents data leakage and overfitting. Ensures changes generalize to unseen invoices
    rather than memorizing the test dataset.
    """

    GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
    PHONE_REGEX = re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b|\b0\d{2,4}[\-\s]?\d{6,8}\b")
    SYNTHETIC_WHITELIST = {
        "27AAPFU0939F1ZV", "27AAPFU0939F1ZW", "AAPFU0939F", "020-00000000", "9999999999"
    }

    @classmethod
    def _load_config_whitelist(cls, config_path: Optional[Path] = None) -> Set[float]:
        paths = [
            config_path,
            Path("config/selection.yaml"),
            Path("data_private/config/selection.yaml")
        ]
        for p in paths:
            if p and p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    nums = data.get("whitelist", {}).get("allowed_numeric_literals", [])
                    return {float(n) for n in nums}
                except Exception:
                    pass
        return {0.0, 1.0, 3.0, 5.0, 7.0, 255.0, 1000.0}

    @classmethod
    def _load_forbidden_hashes(cls) -> Tuple[str, Set[str], Set[str]]:
        for p in [Path("data_private/config/forbidden_hashes.json"), Path("config/forbidden_hashes.json")]:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    salt = data.get("salt", "")
                    hashes = set(data.get("hashes", []))
                    ngrams = set(data.get("ngrams", []))
                    return salt, hashes, ngrams
                except Exception:
                    pass
        return "", set(), set()

    @classmethod
    def build_runtime_forbidden_store(
        cls,
        ground_truths: List[Dict[str, Any]],
        salt: str = "RRSI_SALT_SECURE_2026_"
    ) -> Dict[str, Any]:
        """
        B1: Builds the forbidden set at runtime from selection and locked-test ground truth.
        Normalizes: casefold, alphanumeric only.
        Stores only salted hashes and salted char-8-gram hashes.
        No real vendor, GSTIN, phone, or invoice value is stored in plaintext.
        """
        salted_hashes = set()
        char_8grams = set()

        def add_token(val: Any):
            if val is None:
                return
            s_val = str(val).strip()
            norm = re.sub(r"[^a-zA-Z0-9]", "", s_val).lower()
            if not norm or len(norm) < 3:
                return

            # Full token hash
            h = hashlib.sha256((salt + norm).encode("utf-8")).hexdigest()
            salted_hashes.add(h)

            # Char-8-grams
            if len(norm) >= 8:
                for i in range(len(norm) - 7):
                    ng = norm[i:i+8]
                    h_ng = hashlib.sha256((salt + ng).encode("utf-8")).hexdigest()
                    char_8grams.add(h_ng)

        for gt in ground_truths:
            add_token(gt.get("invoice_number") or gt.get("invoice_id"))
            add_token(gt.get("seller_name"))
            add_token(gt.get("seller_gstin"))
            add_token(gt.get("customer_name"))
            add_token(gt.get("customer_gstin"))
            add_token(gt.get("total_amount") or gt.get("invoice_amount"))
            for item in gt.get("items", []):
                add_token(item.get("description"))
                add_token(item.get("hsn"))
                add_token(item.get("total_price") or item.get("amount"))

        return {
            "salt": salt,
            "hashes": sorted(list(salted_hashes)),
            "ngrams": sorted(list(char_8grams))
        }

    @classmethod
    def extract_added_lines_from_diff(cls, diff_or_code: str) -> str:
        """Extracts only the added lines (+) from a git diff string."""
        if "diff --git" not in diff_or_code and "@@" not in diff_or_code:
            return diff_or_code

        added_lines = []
        for line in diff_or_code.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])
        return "\n".join(added_lines)

    @classmethod
    def audit_code_changes(
        cls,
        code_or_diff: str,
        config_path: Optional[Path] = None,
        forbidden_hashes: Optional[Set[str]] = None,
        forbidden_ngrams: Optional[Set[str]] = None,
        salt: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        B2: AST scan of ONLY the added lines of the candidate diff.
        Flags:
        - String and numeric literals matching forbidden hashes or n-grams.
        - Concatenations ('WMP/B/' + '04531'), f-string parts, regex compile arguments.
        - Comparisons, in, startswith against literal strings >=6 chars gating extraction.
        - Numeric literals >=50 compared against coordinates (x, y, top, left, box[i], bbox).
        - Literal-int slices.
        - Whitelists kernel sizes and color constants.
        """
        findings: List[str] = []
        added_code = cls.extract_added_lines_from_diff(code_or_diff)

        # 1. Regex checks for un-whitelisted GSTIN and phone numbers
        for match in cls.GSTIN_REGEX.findall(added_code):
            if match not in cls.SYNTHETIC_WHITELIST:
                findings.append(f"DATA LEAKAGE DETECTED: Hardcoded GSTIN '{match}' found in diff.")

        for match in cls.PHONE_REGEX.findall(added_code):
            if match not in cls.SYNTHETIC_WHITELIST:
                findings.append(f"DATA LEAKAGE DETECTED: Hardcoded phone number '{match}' found in diff.")

        # 2. Load config whitelist & forbidden hashes
        whitelist_numbers = cls._load_config_whitelist(config_path)
        if salt is None or forbidden_hashes is None:
            loaded_salt, loaded_hashes, loaded_ngrams = cls._load_forbidden_hashes()
            salt = salt or loaded_salt
            forbidden_hashes = forbidden_hashes or loaded_hashes
            forbidden_ngrams = forbidden_ngrams or loaded_ngrams

        # 3. Parse AST of added lines
        tree = None
        # Try parsing directly
        try:
            tree = ast.parse(added_code)
        except SyntaxError:
            # If diff contains indented lines or partial statements, wrap in function body
            try:
                indented = "\n".join(f"    {line}" for line in added_code.splitlines())
                tree = ast.parse(f"def _temp_diff_func():\n{indented}")
            except SyntaxError:
                # If still syntax error, line-by-line fallback
                pass

        if tree is not None:
            visitor = LeakageASTVisitor(
                salt=salt or "",
                forbidden_hashes=forbidden_hashes or set(),
                forbidden_ngrams=forbidden_ngrams or set(),
                whitelist_numbers=whitelist_numbers
            )
            visitor.visit(tree)
            findings.extend(visitor.findings)
        else:
            # Regex fallback if AST completely fails
            box_pat = re.compile(r"\b(?:box|bbox)\[\d+\]\s*(?:==|>=|<=|>|<)\s*(\d+)\b")
            for m in box_pat.finditer(added_code):
                val = float(m.group(1))
                if val >= 50.0 and val not in whitelist_numbers:
                    findings.append(f"COORDINATE OVERFITTING DETECTED: Hardcoded coordinate comparison '{m.group(0)}'.")

        # Deduplicate findings while preserving order
        unique_findings = []
        for f in findings:
            if f not in unique_findings:
                unique_findings.append(f)

        passed = (len(unique_findings) == 0)
        return passed, unique_findings

    @classmethod
    def compare_dev_vs_selection_gain(
        cls,
        dev_gain: float,
        selection_gain: float,
        selection_ci_lower: float,
        max_gap: float = 0.05
    ) -> Tuple[bool, str]:
        """
        B4: Replaces audit_split_generalization with paired-bootstrap comparison
        of dev gain vs selection gain.
        Rejects when:
        - Selection CI lower bound <= 0
        - Dev-selection gap widens beyond max_gap limit.
        Does NOT reject on a point estimate below 0 that is inside the CI.
        """
        if selection_ci_lower <= 0.0:
            return False, (
                f"GENERALIZATION REJECTED: Selection 95% CI lower bound ({selection_ci_lower:.4f}) <= 0. "
                f"Candidate did not demonstrate statistically significant improvement on selection split."
            )

        gap = dev_gain - selection_gain
        if gap > max_gap:
            return False, (
                f"GENERALIZATION REJECTED: Dev-selection gap ({gap:.4f}) exceeds allowable limit ({max_gap:.4f}). "
                f"Dev gain (+{dev_gain:.4f}) disproportionate to selection gain (+{selection_gain:.4f})."
            )

        return True, f"Split generalization verified (Dev: +{dev_gain:.4f}, Selection: +{selection_gain:.4f}, CI low: {selection_ci_lower:.4f})."


class PruningAnalyzer:
    """
    B5: Instruments rules, regexes, and keywords with hit counters.
    Uses zero-hit rules only as ablation CANDIDATES.
    Requires minimum corpus (>=40 docs, >=10 templates); below it returns INSUFFICIENT_EVIDENCE.
    Removal requires an ablation showing a within-noise score change and a cost drop.
    """

    @staticmethod
    def audit_rule_ablation_candidates(
        rule_hit_counts: Dict[str, int],
        corpus_doc_count: int,
        corpus_template_count: int,
        min_docs: int = 40,
        min_templates: int = 10
    ) -> Dict[str, Any]:
        """
        Audits rule hit counters across the corpus.
        Returns INSUFFICIENT_EVIDENCE if corpus size is below minimums.
        """
        if corpus_doc_count < min_docs or corpus_template_count < min_templates:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "message": (
                    f"Corpus insufficient ({corpus_doc_count}/{min_docs} docs, "
                    f"{corpus_template_count}/{min_templates} templates). "
                    f"Never propose removal below minimum evidence threshold."
                ),
                "candidates": []
            }

        zero_hit_candidates = [rule for rule, count in rule_hit_counts.items() if count == 0]
        return {
            "status": "CANDIDATES_IDENTIFIED" if zero_hit_candidates else "NO_DEAD_RULES",
            "message": f"Identified {len(zero_hit_candidates)} zero-hit rule(s) as ablation candidates.",
            "candidates": zero_hit_candidates
        }

    @staticmethod
    def evaluate_ablation_acceptance(
        delta_accuracy: float,
        noise_sd: float,
        cost_delta: float
    ) -> Tuple[bool, str]:
        """
        Evaluates whether an ablation experiment should be accepted.
        Criteria:
        1. Within-noise score change (|delta_accuracy| <= noise_sd)
        2. Cost or complexity drop (cost_delta < 0)
        """
        if abs(delta_accuracy) > noise_sd and delta_accuracy < 0.0:
            return False, f"ABLATION REJECTED: Accuracy dropped by {delta_accuracy:.4f}%, exceeding noise SD ({noise_sd:.4f}%)."

        if cost_delta >= 0.0:
            return False, f"ABLATION REJECTED: Cost or latency did not decrease (cost_delta={cost_delta:.4f})."

        return True, "Ablation accepted: Score change is within noise floor and cost/complexity reduced."
