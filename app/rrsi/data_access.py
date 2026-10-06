# app/rrsi/data_access.py
"""
Data Access Control and Manifest Integrity Verification (Phase 1.4).

Policy:
- app/rrsi/data_access.py is the authoritative gate for dataset reads in RRSI.
- Refuses 'locked_test' split access unless environment variable RRSI_FINAL_EVAL=1 is set.
- Convention & CI Check: This is a software convention enforced in Python and tested in CI,
  not an OS-level hardware security guarantee.
- Enforces manifest integrity:
  1. No split overlap across doc_ids.
  2. No template straddling across splits.
  3. No copy/page split divergence.
  4. No contaminated invoices in selection or locked_test splits.
  5. SHA-256 hash consistency.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set


class DataAccessGate:
    """Controls read access to invoice datasets and validates partition integrity."""

    DEFAULT_MANIFEST_PATHS = [
        Path("data/manifest.json"),
        Path("data_private/manifest_private.json")
    ]

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or self._resolve_manifest_path()
        self._manifest_cache: Optional[Dict[str, Any]] = None

    def _resolve_manifest_path(self) -> Path:
        for p in self.DEFAULT_MANIFEST_PATHS:
            if p.exists():
                return p
        return Path("data/manifest.json")

    def load_manifest(self, force_reload: bool = False) -> Dict[str, Any]:
        """Loads and returns the manifest dictionary."""
        if self._manifest_cache is not None and not force_reload:
            return self._manifest_cache

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._manifest_cache = data
        return data

    def get_invoices_for_split(
        self,
        split_name: str,
        allow_contaminated: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all invoice metadata records assigned to a specific split.
        Refuses 'locked_test' unless RRSI_FINAL_EVAL=1 is present in the environment.
        """
        normalized_split = split_name.strip().lower()

        # Gate 'locked_test'
        if normalized_split in ("locked_test", "test"):
            final_eval_env = os.environ.get("RRSI_FINAL_EVAL", "").strip()
            if final_eval_env != "1":
                raise PermissionError(
                    "ACCESS DENIED: 'locked_test' split is locked against premature evaluation. "
                    "Access is only permitted when environment variable RRSI_FINAL_EVAL=1 is explicitly set."
                )

        manifest_data = self.load_manifest()
        raw_manifest = manifest_data.get("manifest", [])

        # Support both list of dicts and dict of dicts
        items: List[Dict[str, Any]] = []
        if isinstance(raw_manifest, list):
            items = raw_manifest
        elif isinstance(raw_manifest, dict):
            items = list(raw_manifest.values())

        selected: List[Dict[str, Any]] = []
        for inv in items:
            inv_split = str(inv.get("split", "")).strip().lower()
            if inv_split == normalized_split:
                is_contaminated = bool(inv.get("contaminated", False))
                if is_contaminated and normalized_split in ("selection", "locked_test") and not allow_contaminated:
                    raise ValueError(
                        f"INTEGRITY VIOLATION: Contaminated invoice '{inv.get('doc_id')}' found in '{normalized_split}' split!"
                    )
                selected.append(inv)

        return selected

    def verify_manifest_integrity(
        self,
        check_disk_hashes: bool = False,
        images_dir: Optional[Path] = None
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive manifest validation:
        1. Split overlap: no doc_id appears in multiple splits.
        2. Template straddling: no template_group straddles across splits.
        3. Copy integrity: copies belonging to the same invoice share the same split.
        4. Contamination: no contaminated invoices in selection or locked_test.
        5. Hash validation (optional): file sha256 matches manifest.
        """
        errors: List[str] = []
        manifest_data = self.load_manifest()
        raw_manifest = manifest_data.get("manifest", [])

        items: List[Dict[str, Any]] = []
        if isinstance(raw_manifest, list):
            items = raw_manifest
        elif isinstance(raw_manifest, dict):
            items = list(raw_manifest.values())

        doc_splits: Dict[str, str] = {}
        template_splits: Dict[str, Set[str]] = {}
        file_to_doc: Dict[str, str] = {}

        for inv in items:
            doc_id = inv.get("doc_id")
            split = str(inv.get("split", "")).strip().lower()
            tmpl = inv.get("template_group")
            contaminated = bool(inv.get("contaminated", False))
            files = inv.get("files", [])

            if not doc_id:
                errors.append("Manifest entry missing 'doc_id'")
                continue

            # 1. Split overlap
            if doc_id in doc_splits and doc_splits[doc_id] != split:
                errors.append(f"Split overlap: doc_id '{doc_id}' in both '{doc_splits[doc_id]}' and '{split}'")
            doc_splits[doc_id] = split

            # 2. Template straddling
            if tmpl:
                template_splits.setdefault(tmpl, set()).add(split)

            # 3. Contamination guard
            if contaminated and split in ("selection", "locked_test"):
                errors.append(f"Contaminated invoice '{doc_id}' found in forbidden split '{split}'")

            # 4. File-to-doc mapping for copies
            for fname in files:
                if fname in file_to_doc and file_to_doc[fname] != doc_id:
                    errors.append(f"File '{fname}' assigned to multiple doc_ids: '{file_to_doc[fname]}' and '{doc_id}'")
                file_to_doc[fname] = doc_id

            # 5. Check SHA-256 on disk if requested
            if check_disk_hashes and images_dir and images_dir.exists():
                file_hashes = inv.get("file_hashes", {})
                for fname in files:
                    expected_sha = file_hashes.get(fname)
                    img_file = images_dir / fname
                    if img_file.exists() and expected_sha:
                        actual_sha = hashlib.sha256(img_file.read_bytes()).hexdigest()
                        if actual_sha != expected_sha:
                            errors.append(f"SHA-256 mismatch on '{fname}': expected {expected_sha[:10]}, got {actual_sha[:10]}")

        # Check template straddling
        for tmpl, splits in template_splits.items():
            if len(splits) > 1:
                errors.append(f"Template straddling: template_group '{tmpl}' straddles multiple splits: {sorted(list(splits))}")

        is_valid = (len(errors) == 0)
        return is_valid, errors
