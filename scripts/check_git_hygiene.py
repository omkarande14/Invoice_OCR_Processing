# scripts/check_git_hygiene.py
"""
Git Hygiene and PII Leakage Check.
Fails on:
1. Real GSTINs matching \b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b
2. Real PAN numbers matching \b[A-Z]{5}\d{4}[A-Z]\b
3. Indian phone numbers
4. Salted hashes of sensitive vendor and buyer names (when data_private/config/forbidden_hashes.json exists)
5. Tracked private data paths (data_private/, data/deskewed/, venv/, .venv/, etc.)
Supports:
--history : Scans all historical commit messages and blob objects across repository history.
"""

import re
import subprocess
import sys
import json
import hashlib
import argparse
from pathlib import Path

# RegEx for sensitive identifiers
GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
PAN_REGEX = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
PHONE_REGEX = re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b|\b0\d{2,4}[\-\s]?\d{6,8}\b")

# Allowed synthetic test fixtures
SYNTHETIC_WHITELIST = {
    "27AAPFU0939F1ZV",  # Standard statutory dummy test GSTIN
    "27AAPFU0939F1ZW",  # Intentional invalid test GSTIN
    "99AAPFU0939F1ZV",  # Intentional invalid state test GSTIN
    "AAPFU0939F",       # Dummy test PAN
    "020-00000000",
    "9999999999"
}

FORBIDDEN_PATH_SUBSTRINGS = [
    "data_private",
    "data/deskewed",
    "data/ground_truth",
    "data/ocr_results",
    "venv/",
    ".venv/",
    "scratch/",
    "private_snapshots",
    "golden_outputs"
]

def load_forbidden_hashes():
    # Priority: private config path, then legacy config path
    for p in [Path("data_private/config/forbidden_hashes.json"), Path("config/forbidden_hashes.json")]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("salt", ""), set(data.get("hashes", []))
            except Exception:
                pass
    return "", set()

def run_synthetic_canary_test() -> bool:
    """Verifies that the regex checkers are functional and catch synthetic leaks."""
    fake_gstin = "27ABCDE1234F1Z5"
    fake_pan = "ABCDE1234F"
    fake_phone = "9822012345"

    assert bool(GSTIN_REGEX.search(fake_gstin)), "Canary failed: GSTIN regex broken"
    assert bool(PAN_REGEX.search(fake_pan)), "Canary failed: PAN regex broken"
    assert bool(PHONE_REGEX.search(fake_phone)), "Canary failed: PHONE regex broken"
    assert fake_gstin not in SYNTHETIC_WHITELIST, "Canary failed: Whitelist collision"
    return True

def scan_text_lines(text: str, source_label: str, salt: str, forbidden_hashes: set) -> list[str]:
    violations = []
    lines = text.splitlines()
    for line_no, line in enumerate(lines, 1):
        if "re.compile" in line or "REGEX" in line or "pattern" in line or "SYNTHETIC_WHITELIST" in line:
            continue

        for match in GSTIN_REGEX.findall(line):
            if match not in SYNTHETIC_WHITELIST:
                violations.append(f"PII LEAK (GSTIN): Found '{match}' in {source_label}:{line_no}")

        for match in PAN_REGEX.findall(line):
            if match not in SYNTHETIC_WHITELIST:
                violations.append(f"PII LEAK (PAN): Found '{match}' in {source_label}:{line_no}")

        for match in PHONE_REGEX.findall(line):
            if match not in SYNTHETIC_WHITELIST:
                violations.append(f"PII LEAK (PHONE): Found '{match}' in {source_label}:{line_no}")

        if salt and forbidden_hashes:
            tokens = [t.lower().strip(" \t\r\n'\",;:.()[]{}") for t in line.split() if len(t.strip()) >= 4]
            for token in tokens:
                thash = hashlib.sha256((salt + token).encode("utf-8")).hexdigest()
                if thash in forbidden_hashes:
                    violations.append(f"FORBIDDEN ENTITY (Salted Hash): '{token}' in {source_label}:{line_no}")
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                bhash = hashlib.sha256((salt + bigram).encode("utf-8")).hexdigest()
                if bhash in forbidden_hashes:
                    violations.append(f"FORBIDDEN ENTITY (Salted Hash): '{bigram}' in {source_label}:{line_no}")

    return violations

def get_staged_or_tracked_files() -> list[str]:
    res = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True)
    files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    if not files:
        res2 = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
        files = [f.strip() for f in res2.stdout.splitlines() if f.strip()]
    return files

def check_current_hygiene(specified_files: list[str] = None) -> bool:
    run_synthetic_canary_test()
    files = specified_files if specified_files else get_staged_or_tracked_files()
    violations = []
    salt, forbidden_hashes = load_forbidden_hashes()

    if not forbidden_hashes:
        print("[*] Running in Public/CI mode: Regex + Synthetic Canaries active (private hash list omitted).")

    for f_str in files:
        p = Path(f_str)
        norm_path = f_str.replace("\\", "/")
        for forbidden in FORBIDDEN_PATH_SUBSTRINGS:
            if forbidden in norm_path:
                violations.append(f"FORBIDDEN FILE TRACKED: '{f_str}' is inside private/ignored directory '{forbidden}'.")

        if f_str in ["scripts/check_git_hygiene.py", "config/forbidden_hashes.json"] or p.suffix.lower() in [".png", ".jpg", ".pdf", ".pyc", ".zip"]:
            continue

        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                violations.extend(scan_text_lines(content, f_str, salt, forbidden_hashes))
            except Exception:
                continue

    if violations:
        print("[!] GIT HYGIENE CHECK FAILED:")
        for v in violations:
            print(f"    - {v}")
        return False

    print(f"[+] Git hygiene check passed cleanly: No un-anonymized PII, real GSTINs, or private data tracked ({len(files)} files checked).")
    return True

def scan_full_history() -> bool:
    print("[*] Performing full git history scan across all commits and blobs...")
    run_synthetic_canary_test()
    salt, forbidden_hashes = load_forbidden_hashes()
    violations = []

    res_commits = subprocess.run(["git", "log", "--all", "--format=%H"], capture_output=True, text=True)
    commits = [c.strip() for c in res_commits.stdout.splitlines() if c.strip()]

    print(f"    Scanning {len(commits)} commits for commit message and file blob leaks...")
    scanned_blobs = set()

    for commit in commits:
        # Check commit message
        res_msg = subprocess.run(["git", "log", "-1", "--format=%B", commit], capture_output=True, text=True)
        msg_violations = scan_text_lines(res_msg.stdout, f"commit {commit[:7]} message", salt, forbidden_hashes)
        violations.extend(msg_violations)

    # Walk every blob object reachable from any ref
    res_objs = subprocess.run(["git", "rev-list", "--objects", "--all"], capture_output=True, text=True)
    for line in res_objs.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        obj_hash = parts[0]
        obj_path = parts[1] if len(parts) > 1 else ""

        # Check object type
        res_type = subprocess.run(["git", "cat-file", "-t", obj_hash], capture_output=True, text=True)
        if res_type.stdout.strip() != "blob":
            continue

        if obj_path:
            norm_path = obj_path.replace("\\", "/")
            for forbidden in FORBIDDEN_PATH_SUBSTRINGS:
                if forbidden in norm_path:
                    violations.append(f"HISTORICAL FORBIDDEN FILE: '{obj_path}' ({obj_hash[:7]})")

        if obj_hash in scanned_blobs:
            continue
        scanned_blobs.add(obj_hash)

        if any(obj_path.endswith(ext) for ext in [".png", ".jpg", ".pdf", ".pyc", ".zip"]):
            continue
        if obj_path in ["scripts/check_git_hygiene.py", "config/forbidden_hashes.json"]:
            continue

        res_blob = subprocess.run(["git", "cat-file", "-p", obj_hash], capture_output=True, text=True, errors="ignore")
        blob_violations = scan_text_lines(res_blob.stdout, f"blob {obj_hash[:7]} ({obj_path or 'anonymous'})", salt, forbidden_hashes)
        violations.extend(blob_violations)

    if violations:
        print("[!] FULL GIT HISTORY AUDIT FAILED:")
        for v in violations:
            print(f"    - {v}")
        return False

    print(f"[+] Full git history audit PASSED cleanly: {len(commits)} commits and {len(scanned_blobs)} blobs verified zero-leak.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Git Hygiene and PII Leakage Auditor")
    parser.add_argument("--history", action="store_true", help="Scan entire repository commit history and blob tree")
    parser.add_argument("files", nargs="*", default=None, help="Files to scan (passed by pre-commit)")
    args = parser.parse_args()

    success = scan_full_history() if args.history else check_current_hygiene(args.files)
    if not success:
        sys.exit(1)
    sys.exit(0)
