# scripts/check_git_hygiene.py
"""
Git Hygiene and PII Leakage Check.
Fails on:
1. Real GSTINs matching \b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b
2. Real PAN numbers matching \b[A-Z]{5}\d{4}[A-Z]\b
3. Indian phone numbers
4. Salted hashes of sensitive vendor and buyer names (from config/forbidden_hashes.json)
5. Tracked private data paths (data_private/, data/deskewed/, venv/, .venv/, etc.)
"""

import re
import subprocess
import sys
import json
import hashlib
from pathlib import Path

# RegEx for sensitive identifiers
GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
PAN_REGEX = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
PHONE_REGEX = re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b|\b0\d{2,4}[\-\s]?\d{6,8}\b")

# Allowed synthetic test fixtures
SYNTHETIC_WHITELIST = {
    "27AAPFU0939F1ZV",  # Standard statutory dummy test GSTIN
    "27AAPFU0939F1ZW",  # Intentional invalid test GSTIN
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
    "private_snapshots"
]

def load_forbidden_hashes():
    cfg_path = Path("config/forbidden_hashes.json")
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("salt", ""), set(data.get("hashes", []))
        except Exception:
            pass
    return "", set()

def get_staged_or_tracked_files() -> list[str]:
    # Check staged files if any, else check tracked files
    res = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True)
    files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    if not files:
        res2 = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
        files = [f.strip() for f in res2.stdout.splitlines() if f.strip()]
    return files

def check_hygiene() -> bool:
    files = get_staged_or_tracked_files()
    violations = []
    salt, forbidden_hashes = load_forbidden_hashes()

    for f_str in files:
        p = Path(f_str)
        # Check path restrictions
        norm_path = f_str.replace("\\", "/")
        for forbidden in FORBIDDEN_PATH_SUBSTRINGS:
            if forbidden in norm_path:
                violations.append(f"FORBIDDEN FILE TRACKED: '{f_str}' is inside private/ignored directory '{forbidden}'.")

        # Skip binary files, gitignore, forbidden hash file, and this hygiene checker itself
        if f_str in ["scripts/check_git_hygiene.py", "config/forbidden_hashes.json"] or p.suffix.lower() in [".png", ".jpg", ".pdf", ".pyc", ".zip"]:
            continue

        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            lines = content.splitlines()
            for line_no, line in enumerate(lines, 1):
                # Ignore lines defining regexes
                if "re.compile" in line or "REGEX" in line or "pattern" in line:
                    continue

                for match in GSTIN_REGEX.findall(line):
                    if match not in SYNTHETIC_WHITELIST:
                        violations.append(f"PII LEAK (GSTIN): Found '{match}' in {f_str}:{line_no}")

                for match in PAN_REGEX.findall(line):
                    if match not in SYNTHETIC_WHITELIST:
                        violations.append(f"PII LEAK (PAN): Found '{match}' in {f_str}:{line_no}")

                for match in PHONE_REGEX.findall(line):
                    if match not in SYNTHETIC_WHITELIST:
                        violations.append(f"PII LEAK (PHONE): Found '{match}' in {f_str}:{line_no}")

                # Salted hash name check (single tokens and bigrams)
                if salt and forbidden_hashes:
                    tokens = [t.lower().strip(" \t\r\n'\",;:.()[]{}") for t in line.split() if len(t.strip()) >= 4]
                    for token in tokens:
                        thash = hashlib.sha256((salt + token).encode("utf-8")).hexdigest()
                        if thash in forbidden_hashes:
                            violations.append(f"FORBIDDEN ENTITY NAME (Salted Hash Match): '{token}' in {f_str}:{line_no}")
                    # Check bigrams
                    for i in range(len(tokens) - 1):
                        bigram = f"{tokens[i]} {tokens[i+1]}"
                        bhash = hashlib.sha256((salt + bigram).encode("utf-8")).hexdigest()
                        if bhash in forbidden_hashes:
                            violations.append(f"FORBIDDEN ENTITY NAME (Salted Hash Match): '{bigram}' in {f_str}:{line_no}")

    if violations:
        print("[!] GIT HYGIENE CHECK FAILED:")
        for v in violations:
            print(f"    - {v}")
        return False

    print("[+] Git hygiene check passed cleanly: No un-anonymized PII, real GSTINs, or private data tracked.")
    return True

if __name__ == "__main__":
    if not check_hygiene():
        sys.exit(1)
    sys.exit(0)
