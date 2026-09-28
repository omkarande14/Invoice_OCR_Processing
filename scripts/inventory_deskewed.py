# scripts/inventory_deskewed.py
import re
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import imagehash

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing.template_similarity_detector import find_template_similar_images, compute_perceptual_hash

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def parse_filename_metadata(filename: str) -> Dict[str, Any]:
    # Extract page number
    page_match = re.search(r"page_(\d+)", filename, re.IGNORECASE)
    page_num = int(page_match.group(1)) if page_match else 1

    # Extract copy role
    role = "page"
    lower_name = filename.lower()
    if "customer_copy" in lower_name or "customer copy" in lower_name:
        role = "customer_copy"
    elif "transporter" in lower_name:
        role = "transporter_copy"
    elif "duplicate" in lower_name:
        role = "duplicate_copy"

    # Derive logical invoice prefix
    # e.g. "5504_page_1..." -> "INV_5504", "CC-1415_page_1..." -> "INV_CC-1415"
    base_prefix = filename.split("_page_")[0].split("_oriented")[0].strip()
    # Clean up commas or spaces
    cleaned_prefix = re.sub(r"[^\w\-]", "_", base_prefix).strip("_")

    return {
        "page_num": page_num,
        "role": role,
        "logical_id": f"DOC_{cleaned_prefix}"
    }

def build_inventory():
    deskewed_dir = Path("data/deskewed")
    if not deskewed_dir.exists():
        print(f"[!] Directory not found: {deskewed_dir}")
        return

    files = sorted([f for f in deskewed_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"])
    print(f"[*] Found {len(files)} PNG files in {deskewed_dir}")

    # 1. Run template clustering using pHash (Hamming distance <= 4)
    cluster_res = find_template_similar_images(str(deskewed_dir), threshold=4)
    groups = cluster_res.get("template_similarity_groups", [])

    # Map file -> template_group_id
    file_to_group = {}
    for grp_idx, group_files in enumerate(groups, 1):
        grp_name = f"T{grp_idx:02d}"
        for f in group_files:
            file_to_group[f] = grp_name

    # Unclustered files get their own singleton template group
    next_singleton = len(groups) + 1
    for f in files:
        if f.name not in file_to_group:
            file_to_group[f.name] = f"T{next_singleton:02d}"
            next_singleton += 1

    # 2. Build detailed records
    inventory_records = []
    logical_invoices: Dict[str, List[str]] = {}
    template_counts: Dict[str, int] = {}
    copy_count = 0

    for f in files:
        meta = parse_filename_metadata(f.name)
        sha = sha256_file(f)
        phash_str = str(compute_perceptual_hash(str(f)))
        t_group = file_to_group[f.name]

        template_counts[t_group] = template_counts.get(t_group, 0) + 1
        logical_invoices.setdefault(meta["logical_id"], []).append(f.name)
        if meta["role"] != "page" or "copy" in meta["role"]:
            copy_count += 1

        record = {
            "filename": f.name,
            "logical_invoice_id": meta["logical_id"],
            "page_num": meta["page_num"],
            "role": meta["role"],
            "template_group": t_group,
            "anonymous_vendor": f"Vendor_{t_group}",
            "sha256": sha,
            "phash": phash_str
        }
        inventory_records.append(record)

    summary = {
        "total_files": len(files),
        "total_unique_logical_invoices": len(logical_invoices),
        "total_unique_template_groups": len(set(file_to_group.values())),
        "total_copy_files": copy_count,
        "invoices_per_template": template_counts
    }

    rep_dir = Path("reports")
    rep_dir.mkdir(exist_ok=True, parents=True)
    out_file = rep_dir / "deskewed_inventory.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": inventory_records}, f, indent=2)

    print(f"[+] Inventory complete. Saved to {out_file}")
    print(f"    Total files: {summary['total_files']}")
    print(f"    Unique logical invoices: {summary['total_unique_logical_invoices']}")
    print(f"    Unique template groups: {summary['total_unique_template_groups']}")
    print(f"    Copies identified: {summary['total_copy_files']}")

if __name__ == "__main__":
    build_inventory()
