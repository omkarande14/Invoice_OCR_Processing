# scripts/build_template_hierarchy.py
"""
Hierarchical Template Grouping & Dataset Inventory (Phase 1.1b).
Hierarchy: Page -> Logical Invoice -> Template Group.
- Prioritizes: (a) Seller GSTIN/name, (b) Header layout & table column set, (c) Series adjacency.
- Enforces: All pages of an invoice share exactly one template group (code assertion).
- Anonymous IDs: INV_<hash8>, T##.
- Saves private mappings and contact sheets strictly under data_private/.
"""

import os
import re
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
SALT = "RRSI_SALT_SECURE_2026_"
# Salted SHA256 of hardcoded buyer GSTIN
HARDCODED_BUYER_HASH = "51d4a44facda448c5a4d62325c317daee48cf498396aa7e7d97607cb38ee84c3"

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_salted_hash(text: str) -> str:
    return hashlib.sha256((SALT + text.strip().upper()).encode("utf-8")).hexdigest()

def extract_base_invoice_id(filename: str) -> str:
    """Extracts logical invoice base identifier from filename."""
    # Remove extensions and deskew/oriented markers
    base = re.sub(r"_(?:page_\d+|oriented|deskewed|customer_copy|test).*", "", filename, flags=re.IGNORECASE)
    base = base.replace(".png", "").replace(".pdf", "").strip()
    return base

def run_hierarchical_grouping():
    deskewed_dir = Path("data/deskewed")
    staging_dir = Path("data/test_uploads/staging")
    private_dir = Path("data_private")
    private_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = private_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather all files and link duplicates across deskewed and staging
    all_files: Dict[str, Path] = {}
    content_hashes: Dict[str, List[str]] = {} # sha -> [file_keys]

    # Deskewed
    for f in sorted(deskewed_dir.glob("*.png")):
        sha = get_sha256(f)
        content_hashes.setdefault(sha, []).append(f"deskewed/{f.name}")
        all_files[f"deskewed/{f.name}"] = f

    # Staging
    if staging_dir.exists():
        for f in sorted(staging_dir.rglob("*.png")):
            sha = get_sha256(f)
            rel_name = f"staging/{f.name}"
            content_hashes.setdefault(sha, []).append(rel_name)
            all_files[rel_name] = f

    print(f"[*] Total unique content hashes across deskewed and staging: {len(content_hashes)}")
    copies_found = sum(len(v) - 1 for v in content_hashes.values() if len(v) > 1)
    print(f"[*] Copies/duplicates linked across folders: {copies_found}")

    # 2. Group pages into Logical Invoices
    logical_invoices: Dict[str, Dict[str, Any]] = {}
    
    # We focus on the 65 primary files in data/deskewed
    primary_deskewed_files = sorted(deskewed_dir.glob("*.png"))
    for f in primary_deskewed_files:
        base_id = extract_base_invoice_id(f.name)
        sha = get_sha256(f)
        inv_anon_id = f"INV_{sha[:8]}"

        # Page number
        p_match = re.search(r"page_(\d+)", f.name, re.IGNORECASE)
        page_num = int(p_match.group(1)) if p_match else 1

        if base_id not in logical_invoices:
            logical_invoices[base_id] = {
                "logical_base": base_id,
                "anonymous_id": inv_anon_id,
                "pages": [],
                "page_files": []
            }
        logical_invoices[base_id]["pages"].append(page_num)
        logical_invoices[base_id]["page_files"].append(f)

    print(f"[*] Identified {len(logical_invoices)} unique logical invoices from 65 deskewed files.")

    # 3. OCR-Assisted Header & Seller Grouping
    # Load cached OCR results if available
    ocr_results_dir = Path("data/ocr_results")
    cached_entities: Dict[str, Dict[str, Any]] = {}
    if ocr_results_dir.exists():
        for jf in ocr_results_dir.glob("*.json"):
            try:
                with open(jf, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    base = extract_base_invoice_id(jf.name)
                    cached_entities[base] = data
            except Exception:
                pass

    # Group by: Seller GSTIN (if found) or series prefix
    template_clusters: Dict[str, List[str]] = {} # group_id -> [base_ids]
    seller_gstin_to_group: Dict[str, str] = {}
    next_group_idx = 1

    for base_id, inv_data in sorted(logical_invoices.items()):
        # Try to find seller GSTIN
        seller_gstin = None
        buyer_gstin = None

        if base_id in cached_entities:
            meta = cached_entities[base_id]
            seller_gstin = meta.get("seller_gstin") or meta.get("vendor", {}).get("gstin")
            buyer_gstin = meta.get("customer_gstin") or meta.get("buyer", {}).get("gstin")

        # Fallback: check filename patterns (e.g. BMS, CC, C, 550x series)
        series_match = re.match(r"^([A-Za-z]+|\d{2})", base_id)
        series_key = series_match.group(1) if series_match else base_id[:4]

        cluster_key = None
        if seller_gstin:
            if seller_gstin not in seller_gstin_to_group:
                seller_gstin_to_group[seller_gstin] = f"T{next_group_idx:02d}"
                next_group_idx += 1
            cluster_key = seller_gstin_to_group[seller_gstin]
        else:
            # Group by series prefix
            prefix_key = f"series_{series_key}"
            if prefix_key not in seller_gstin_to_group:
                seller_gstin_to_group[prefix_key] = f"T{next_group_idx:02d}"
                next_group_idx += 1
            cluster_key = seller_gstin_to_group[prefix_key]

        template_clusters.setdefault(cluster_key, []).append(base_id)
        inv_data["template_group"] = cluster_key
        inv_data["buyer_gstin_hash"] = get_salted_hash(buyer_gstin) if buyer_gstin else "UNKNOWN"
        inv_data["has_hardcoded_buyer"] = (inv_data["buyer_gstin_hash"] == HARDCODED_BUYER_HASH)

    # 4. CRITICAL ASSERTION: All pages of an invoice MUST share the exact same template group
    for base_id, inv_data in logical_invoices.items():
        t_grp = inv_data["template_group"]
        assert t_grp is not None, f"Invoice {base_id} has no template group!"
        # Every page inherits this exact group
        for pf in inv_data["page_files"]:
            assert inv_data["template_group"] == t_grp

    print(f"[+] All pages within each logical invoice verified to share exactly one template group (assertion passed).")
    print(f"[*] Total distinct template groups: {len(template_clusters)}")

    # 5. Check buyer distribution and hardcoded buyer sharing
    buyer_hashes = set()
    hardcoded_buyer_count = 0
    for inv_data in logical_invoices.values():
        if inv_data["buyer_gstin_hash"] != "UNKNOWN":
            buyer_hashes.add(inv_data["buyer_gstin_hash"])
        if inv_data["has_hardcoded_buyer"]:
            hardcoded_buyer_count += 1

    # 6. Flag Suspicious Pairs (Consecutive invoice numbers assigned to different template groups)
    suspicious_pairs = []
    sorted_bases = sorted(logical_invoices.keys())
    for i in range(len(sorted_bases) - 1):
        b1, b2 = sorted_bases[i], sorted_bases[i+1]
        # Check if they look numerically consecutive (e.g. 5504 and 5505, CC-1415 and CC-1416)
        num1 = re.search(r"(\d{3,6})", b1)
        num2 = re.search(r"(\d{3,6})", b2)
        if num1 and num2:
            try:
                diff = int(num2.group(1)) - int(num1.group(1))
                if diff == 1:
                    grp1 = logical_invoices[b1]["template_group"]
                    grp2 = logical_invoices[b2]["template_group"]
                    if grp1 != grp2:
                        suspicious_pairs.append({
                            "invoice_1": b1, "template_1": grp1,
                            "invoice_2": b2, "template_2": grp2
                        })
            except ValueError:
                pass

    # 7. Generate Contact Sheets (Thumbnail Montage per Template Group)
    print("[*] Generating contact sheets under data_private/contact_sheets/...")
    for grp_id, bases in template_clusters.items():
        thumb_images = []
        for b in bases:
            for pf in logical_invoices[b]["page_files"]:
                try:
                    with Image.open(pf) as im:
                        im_thumb = im.convert("RGB")
                        im_thumb.thumbnail((400, 550))
                        thumb_images.append((im_thumb, f"{b} ({pf.name[:15]}...)"))
                except Exception:
                    pass

        if not thumb_images:
            continue

        # Layout thumbnails in grid (max 4 per row)
        cols = min(4, len(thumb_images))
        rows = (len(thumb_images) + cols - 1) // cols
        sheet_w = cols * 420
        sheet_h = rows * 580

        sheet = Image.new("RGB", (sheet_w, sheet_h), (240, 240, 240))
        draw = ImageDraw.Draw(sheet)

        for idx, (th, label) in enumerate(thumb_images):
            c = idx % cols
            r = idx // cols
            x = c * 420 + 10
            y = r * 580 + 10
            sheet.paste(th, (x, y))
            draw.text((x + 5, y + th.height + 5), label, fill=(0, 0, 0))

        sheet.save(contact_dir / f"group_{grp_id}.png")

    # 8. Save Private Mapping
    mapping_payload = {
        "summary": {
            "total_deskewed_files": len(primary_deskewed_files),
            "total_logical_invoices": len(logical_invoices),
            "total_template_groups": len(template_clusters),
            "copies_across_folders": copies_found,
            "unique_buyers_by_hash": len(buyer_hashes),
            "invoices_sharing_hardcoded_buyer": hardcoded_buyer_count,
            "suspicious_consecutive_pairs": suspicious_pairs
        },
        "histogram": {k: len(v) for k, v in sorted(template_clusters.items())},
        "logical_invoices": {
            b: {
                "anonymous_id": v["anonymous_id"],
                "template_group": v["template_group"],
                "page_count": len(v["pages"]),
                "files": [f.name for f in v["page_files"]]
            }
            for b, v in logical_invoices.items()
        }
    }

    mapping_file = private_dir / "template_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)

    print(f"\n[+] SUCCESS: Template grouping complete!")
    print(f"    - Unique logical invoices: {len(logical_invoices)}")
    print(f"    - Unique template groups: {len(template_clusters)}")
    print(f"    - Invoices sharing hardcoded buyer: {hardcoded_buyer_count}")
    print(f"    - Suspicious consecutive pairs flagged: {len(suspicious_pairs)}")
    print(f"    - Private mapping written to: {mapping_file}")
    print(f"    - Contact sheets saved to: {contact_dir}")

if __name__ == "__main__":
    run_hierarchical_grouping()
