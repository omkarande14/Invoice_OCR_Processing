# scripts/build_manifest_and_splits.py
"""
Seeded Split Assignment & Annotation Queue Generator (Phase 1.2b).
- Contamination propagates to entire template group (assigned to dev).
- Group-level seeded splitting: dev, selection, locked_test.
- Writes git-safe data/manifest.json and private data_private/manifest_private.json.
- Creates blank annotation templates and prioritized annotation queue.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Set

CONTAMINATED_KEYWORDS = [
    "4531", "196", "1415", "170588", "3930", "264882", "8497", "8184", "T07"
]

def generate_splits_and_manifest():
    private_dir = Path("data_private")
    mapping_file = private_dir / "template_mapping.json"
    if not mapping_file.exists():
        print(f"[!] File not found: {mapping_file}. Run scripts/build_template_hierarchy.py first.")
        return

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    logical_invoices = mapping_data["logical_invoices"]

    # 1. Identify Contaminated Groups
    contaminated_groups: Set[str] = set()
    for base_id, inv_data in logical_invoices.items():
        lower_base = base_id.lower()
        if any(k in lower_base for k in CONTAMINATED_KEYWORDS):
            contaminated_groups.add(inv_data["template_group"])

    print(f"[*] Contaminated template groups (propagated to dev only): {sorted(contaminated_groups)}")

    # Group logical invoices by template group
    group_to_invoices: Dict[str, List[str]] = {}
    for base_id, inv_data in logical_invoices.items():
        group_to_invoices.setdefault(inv_data["template_group"], []).append(base_id)

    all_groups = sorted(group_to_invoices.keys())
    uncontaminated_groups = [g for g in all_groups if g not in contaminated_groups]

    # Seeded split allocation for uncontaminated groups
    rng = random.Random(42)
    rng.shuffle(uncontaminated_groups)

    # Assign uncontaminated groups to meet targets:
    # dev gets all contaminated groups + some uncontaminated
    # selection target >= 12 invoices, locked_test target >= 8 invoices
    group_splits: Dict[str, str] = {}
    for g in contaminated_groups:
        group_splits[g] = "dev"

    selection_groups = []
    locked_test_groups = []
    dev_extra_groups = []

    sel_inv_count = 0
    test_inv_count = 0

    for g in uncontaminated_groups:
        count = len(group_to_invoices[g])
        if sel_inv_count < 15 and len(selection_groups) < 6:
            group_splits[g] = "selection"
            selection_groups.append(g)
            sel_inv_count += count
        elif test_inv_count < 10 and len(locked_test_groups) < 4:
            group_splits[g] = "locked_test"
            locked_test_groups.append(g)
            test_inv_count += count
        else:
            group_splits[g] = "dev"
            dev_extra_groups.append(g)

    # Assign Tier levels:
    # Target: All target invoices get Tier A (header, totals, GSTIN)
    # At least 15 invoices get Tier B (full line items), prioritizing selection and test
    tier_b_count = 0
    manifest_records = []
    private_manifest_records = []
    annotation_queue = []

    for base_id, inv_data in sorted(logical_invoices.items()):
        grp = inv_data["template_group"]
        split = group_splits[grp]
        is_contaminated = grp in contaminated_groups

        # Determine Tier
        if split in ["selection", "locked_test"] and tier_b_count < 15:
            tier = "Tier_B"
            tier_b_count += 1
        elif is_contaminated and base_id in ["4531", "264882"]:
            tier = "Tier_B"
        else:
            tier = "Tier_A"

        # Git-safe record (hashes and anonymous IDs only)
        git_safe_rec = {
            "doc_id": inv_data["anonymous_id"],
            "template_group": grp,
            "split": split,
            "tier": tier,
            "page_count": inv_data["page_count"],
            "annotation_status": "annotated" if base_id in ["4531", "264882"] else "unannotated",
            "is_contaminated": is_contaminated
        }
        manifest_records.append(git_safe_rec)

        # Private record with actual file paths
        private_rec = dict(git_safe_rec)
        private_rec["logical_base_id"] = base_id
        private_rec["files"] = inv_data["files"]
        private_manifest_records.append(private_rec)

        # Add to annotation queue if unannotated
        if git_safe_rec["annotation_status"] == "unannotated":
            # Priority: selection first, then locked_test, then dev
            priority = 1 if split == "selection" else (2 if split == "locked_test" else 3)
            labor_mins = 10 if tier == "Tier_B" else 3
            annotation_queue.append({
                "priority": priority,
                "split": split,
                "template_group": grp,
                "anonymous_id": inv_data["anonymous_id"],
                "logical_base": base_id,
                "tier": tier,
                "files": inv_data["files"],
                "est_labor_minutes": labor_mins
            })

    # Sort queue by priority and template diversity
    annotation_queue.sort(key=lambda x: (x["priority"], x["template_group"], x["anonymous_id"]))

    # 2. Write Git-Safe data/manifest.json
    manifest_path = Path("data/manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest_records}, f, indent=2)

    # 3. Write Private Manifest
    private_manifest_path = private_dir / "manifest_private.json"
    with open(private_manifest_path, "w", encoding="utf-8") as f:
        json.dump({"manifest": private_manifest_records}, f, indent=2)

    # 4. Generate Blank Annotation Templates
    template_dir = private_dir / "annotation_templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    blank_tier_a = {
        "metadata": {
            "annotator": "",
            "date": "YYYY-MM-DD",
            "source": "from_image_blind",
            "tier": "Tier_A"
        },
        "invoice_number": "",
        "invoice_date": "YYYY-MM-DD",
        "due_date": None,
        "PO_number": None,
        "customer_name": "",
        "customer_gstin": "",
        "seller_name": "",
        "seller_gstin": "",
        "invoice_amount": "0.00",
        "total_gst_amount": "0.00",
        "round_off": "0.00"
    }
    with open(template_dir / "blank_tier_a.json", "w", encoding="utf-8") as f:
        json.dump(blank_tier_a, f, indent=2)

    blank_tier_b = dict(blank_tier_a)
    blank_tier_b["metadata"]["tier"] = "Tier_B"
    blank_tier_b["items"] = [
        {
            "description": "",
            "hsn_sac": "",
            "quantity": 1,
            "unit_price": "0.00",
            "total_price": "0.00",
            "Discount": "0.00",
            "Gst%": 5
        }
    ]
    with open(template_dir / "blank_tier_b.json", "w", encoding="utf-8") as f:
        json.dump(blank_tier_b, f, indent=2)

    # Save queue to private dir
    queue_file = private_dir / "annotation_queue.json"
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump({"annotation_queue": annotation_queue}, f, indent=2)

    # Summary calculations
    split_counts = {"dev": 0, "selection": 0, "locked_test": 0}
    split_groups = {"dev": set(), "selection": set(), "locked_test": set()}
    for r in manifest_records:
        split_counts[r["split"]] += 1
        split_groups[r["split"]].add(r["template_group"])

    total_queue_mins = sum(item["est_labor_minutes"] for item in annotation_queue)

    print("\n[+] SPLIT ALLOCATION SUMMARY:")
    print(f"    - Dev split:         {split_counts['dev']} invoices across {len(split_groups['dev'])} template groups")
    print(f"    - Selection split:   {split_counts['selection']} invoices across {len(split_groups['selection'])} template groups")
    print(f"    - Locked Test split: {split_counts['locked_test']} invoices across {len(split_groups['locked_test'])} template groups")
    print(f"    - Total Queue Items: {len(annotation_queue)} unannotated invoices")
    print(f"    - Est. Total Labor:  {total_queue_mins / 60.0:.1f} hours ({total_queue_mins} minutes)")
    print(f"    - Git-Safe manifest: {manifest_path}")
    print(f"    - Private manifest:  {private_manifest_path}")
    print(f"    - Queue file:        {queue_file}")

if __name__ == "__main__":
    generate_splits_and_manifest()
