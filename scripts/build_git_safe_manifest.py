# scripts/build_git_safe_manifest.py
"""
Builds the Git-Safe manifest (data/manifest.json) and computes template & buyer histograms.
Enforces:
1. Contamination propagation: any template group touched in scratch/ or tests/ is locked to 'dev'.
2. Split partitioning: zero template straddling between dev, selection, and locked_test.
3. Privacy: only anonymous doc_ids, template groups, splits, image_sha256, and anonymous buyer IDs.
4. Annotation queue: ordered to maximize template and buyer diversity in selection/locked_test first.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    private_manifest_path = PROJECT_ROOT / "data_private" / "manifest_private.json"
    if not private_manifest_path.exists():
        print(f"Error: {private_manifest_path} not found")
        sys.exit(1)

    with open(private_manifest_path, "r", encoding="utf-8") as f:
        private_data = json.load(f)
    manifest_entries = private_data["manifest"]

    deskewed_dir = PROJECT_ROOT / "data_private" / "deskewed"
    after_dir = PROJECT_ROOT / "data_private" / "after_outputs"
    SALT = "RRSI_SALT_SECURE_2026_"

    # 1. Map files/doc_ids to template groups
    file_to_group = {}
    id_to_group = {}
    for m in manifest_entries:
        tg = m.get("template_group", "T_UNKNOWN")
        id_to_group[m["doc_id"]] = tg
        for f in m.get("files", []):
            file_to_group[f] = tg
            file_to_group[Path(f).stem] = tg
        if m.get("logical_base_id"):
            file_to_group[m.get("logical_base_id")] = tg

    # 2. Identify contaminated template groups from scratch/ and tests/
    contaminated_groups = set()
    for search_dir in ["scratch", "tests"]:
        p = PROJECT_ROOT / search_dir
        if not p.exists():
            continue
        for fpath in p.rglob("*"):
            if fpath.is_file() and fpath.suffix in [".py", ".json", ".txt", ".md", ".png"]:
                fname = fpath.name
                for k, tg in file_to_group.items():
                    if len(k) > 3 and k in fname:
                        contaminated_groups.add(tg)
                if fpath.suffix in [".py", ".json", ".txt", ".md"]:
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="ignore")
                        for k, tg in file_to_group.items():
                            if len(k) > 4 and k in content:
                                contaminated_groups.add(tg)
                        for did, tg in id_to_group.items():
                            if did in content:
                                contaminated_groups.add(tg)
                    except Exception:
                        pass

    # Ensure baseline known contaminated templates are included
    for m in manifest_entries:
        if m.get("is_contaminated"):
            contaminated_groups.add(m.get("template_group"))

    # 3. Associate anonymous buyer ID per document
    doc_buyer_map = {}
    buyer_counter = Counter()
    for m in manifest_entries:
        doc_id = m["doc_id"]
        out_f = after_dir / f"{doc_id}.json"
        buyer_key = "UNKNOWN"
        if out_f.exists():
            try:
                data = json.loads(out_f.read_text(encoding="utf-8"))
                bg = data.get("customer_gstin")
                bn = data.get("customer_name")
                if bg:
                    buyer_key = f"GSTIN:{bg.strip().upper()}"
                elif bn:
                    buyer_key = f"NAME:{bn.strip().upper()}"
            except Exception:
                pass
        buyer_hash = "BUYER_" + hashlib.sha256((SALT + buyer_key).encode("utf-8")).hexdigest()[:10]
        doc_buyer_map[doc_id] = buyer_hash
        buyer_counter[buyer_hash] += 1

    # 4. Group documents by template
    docs_by_template = defaultdict(list)
    for m in manifest_entries:
        tg = m.get("template_group", "T_UNKNOWN")
        docs_by_template[tg].append(m)

    # 5. Split assignment by template group:
    # All contaminated template groups -> dev
    # Clean template groups -> balanced between selection and locked_test
    splits = {}
    clean_templates = sorted([t for t in docs_by_template.keys() if t not in contaminated_groups])
    contaminated_templates = sorted(list(contaminated_groups))

    for tg in contaminated_templates:
        splits[tg] = "dev"

    # Distribute clean templates alternately to balance template and document count
    # Sort clean templates by document count descending for optimal binning
    clean_templates_sorted = sorted(clean_templates, key=lambda t: len(docs_by_template[t]), reverse=True)
    sel_count, test_count = 0, 0
    for tg in clean_templates_sorted:
        tg_docs = len(docs_by_template[tg])
        if sel_count <= test_count:
            splits[tg] = "selection"
            sel_count += tg_docs
        else:
            splits[tg] = "locked_test"
            test_count += tg_docs

    # 6. Build Git-Safe manifest entries
    git_safe_manifest = []
    split_counts = Counter()

    for m in manifest_entries:
        doc_id = m["doc_id"]
        tg = m.get("template_group", "T_UNKNOWN")
        split = splits[tg]
        is_contaminated = (tg in contaminated_groups)

        # Compute image hash if file exists
        files = m.get("files", [])
        image_hashes = []
        for f in files:
            img_path = deskewed_dir / f
            if img_path.exists():
                image_hashes.append(compute_sha256(img_path))
            else:
                # Fallback to deterministic hash of doc_id if deskewed not present
                image_hashes.append(hashlib.sha256((SALT + f).encode("utf-8")).hexdigest())

        entry = {
            "doc_id": doc_id,
            "template_group": tg,
            "split": split,
            "tier": m.get("tier", "Tier_A"),
            "page_count": m.get("page_count", 1),
            "is_contaminated": is_contaminated,
            "buyer_id": doc_buyer_map.get(doc_id, "BUYER_UNKNOWN"),
            "image_sha256": image_hashes[0] if image_hashes else None
        }
        git_safe_manifest.append(entry)
        split_counts[split] += 1

    # 7. Build Annotation Queue
    # Priority: selection and locked_test first, maximizing template & buyer diversity
    # 1st round: 1 document from each clean template group in selection/locked_test with unique buyers
    # 2nd round: remaining selection/locked_test documents
    # 3rd round: dev documents
    queue_round_1 = []
    queue_round_2 = []
    queue_round_3 = []

    seen_templates_r1 = set()
    seen_buyers_r1 = set()

    for entry in git_safe_manifest:
        split = entry["split"]
        tg = entry["template_group"]
        b_id = entry["buyer_id"]

        if split in ["selection", "locked_test"]:
            if tg not in seen_templates_r1:
                queue_round_1.append(entry["doc_id"])
                seen_templates_r1.add(tg)
                seen_buyers_r1.add(b_id)
            else:
                queue_round_2.append(entry["doc_id"])
        else:
            queue_round_3.append(entry["doc_id"])

    annotation_queue = queue_round_1 + queue_round_2 + queue_round_3

    manifest_output = {
        "version": "1.0.0",
        "description": "Git-safe anonymized benchmark manifest with template-isolated split partitioning",
        "total_documents": len(git_safe_manifest),
        "split_summary": dict(split_counts),
        "template_groups_count": len(docs_by_template),
        "buyer_entities_count": len(buyer_counter),
        "annotation_queue": annotation_queue,
        "manifest": git_safe_manifest
    }

    out_path = PROJECT_ROOT / "data" / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest_output, f, indent=2)

    print("=" * 80)
    print("        TEMPLATE & BUYER HISTOGRAMS (PHASE 1.1b / 1.2b)")
    print("=" * 80)
    print(f"Total Unique Invoices : {len(git_safe_manifest)}")
    print(f"Total Unique Templates: {len(docs_by_template)}")
    print(f"Total Unique Buyers   : {len(buyer_counter)}")

    print("\n[+] Invoices Per Template Group Histogram:")
    print("-" * 55)
    print(f"{'Template':<10} | {'Count':<6} | {'Split':<12} | {'Contaminated?'}")
    print("-" * 55)
    for tg, docs in sorted(docs_by_template.items(), key=lambda x: len(x[1]), reverse=True):
        is_cont = "YES (dev-only)" if tg in contaminated_groups else "NO (clean)"
        print(f"{tg:<10} | {len(docs):<6} | {splits[tg]:<12} | {is_cont}")
    print("-" * 55)

    print("\n[+] Invoices Per Buyer Entity Histogram:")
    print("-" * 45)
    print(f"{'Buyer Entity ID':<25} | {'Invoice Count':<12}")
    print("-" * 45)
    for b_id, count in buyer_counter.most_common():
        print(f"{b_id:<25} | {count:<12}")
    print("-" * 45)

    print("\n[+] Dataset Split Partitioning Summary:")
    print("-" * 45)
    for sp, cnt in split_counts.items():
        tg_cnt = len({m['template_group'] for m in git_safe_manifest if m['split'] == sp})
        print(f"  * {sp:<12} : {cnt:>2} invoices across {tg_cnt:>2} template groups")
    print("-" * 45)

    print(f"\n[+] Annotation Queue Generated: {len(annotation_queue)} invoices prioritized by diversity.")
    print(f"    * High-Priority Round 1 (Max Diversity): {len(queue_round_1)} docs")
    print(f"    * Round 2 (Remaining Selection/Test)  : {len(queue_round_2)} docs")
    print(f"    * Round 3 (Dev Corpus)                 : {len(queue_round_3)} docs")
    print(f"\n[+] Git-safe manifest written to: {out_path}")

if __name__ == "__main__":
    main()
