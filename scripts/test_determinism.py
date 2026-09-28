# scripts/test_determinism.py
import json
import hashlib
import sys
import platform
from pathlib import Path
import paddle

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure GPU DLLs are loaded
from app.ocr.dual_engine_benchmarker import configure_gpu_dll_paths
configure_gpu_dll_paths()

from app.ocr.invoice_pipeline import InvoicePipeline

def get_json_hash(data: dict) -> str:
    s = json.dumps(data, sort_keys=True, indent=2)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def run_determinism_test():
    img_4531 = Path("data/test_uploads/staging/deskewed/invoice_4531 (1)_page_1_oriented_deskewed.png")
    img_kundan = Path("data/test_uploads/staging/deskewed/264882_SB_DEENANATH MEDICAL STORES_20_08_2026_page_1_oriented_deskewed.png")

    pipeline_gpu = InvoicePipeline(use_gpu=True)

    results = {}
    for img_path, name in [(img_4531, "invoice_4531"), (img_kundan, "kundan")]:
        if not img_path.exists():
            print(f"[!] Image not found: {img_path}")
            continue

        print(f"[*] Testing 5 runs for {name} on GPU...")
        runs = []
        hashes = []
        for i in range(5):
            out = pipeline_gpu.process_invoice(str(img_path))
            h = get_json_hash(out)
            runs.append(out)
            hashes.append(h)
            print(f"    Run {i+1}: SHA256={h[:16]}...")

        is_byte_identical = len(set(hashes)) == 1
        results[name] = {
            "is_byte_identical": is_byte_identical,
            "hashes": hashes,
            "distinct_hashes_count": len(set(hashes))
        }

    # Record environment
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "paddle_version": paddle.__version__,
        "paddle_cuda_compiled": paddle.is_compiled_with_cuda(),
        "gpu_device": paddle.device.cuda.get_device_name(0) if paddle.is_compiled_with_cuda() else None,
        "determinism_results": results
    }

    rep_dir = Path("reports")
    rep_dir.mkdir(exist_ok=True, parents=True)
    env_file = rep_dir / "environment.json"
    with open(env_file, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    print(f"\n[+] Environment and determinism report saved to {env_file}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_determinism_test()
