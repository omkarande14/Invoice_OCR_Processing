import json
from pathlib import Path
from app.ocr.dual_engine_benchmarker import DualEngineBenchmarker

def main():
    images = [
        Path("data/deskewed/196,Pune,Deenanath Mangeshakar (1)_page_1_oriented_deskewed.png"),
        Path("data/deskewed/CC-1415_page_1_oriented_deskewed.png"),
        Path("data/deskewed/170588 (1)_page_1_oriented_deskewed.png")
    ]

    benchmarker = DualEngineBenchmarker(use_gpu=True)
    report_path = Path("data/benchmarks/dual_engine_benchmark.json")
    summary = benchmarker.benchmark_sample_set(images, output_report=report_path)

    print("\n" + "=" * 65)
    print("           DUAL-ENGINE BENCHMARK SUMMARY (3 INVOICES)")
    print("=" * 65)
    print(f"Total Invoices Tested:                   {summary['total_images_tested']}")
    print(f"PaddleOCR (RTX 3050 GPU) Avg Runtime:   {summary['average_paddle_gpu_runtime']}s")
    print(f"Tesseract (CPU) Avg Runtime:            {summary['average_tesseract_cpu_runtime']}s")
    print(f"Average Consensus Agreement:            {summary['average_consensus_agreement_pct']}%")
    print("=" * 65 + "\n")

    for comp in summary["per_image_comparisons"]:
        name = comp["image_name"]
        p = comp["paddle_metrics"]
        t = comp["tesseract_metrics"]
        c = comp["consensus"]
        
        print(f"=== INVOICE: {name} ===")
        print(f"  PaddleOCR (GPU): {p['runtime_seconds']}s | Elements: {p['total_elements']} | Confidence: {p['average_confidence']*100:.1f}%")
        print(f"    GSTIN:      {p['entities']['gstin']}")
        print(f"    Invoice No: {p['entities']['invoice_numbers']}")
        print(f"    Totals:     {p['entities']['totals']}")
        print(f"  Tesseract (CPU): {t['runtime_seconds']}s | Words: {t['total_words']} | Confidence: {t['average_confidence']*100:.1f}%")
        print(f"    GSTIN:      {t['entities']['gstin']}")
        print(f"    Invoice No: {t['entities']['invoice_numbers']}")
        print(f"    Totals:     {t['entities']['totals']}")
        print(f"  >>> Consensus Score: {comp['consensus_score']}%")
        print(f"  >>> Overlap Agreed GSTIN: {c['gstin_agreed']}")
        print(f"  >>> Overlap Agreed Totals: {c['total_amount_agreed']}\n")

if __name__ == "__main__":
    main()
