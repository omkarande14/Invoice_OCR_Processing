import json
from pathlib import Path
from app.ocr.invoice_pipeline import InvoicePipeline

def main():
    pipeline = InvoicePipeline(use_gpu=True)
    input_folder = Path("data/deskewed")
    output_folder = Path("data/ocr_results")
    output_folder.mkdir(parents=True, exist_ok=True)

    test_files = [
        Path("data/test_uploads/staging/deskewed/invoice_4531 (1)_page_1_oriented_deskewed.png"),
        input_folder / "196,Pune,Deenanath Mangeshakar (1)_page_1_oriented_deskewed.png",
        input_folder / "CC-1415_page_1_oriented_deskewed.png",
        input_folder / "170588 (1)_page_1_oriented_deskewed.png"
    ]

    print("\n" + "=" * 75)
    print("        RUNNING COMPLETE INVOICE OCR & VERIFICATION PIPELINE")
    print("=" * 75)

    for img in test_files:
        if not img.exists():
            continue
        print(f"\nProcessing: {img.name} ...")
        official_json, audit_json = pipeline.process_invoice(img, return_diagnostics=True)

        # 1. Save individual official JSON result (strictly official schema)
        out_file = output_folder / f"{img.stem}_invoice.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(official_json, f, indent=2)

        # 2. Save individual sidecar audit JSON result
        audit_file = output_folder / f"{img.stem}_audit.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(audit_json, f, indent=2)

        ver = audit_json.get("verification", {})
        perf = audit_json.get("pipeline_performance", {})

        print(f"  --> Invoice ID / No:        {official_json['invoice_id']} / {official_json['invoice_number']}")
        print(f"  --> Invoice Date / Due Date:{official_json['invoice_date']} / {official_json['due_date']}")
        print(f"  --> PO Number:              {official_json['PO_number']}")
        print(f"  --> Seller Name:            {official_json['seller_name']}")
        print(f"  --> Seller GSTIN:           {official_json['seller_gstin']}")
        print(f"  --> Customer Name:          {official_json['customer_name']}")
        print(f"  --> Customer GSTIN:         {official_json['customer_gstin']}")
        print(f"  --> Line Items Count:       {len(official_json['items'])}")
        print(f"  --> Total Quantity:         {official_json['total_quantity']}")
        print(f"  --> Total GST Amount:       Rs. {official_json['total_gst_amount']}")
        print(f"  --> Invoice Amount:         Rs. {official_json['invoice_amount']}")
        print(f"  --> Arithmetic Status:      {ver.get('arithmetic_status')}")
        print(f"  --> Dual-Engine Consensus:  {ver.get('dual_engine_consensus')} ({ver.get('consensus_score_pct')}%)")
        print(f"  --> Pipeline Runtime:       {perf.get('total_runtime_seconds')}s (Paddle GPU: {perf.get('paddle_gpu_runtime')}s)")
        print(f"  --> Saved Official Output:  {out_file}")
        print(f"  --> Saved Sidecar Audit:    {audit_file}")

    print("\n" + "=" * 75)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    main()
