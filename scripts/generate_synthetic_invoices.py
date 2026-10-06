# scripts/generate_synthetic_invoices.py
"""
CLI script to generate synthetic test invoices for development, unit testing,
and ablation experiments (never for selection or test).
"""

import argparse
from pathlib import Path
from app.rrsi.synthetic_generator import SyntheticInvoiceGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic invoices with ground truth.")
    parser.add_argument("--count", type=int, default=10, help="Number of invoices to generate")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--noise", action="store_true", help="Add visual noise (rotation, blur)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} synthetic invoices (seed={args.seed}) into {out_dir}...")
    generator = SyntheticInvoiceGenerator(seed=args.seed)

    for i in range(1, args.count + 1):
        _, data = generator.generate_sample(invoice_idx=i, output_dir=out_dir, add_noise=args.noise)
        if i % 5 == 0 or i == args.count:
            print(f"  [{i}/{args.count}] Generated {data['invoice_number']}")

    print(f"Finished generating {args.count} invoices.")


if __name__ == "__main__":
    main()
