# scripts/statistical_power.py
"""
Statistical Power Analysis for RRSI Selection & Audit (Phase 1.7).

Calculates Minimum Detectable Effect (MDE) under:
- Paired McNemar discordant-pair formulation
- Paired Cluster Bootstrap accounting for template clustering (Design Effect)
Evaluates whether current sample size N has sufficient power (80% power at alpha=0.05)
to detect realistic pipeline gains (typically 1.5% - 3.0%).
"""

import math
import argparse
from typing import Dict, Any, Optional


def compute_statistical_power(
    sample_count: int,
    template_count: int,
    discordant_rate: float = 0.10,
    icc: float = 0.15,
    alpha: float = 0.05,
    power: float = 0.80
) -> Dict[str, Any]:
    """
    Computes MDE for paired evaluation with template clustering.
    
    Args:
        sample_count: Total number of documents (N)
        template_count: Number of unique template groups (clusters)
        discordant_rate: Fraction of documents with discordant outcomes (cand != base)
        icc: Intra-cluster correlation coefficient
        alpha: Type I error rate (two-tailed)
        power: Desired statistical power (1 - Type II error)
    """
    if sample_count <= 0:
        return {
            "sample_count": 0,
            "mde_pct": 100.0,
            "is_underpowered": True,
            "assessment": "Insufficient sample count (N=0)."
        }

    # Standard normal critical values (alpha=0.05 -> 1.96, power=0.80 -> 0.8416)
    z_alpha = 1.96
    z_beta = 0.8416

    # Unclustered MDE (McNemar paired)
    se_paired = math.sqrt(max(0.001, discordant_rate) / sample_count)
    mde_unclustered = (z_alpha + z_beta) * se_paired

    # Cluster Design Effect (DEFF)
    avg_cluster_size = sample_count / max(1, template_count)
    deff = 1.0 + (max(1.0, avg_cluster_size) - 1.0) * icc
    deff_multiplier = math.sqrt(max(1.0, deff))

    mde_clustered = mde_unclustered * deff_multiplier
    mde_pct = round(mde_clustered * 100.0, 2)

    # Realistic gain threshold: realistic pipeline gains are 1.5% - 3.0%
    is_underpowered = (mde_pct > 3.0)

    if mde_pct > 5.0:
        assessment = (
            f"SEVERELY UNDERPOWERED: Minimum detectable improvement is {mde_pct}%, "
            f"which is significantly larger than realistic incremental pipeline gains (1.5% - 3.0%). "
            f"To detect a 2.0% gain at 80% power, annotate at least N >= {int(math.ceil((z_alpha+z_beta)**2 * discordant_rate * deff / (0.02**2)))} documents."
        )
    elif mde_pct > 3.0:
        assessment = (
            f"MARGINALLY UNDERPOWERED: MDE is {mde_pct}%. Only large architectural improvements will be statistically detectable."
        )
    else:
        assessment = f"ADEQUATELY POWERED: MDE is {mde_pct}%, capable of detecting fine-grained gains >= {mde_pct}%."

    return {
        "sample_count": sample_count,
        "template_count": template_count,
        "avg_cluster_size": round(avg_cluster_size, 2),
        "discordant_rate": discordant_rate,
        "design_effect": round(deff, 3),
        "mde_pct": mde_pct,
        "is_underpowered": is_underpowered,
        "assessment": assessment
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate statistical power and MDE for invoice benchmark.")
    parser.add_argument("--samples", type=int, default=12, help="Selection sample count")
    parser.add_argument("--templates", type=int, default=3, help="Number of template groups")
    parser.add_argument("--discordant-rate", type=float, default=0.10, help="Discordant pair rate")
    args = parser.parse_args()

    res = compute_statistical_power(
        sample_count=args.samples,
        template_count=args.templates,
        discordant_rate=args.discordant_rate
    )

    print("=" * 65)
    print("        RRSI STATISTICAL POWER & SENSITIVITY REPORT")
    print("=" * 65)
    print(f"Sample Count (N)         : {res['sample_count']}")
    print(f"Template Clusters        : {res['template_count']}")
    print(f"Average Cluster Size     : {res['avg_cluster_size']}")
    print(f"Cluster Design Effect    : {res['design_effect']}")
    print(f"Discordant Pair Rate     : {res['discordant_rate'] * 100:.1f}%")
    print(f"Minimum Detectable Effect: {res['mde_pct']}% (at 80% power, alpha=0.05)")
    print(f"\nAssessment:\n  {res['assessment']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
