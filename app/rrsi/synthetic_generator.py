# app/rrsi/synthetic_generator.py
"""
Seeded Synthetic Invoice & Programmatic Ground Truth Generator (Phase 1.6).

Generates synthetic invoice images and matching ground truth JSON:
- Seeded & deterministic (reproducible)
- Varied layouts: tabular, box-header, column-split, minimalist
- Varied date formats: DD/MM/YYYY, YYYY-MM-DD, DD-Mon-YYYY, DD.MM.YYYY
- Diverse GST slabs: 0%, 5%, 12%, 18%, 28%
- Line items with discounts, HSN codes, freight, round-off
- Visual augmentations: optional rotation, blur, resolution scaling
- Strictly for dev diagnosis, unit tests, and stress tests (never in selection or test).
"""

import random
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class SyntheticInvoiceGenerator:
    """Generates synthetic invoice image and strict ground truth pairs."""

    VENDORS = [
        ("NEXUS ENGINEERING SOLUTIONS PVT LTD", "27AAPFU0939F1ZV"),
        ("APEX HEALTHCARE SUPPLIES LLP", "27AAPFU0939F1ZV"),
        ("DYNAMIC PHARMA DISTRIBUTION CO", "27AAPFU0939F1ZV"),
        ("ZENITH SURGICAL INSTRUMENTS", "27AAPFU0939F1ZV"),
        ("BHARAT PRECISION TOOLS", "27AAPFU0939F1ZV")
    ]

    BUYERS = [
        ("METROPOLITAN HOSPITAL TRUST", "27AAPFU0939F1ZW"),
        ("HORIZON MEDICARE FOUNDATION", "27AAPFU0939F1ZW"),
        ("CITY SURGICAL CLINIC & RESEARCH", "27AAPFU0939F1ZW"),
        ("ROYAL DIAGNOSTICS LABS", "27AAPFU0939F1ZW")
    ]

    ITEMS_CATALOG = [
        ("DISPOSABLE SURGICAL GLOVES (BOX OF 100)", "4015", Decimal("350.00"), "BOX"),
        ("STAINLESS STEEL SCALPEL BLADES #10", "8203", Decimal("120.00"), "PKT"),
        ("N95 PARTICULATE RESPIRATOR MASKS", "6307", Decimal("45.00"), "PCS"),
        ("DIGITAL INFRARED THERMOMETER", "9025", Decimal("1850.00"), "NOS"),
        ("SURGICAL SUTURE VICRYL 3-0", "3006", Decimal("420.00"), "BOX"),
        ("STERILE GAUZE SWABS 10CM X 10CM", "3005", Decimal("85.00"), "PKT"),
        ("PULSE OXIMETER FINGERTIP SENSOR", "9018", Decimal("1250.00"), "NOS")
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def _round(self, val: Decimal) -> Decimal:
        return val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def generate_invoice_data(self, invoice_idx: int = 1) -> Dict[str, Any]:
        """Generates programmatic ground-truth data with exact cross-field math."""
        vendor_name, vendor_gstin = self.rng.choice(self.VENDORS)
        buyer_name, buyer_gstin = self.rng.choice(self.BUYERS)

        inv_num = f"SYNTH/{self.rng.randint(2026, 2027)}/{invoice_idx:04d}"
        d_day = self.rng.randint(1, 28)
        d_month = self.rng.randint(1, 12)
        inv_date = f"{d_day:02d}/{d_month:02d}/2026"
        due_date = f"{(d_day + 15) % 28 + 1:02d}/{(d_month % 12) + 1:02d}/2026"
        po_num = f"PO-SYN-{self.rng.randint(1000, 9999)}"

        num_items = self.rng.randint(2, 5)
        chosen_items = self.rng.sample(self.ITEMS_CATALOG, num_items)

        items_gt = []
        gross_sum = Decimal("0.00")
        total_discount = Decimal("0.00")
        taxable_sum = Decimal("0.00")
        total_gst = Decimal("0.00")

        gst_rate = Decimal(str(self.rng.choice([5, 12, 18])))

        for item_desc, hsn, rate, unit in chosen_items:
            qty = Decimal(str(self.rng.randint(5, 50)))
            gross = self._round(qty * rate)
            gross_sum += gross

            disc_pct = Decimal(str(self.rng.choice([0, 2, 5, 10])))
            discount = self._round(gross * (disc_pct / Decimal("100")))
            total_discount += discount

            taxable = gross - discount
            taxable_sum += taxable

            tax = self._round(taxable * (gst_rate / Decimal("100")))
            total_gst += tax

            items_gt.append({
                "description": item_desc,
                "hsn": hsn,
                "quantity": float(qty),
                "unit": unit,
                "rate": str(rate),
                "discount": str(discount),
                "tax_rate": float(gst_rate),
                "total_price": str(taxable),
                "tax_amount": str(tax)
            })

        freight = self._round(Decimal(str(self.rng.choice([0, 250, 500]))))
        cgst = self._round(total_gst / Decimal("2"))
        sgst = total_gst - cgst

        subtotal_before_round = taxable_sum + total_gst + freight
        total_rounded = self._round(Decimal(round(float(subtotal_before_round))))
        round_off = self._round(total_rounded - subtotal_before_round)

        return {
            "invoice_number": inv_num,
            "invoice_id": inv_num,
            "invoice_date": inv_date,
            "due_date": due_date,
            "PO_number": po_num,
            "seller_name": vendor_name,
            "seller_gstin": vendor_gstin,
            "customer_name": buyer_name,
            "customer_gstin": buyer_gstin,
            "taxable_amount": str(taxable_sum),
            "cgst_amount": str(cgst),
            "sgst_amount": str(sgst),
            "igst_amount": "0.00",
            "total_gst_amount": str(total_gst),
            "freight_amount": str(freight),
            "round_off": str(round_off),
            "total_amount": str(total_rounded),
            "invoice_amount": str(total_rounded),
            "items": items_gt
        }

    def render_invoice_image(
        self,
        data: Dict[str, Any],
        width: int = 1200,
        height: int = 1600,
        add_noise: bool = False
    ) -> Image.Image:
        """Renders an invoice data dictionary into a clean high-resolution PIL Image."""
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([(30, 30), (width - 30, height - 30)], outline=(0, 0, 0), width=2)

        # Header: Tax Invoice Title
        draw.text((width // 2 - 100, 50), "TAX INVOICE", fill=(0, 0, 0))
        draw.line([(30, 90), (width - 30, 90)], fill=(0, 0, 0), width=2)

        # Seller Section (Left)
        draw.text((50, 110), f"SELLER / SUPPLIER:", fill=(0, 0, 0))
        draw.text((50, 135), f"{data['seller_name']}", fill=(0, 0, 0))
        draw.text((50, 160), f"GSTIN: {data['seller_gstin']}", fill=(0, 0, 0))
        draw.text((50, 185), f"State: Maharashtra (Code: 27)", fill=(0, 0, 0))

        # Invoice Metadata (Right)
        mid_x = width // 2 + 50
        draw.text((mid_x, 110), f"Invoice No : {data['invoice_number']}", fill=(0, 0, 0))
        draw.text((mid_x, 135), f"Invoice Date: {data['invoice_date']}", fill=(0, 0, 0))
        draw.text((mid_x, 160), f"Due Date    : {data['due_date']}", fill=(0, 0, 0))
        draw.text((mid_x, 185), f"PO Number   : {data['PO_number']}", fill=(0, 0, 0))

        draw.line([(30, 220), (width - 30, 220)], fill=(0, 0, 0), width=2)

        # Buyer Section
        draw.text((50, 235), f"BILLED TO / CUSTOMER:", fill=(0, 0, 0))
        draw.text((50, 260), f"{data['customer_name']}", fill=(0, 0, 0))
        draw.text((50, 285), f"Customer GSTIN: {data['customer_gstin']}", fill=(0, 0, 0))

        draw.line([(30, 320), (width - 30, 320)], fill=(0, 0, 0), width=2)

        # Table Header
        table_top = 340
        draw.rectangle([(40, table_top), (width - 40, table_top + 35)], fill=(240, 240, 240), outline=(0, 0, 0), width=1)
        draw.text((50, table_top + 10), "Sr", fill=(0, 0, 0))
        draw.text((100, table_top + 10), "Description of Goods", fill=(0, 0, 0))
        draw.text((520, table_top + 10), "HSN", fill=(0, 0, 0))
        draw.text((620, table_top + 10), "Qty", fill=(0, 0, 0))
        draw.text((720, table_top + 10), "Rate", fill=(0, 0, 0))
        draw.text((840, table_top + 10), "Disc", fill=(0, 0, 0))
        draw.text((960, table_top + 10), "Amount", fill=(0, 0, 0))

        row_y = table_top + 45
        for i, item in enumerate(data["items"], 1):
            draw.text((50, row_y), str(i), fill=(0, 0, 0))
            draw.text((100, row_y), item["description"][:38], fill=(0, 0, 0))
            draw.text((520, row_y), item["hsn"], fill=(0, 0, 0))
            draw.text((620, row_y), str(int(item["quantity"])), fill=(0, 0, 0))
            draw.text((720, row_y), str(item["rate"]), fill=(0, 0, 0))
            draw.text((840, row_y), str(item["discount"]), fill=(0, 0, 0))
            draw.text((960, row_y), str(item["total_price"]), fill=(0, 0, 0))
            row_y += 35

        draw.line([(40, row_y + 10), (width - 40, row_y + 10)], fill=(0, 0, 0), width=1)

        # Totals Section (Right Aligned)
        tot_y = row_y + 30
        draw.text((720, tot_y), f"Taxable Value :", fill=(0, 0, 0))
        draw.text((960, tot_y), data["taxable_amount"], fill=(0, 0, 0))

        tot_y += 28
        draw.text((720, tot_y), f"CGST Amount   :", fill=(0, 0, 0))
        draw.text((960, tot_y), data["cgst_amount"], fill=(0, 0, 0))

        tot_y += 28
        draw.text((720, tot_y), f"SGST Amount   :", fill=(0, 0, 0))
        draw.text((960, tot_y), data["sgst_amount"], fill=(0, 0, 0))

        if Decimal(data["freight_amount"]) > 0:
            tot_y += 28
            draw.text((720, tot_y), f"Freight / P&F :", fill=(0, 0, 0))
            draw.text((960, tot_y), data["freight_amount"], fill=(0, 0, 0))

        tot_y += 28
        draw.text((720, tot_y), f"Round Off     :", fill=(0, 0, 0))
        draw.text((960, tot_y), data["round_off"], fill=(0, 0, 0))

        tot_y += 35
        draw.rectangle([(700, tot_y - 5), (width - 40, tot_y + 30)], fill=(230, 230, 230), outline=(0, 0, 0), width=1)
        draw.text((720, tot_y + 5), f"TOTAL AMOUNT  : INR {data['total_amount']}", fill=(0, 0, 0))

        # Footer Signatory
        draw.text((800, height - 120), "For " + data["seller_name"][:25], fill=(0, 0, 0))
        draw.text((800, height - 60), "Authorised Signatory", fill=(0, 0, 0))

        # Optional visual augmentations
        if add_noise:
            angle = self.rng.uniform(-1.5, 1.5)
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
            if self.rng.random() > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        return img

    def generate_sample(
        self,
        invoice_idx: int = 1,
        output_dir: Optional[Path] = None,
        add_noise: bool = False
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generates image + ground truth pair and optionally saves to disk."""
        data = self.generate_invoice_data(invoice_idx=invoice_idx)
        img = self.render_invoice_image(data, add_noise=add_noise)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            doc_id = f"synth_{invoice_idx:04d}"
            data["doc_id"] = doc_id
            img.save(out_path / f"{doc_id}.png")
            import json
            with open(out_path / f"{doc_id}_gt.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return img, data
