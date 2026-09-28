# app/ocr/vision_fallback.py
"""
Multimodal Vision-AI Hybrid Fallback Engine (Gemini Flash).
Provides zero-shot document intelligence and layout comprehension
when local OCR encounters missing critical fields or mathematical discrepancies.
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Document AI and financial invoice extraction system.
Analyze the provided invoice image and extract all structured data into the EXACT JSON format specified below.

CRITICAL RULES:
1. Accuracy & Completeness: Never omit an available field. If a field exists in the document, extract it. If absent, set it to null.
2. Party Separation:
   - "seller_name" and "seller_gstin": The vendor/distributor issuing the invoice (e.g. from top/header).
   - "customer_name" and "customer_gstin": The buyer/purchaser/billed-to party.
3. Due Date: If an explicit due date (e.g. "Due Dt.") is shown, extract it as DD/MM/YYYY. If not shown but credit terms (e.g. "Credit: 30 Days") are present, calculate invoice_date + days.
4. Total Payable Amount: "invoice_amount" MUST be the final net payable grand total (e.g., "TO PAY", "Net Payable", "Grand Total"). Do not confuse it with taxable subtotal.
5. Line Items: Extract EVERY line item in the table. Never merge separate items.
   - For compound quantities like "20+ 2" or "50+5", split into "quantity" (50) and "free_quantity" (5), set "total_quantity" (55), and "free_item_yn": "1".
   - Extract MRP, Rate ("unit_price"), Amount ("total_price"), Batch, Expiry ("expiry_date" in DD/MM/YYYY), HSN ("hsn_sac"), Item Code ("item_code"), and Discount.
   - Ensure "total_quantity" at header level equals the sum of all item total quantities.

OUTPUT JSON SCHEMA:
{
  "invoice_id": string or null,
  "invoice_number": string or null,
  "invoice_date": string (DD/MM/YYYY) or null,
  "due_date": string (DD/MM/YYYY) or null,
  "customer_name": string or null,
  "customer_gstin": string or null,
  "seller_name": string or null,
  "seller_gstin": string or null,
  "currency_code": "INR",
  "PO_number": string or null,
  "DC_date": string or null,
  "DC_number": string or null,
  "invoice_amount": string ("XX.XX"),
  "round_off": string ("XX.XX") or null,
  "total_gst_rate": number or null,
  "total_quantity": integer,
  "total_cgst_rate": number or null,
  "total_cgst_amount": string ("XX.XX") or null,
  "total_sgst_rate": number or null,
  "total_sgst_amount": string ("XX.XX") or null,
  "total_igst_rate": number or null,
  "total_igst_amount": string ("XX.XX") or null,
  "total_gst_amount": string ("XX.XX") or null,
  "items": [
    {
      "description": string,
      "Pack": string or null,
      "Batch": string or null,
      "quantity": integer,
      "free_quantity": integer,
      "total_quantity": integer,
      "free_item_yn": "0" or "1",
      "unit_price": string ("XX.XX"),
      "total_price": string ("XX.XX"),
      "reference_number": string or null,
      "hsn_sac": string or null,
      "item_code": string or null,
      "expiry_date": string (DD/MM/YYYY) or null,
      "Discount": string ("XX.XX") or null,
      "Discount_type": string or null,
      "Value": string or null,
      "Gst%": number or null,
      "MRP": string ("XX.XX") or null,
      "cgst_rate": number or null,
      "cgst_amount": string ("XX.XX") or null,
      "sgst_rate": number or null,
      "sgst_amount": string ("XX.XX") or null,
      "igst_rate": number or null,
      "igst_amount": string ("XX.XX") or null,
      "GST_AMT": string ("XX.XX") or null,
      "taxable_value": string ("XX.XX") or null
    }
  ]
}
"""

class GeminiVisionFallback:
    """
    Multimodal Document AI fallback using Gemini Flash.
    Executes zero-shot invoice comprehension directly on page images.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai client: {e}")

    def is_available(self) -> bool:
        """Returns True if Gemini client is successfully configured with an API key."""
        return self._client is not None

    def extract_from_image(self, image_input: Union[str, Path, Image.Image]) -> Optional[Dict[str, Any]]:
        """
        Runs multimodal extraction on an invoice image.
        Returns validated official JSON dictionary or None if unavailable/failed.
        """
        if not self.is_available():
            logger.info("Gemini Vision Fallback unavailable: GEMINI_API_KEY not configured.")
            return None

        try:
            pil_img = image_input
            if isinstance(image_input, (str, Path)):
                pil_img = Image.open(str(image_input)).convert("RGB")
            elif not isinstance(pil_img, Image.Image):
                return None

            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[
                    pil_img,
                    SYSTEM_PROMPT
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )

            raw_text = response.text
            if not raw_text:
                return None

            data = json.loads(raw_text)
            return data

        except Exception as e:
            logger.error(f"Gemini Vision extraction failed: {e}")
            return None

    def patch_missing_fields(
        self,
        local_output: Dict[str, Any],
        image_input: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """
        Hybrid fusion: if local output has missing/null critical fields,
        queries Gemini Vision to patch those specific missing fields.
        """
        critical_fields = ["due_date", "customer_gstin", "seller_gstin", "invoice_amount", "customer_name"]
        missing = [f for f in critical_fields if not local_output.get(f)]
        items = local_output.get("items", [])

        # Trigger if critical fields are missing or if 0 items were extracted
        if not missing and len(items) > 0:
            return local_output

        if not self.is_available():
            return local_output

        gemini_result = self.extract_from_image(image_input)
        if not gemini_result:
            return local_output

        patched = dict(local_output)

        # Patch missing header fields
        for field in critical_fields:
            if not patched.get(field) and gemini_result.get(field):
                patched[field] = gemini_result[field]
                logger.info(f"Patched '{field}' from Gemini Vision: {patched[field]}")

        # If local dropped items or extracted 0 items, use Gemini items
        if len(patched.get("items", [])) < len(gemini_result.get("items", [])):
            patched["items"] = gemini_result.get("items", [])
            patched["total_quantity"] = gemini_result.get("total_quantity", patched.get("total_quantity"))
            logger.info("Repaired line items from Gemini Vision")

        return patched
