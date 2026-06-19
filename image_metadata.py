# image_metadata.py
from pathlib import Path

def extract_image_metadata(layout_json, pdf_path):
    pdf_name = Path(pdf_path).name
    pdf_id = Path(pdf_path).stem

    images = []

    for page in layout_json.get("pages", []):
        page_num = page["page_number"]

        for idx, img in enumerate(page.get("images", []), start=1):
            images.append({
                "image_id": f"{pdf_id}_p{page_num}_img_{idx}",
                "image_path": f"extracted_images/{pdf_id}_p{page_num}_img_{idx}.png",
                "source_pdf": pdf_name,
                "page": page_num,
                "index": idx,
                "bbox": img.get("bbox"),
                "xref": img.get("xref"),
            })

    return images
