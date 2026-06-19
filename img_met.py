import json
import fitz
from pathlib import Path

PDF_DIR = "/Users/tusharc/Code/Python/RAG/RAG-Model/pdfs"
OUTPUT_PATH = "/Users/tusharc/Code/Python/RAG/RAG-Model/Image_Metadata/image_metadata.json"
EXT_IMG_DIR = ""

def extract_image_metadata(
    pdf_dir: str,
    output_path: str,
    extracted_img_dir: str = "extracted_images",
):
    """
    Extract image metadata from all PDFs in a directory.
    Safely appends new images without duplicating existing ones.
    """

    pdf_dir = Path(pdf_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metadata (if any)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            image_metadata = json.load(f)
    else:
        image_metadata = []

    existing_ids = {img["image_id"] for img in image_metadata}

    added = 0

    for pdf_path in pdf_dir.glob("*.pdf"):
        pdf_id = pdf_path.stem.lower().replace(" ", "_")
        doc = fitz.open(pdf_path)

        print(f"📄 Scanning {len(doc)} pages in {pdf_path.name}")

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, _ in enumerate(image_list, start=1):
                image_id = f"{pdf_id}_p{page_num+1}_img_{img_index}"

                if image_id in existing_ids:
                    continue

                image_metadata.append({
                    "image_id": image_id,
                    "image_path": f"{extracted_img_dir}/{image_id}.png",
                    "source_pdf": pdf_path.name,
                    "page": page_num + 1,
                    "index": img_index,
                    "summary": None,
                })

                existing_ids.add(image_id)
                added += 1

        doc.close()

    # Save updated metadata
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(image_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Added {added} new image entries")
    print(f"📦 Total images tracked: {len(image_metadata)}")
    print(f"💾 Saved to: {output_path}")

    return {
        "added": added,
        "total": len(image_metadata),
        "output_file": str(output_path),
    }
