import json
import fitz
from pathlib import Path


PDF_DIR = Path("/Users/tusharc/Code/Python/RAG/RAG-Model/pdfs")
EXTRACTED_IMG_DIR = "extracted_images"

IMAGE_METADATA_FILE = Path("Image_Metadata/image_metadata.json")
IMAGE_SUMMARIES_FILE = Path("Image_Metadata/image_summaries_1.json")
FINAL_OUTPUT_FILE = Path("Image_Metadata/image_metadata_final.json")

IMAGE_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)


def extract_image_metadata():
    if IMAGE_METADATA_FILE.exists():
        with open(IMAGE_METADATA_FILE, "r", encoding="utf-8") as f:
            image_metadata = json.load(f)
    else:
        image_metadata = []

    existing_ids = {img["image_id"] for img in image_metadata}
    added = 0

    for pdf_path in PDF_DIR.glob("*.pdf"):
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
                    "image_path": f"{EXTRACTED_IMG_DIR}/{image_id}.png",
                    "source_pdf": pdf_path.name,
                    "page": page_num + 1,
                    "index": img_index,
                    "summary": None
                })

                existing_ids.add(image_id)
                added += 1

        doc.close()

    with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(image_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Added {added} new image entries")
    print(f"📦 Total images tracked: {len(image_metadata)}")
    print(f"💾 Saved to: {IMAGE_METADATA_FILE}")

    return image_metadata

def merge_image_summaries(image_metadata):
    if not IMAGE_SUMMARIES_FILE.exists():
        print("⚠️ Image summaries file not found — skipping merge")
        return image_metadata

    with open(IMAGE_SUMMARIES_FILE, "r", encoding="utf-8") as f:
        image_summaries = json.load(f)

    summary_map = {
        image_id: data["summary"]
        for image_id, data in image_summaries.items()
        if data.get("summary")
    }

    updated = 0

    for img in image_metadata:
        image_id = img.get("image_id")
        if image_id in summary_map:
            img["summary"] = summary_map[image_id]
            updated += 1

    print(f"🧠 Attached summaries to {updated} images")

    return image_metadata

def write_final_metadata(image_metadata):
    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(image_metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ Final metadata written to: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    metadata = extract_image_metadata()
    metadata = merge_image_summaries(metadata)
    write_final_metadata(metadata)

    print("\n🎉 IMAGE PIPELINE COMPLETE")
