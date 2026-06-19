import json
from pathlib import Path

IMAGE_METADATA_FILE = "Image_Metadata/image_metadata.json"
IMAGE_SUMMARIES_FILE = "Image_Metadata/image_summaries_1.json"
OUTPUT_FILE = "Image_Metadata/image_metadata_final.json"

# Load files
with open(IMAGE_METADATA_FILE, "r", encoding="utf-8") as f:
    image_metadata = json.load(f)

with open(IMAGE_SUMMARIES_FILE, "r", encoding="utf-8") as f:
    image_summaries = json.load(f)

# Build lookup: image_id -> summary
summary_map = {
    image_id: data["summary"]
    for image_id, data in image_summaries.items()
    if data.get("summary")
}

updated = 0

# Merge summaries into metadata
for img in image_metadata:
    image_id = img.get("image_id")
    if image_id in summary_map:
        img["summary"] = summary_map[image_id]
        updated += 1

# Save merged metadata
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(image_metadata, f, indent=2, ensure_ascii=False)

print(f"Updated summaries for {updated} images")
print(f"Final metadata saved to: {OUTPUT_FILE}")
