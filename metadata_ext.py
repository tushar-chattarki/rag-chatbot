def metadata_extraction(PDF_PATH, OUTPUT_JSON):
    import fitz 
    import json
    import os
    from pathlib import Path

    doc = fitz.open(PDF_PATH)

    def rect_to_dict(rect):
        return {
            "x0": rect.x0,
            "y0": rect.y0,
            "x1": rect.x1,
            "y1": rect.y1,
        }


    def make_json_safe(obj):
        """
        Recursively convert PyMuPDF / non-JSON objects
        into JSON-serializable primitives.
        """

        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]

        elif isinstance(obj, tuple):
            return [make_json_safe(v) for v in obj]

        elif isinstance(obj, fitz.Rect):
            return {
                "x0": obj.x0,
                "y0": obj.y0,
                "x1": obj.x1,
                "y1": obj.y1,
            }

        elif isinstance(obj, bytes):
            return {
                "type": "bytes",
                "length": len(obj)
            }

        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        else:
            # Fallback: stringify unknown objects
            return str(obj)


    def extract_reading_order(page, doc, page_number, seq_start):
        blocks = page.get_text("dict")["blocks"]
        body_font_size = estimate_body_font_size(blocks)

        elements = []

        for b in blocks:
            bbox = b["bbox"]

            if b["type"] == 0: 
                text = " ".join(
                    span["text"]
                    for line in b["lines"]
                    for span in line["spans"]
                ).strip()

                if text:
                    block_type = classify_text_block(b, body_font_size)

                    elements.append({
                        "type": block_type,
                        "text": text,
                        "bbox": bbox,
                        "page_number": page_number,
                    })  


            elif b["type"] == 1:
                elements.append({
                    "type": "image",
                    "xref": b["number"],
                    "bbox": bbox,
                    "page_number": page_number,
                })

        elements.sort(key=lambda e: (-e["bbox"][1], e["bbox"][0]), reverse=True)

        seq_id = seq_start
        last_paragraph_id = None

        for el in elements:
            el["sequence_id"] = seq_id

            if el["type"] == "paragraph":
                last_paragraph_id = seq_id
            elif el["type"] == "image":
                el["after_sequence_id"] = last_paragraph_id

            seq_id += 1

        return elements, seq_id


    def classify_text_block(block, body_font_size):
        """
        Decide whether a text block is a header or paragraph.
        """
        spans = [
            span
            for line in block["lines"]
            for span in line["spans"]
        ]

        if not spans:
            return "paragraph"

        font_sizes = [s["size"] for s in spans]
        avg_size = sum(font_sizes) / len(font_sizes)

        text = " ".join(s["text"] for s in spans).strip()

        is_bold = any("Bold" in s["font"] for s in spans)
        is_short = len(text) < 80
        is_large = avg_size > body_font_size * 1.2

        if is_large and (is_bold or is_short):
            return "header"

        return "paragraph"


    def estimate_body_font_size(blocks):
        sizes = []

        for b in blocks:
            if b["type"] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        sizes.append(span["size"])

        # Most common font size = body text
        return max(set(sizes), key=sizes.count)


    def extract_images_from_pdf(pdf_path, output_folder="Image_Metadata/extracted_images"):
        """Extract all images from PDF"""
        os.makedirs(output_folder, exist_ok=True)
        
        pdf_id = Path(pdf_path).stem.lower().replace(" ", "_")
        doc = fitz.open(pdf_path)
        image_info = []
        
        print(f"Scanning {len(doc)} pages for images...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image_path = f"{output_folder}/{pdf_id}_p{page_num+1}_img_{img_index+1}.png"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                image_info.append({
                    "path": image_path,
                    "page": page_num + 1,
                    "index": img_index + 1
                })
        
        doc.close()
        return image_info

    extract_images_from_pdf(PDF_PATH)


    def image_metadata(pdf_path, output_path="Image_Metadata/image_metadata.json"):
        pdf_path = Path(pdf_path)
        pdf_id = pdf_path.stem.lower().replace(" ", "_")

        doc = fitz.open(pdf_path)
        image_info = []

        print(f"Scanning {len(doc)} pages for images...")

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                image_id = f"{pdf_id}_p{page_num+1}_img_{img_index+1}"
                image_filename = f"{image_id}.png"

                image_info.append({
                    "image_id": image_id,
                    "image_path": f"extracted_images/{image_filename}",
                    "source_pdf": pdf_path.name,
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "summary": None  # to be filled later
                })

        doc.close()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Overwrite safely
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(image_info, f, indent=4, ensure_ascii=False)

        return image_info
    
    image_metadata(PDF_PATH)


    document_metadata = {
        "file_name": Path(PDF_PATH).name,
        "file_path": str(Path(PDF_PATH).resolve()),
        "page_count": doc.page_count,
        "is_encrypted": doc.is_encrypted,
        "metadata": doc.metadata, 
    }

    pages_metadata = []
    global_sequence_id = 0


    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)

        page_data = {
            "page_number": page_index + 1,
            "rotation": page.rotation,
            "width": float(page.rect.width),
            "height": float(page.rect.height),

            # Plain text only (safe)
            "text": page.get_text("text"),

            # Images metadata only (NO bytes)
            "images": [],

            # Links (already dicts)
            "links": page.get_links(),

            # Annotations
            "annotations": [],
        }

        reading_order, global_sequence_id = extract_reading_order(
            page=page,
            doc=doc,
            page_number=page_index + 1,
            seq_start=global_sequence_id
        )

        page_data["reading_order"] = reading_order
        


        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)

            page_data["images"].append({
                "xref": xref,
                "width": base_image["width"],
                "height": base_image["height"],
                "color_space": base_image["colorspace"],
                "image_format": base_image["ext"],
                "size_bytes": len(base_image["image"]),
            })

        for annot in page.annots() or []:
            page_data["annotations"].append({
                "type": annot.type[1],
                "rect": rect_to_dict(annot.rect),
                "content": annot.info.get("content"),
            })

        pages_metadata.append(page_data)


    output = {
        "document": document_metadata,
        "pages": pages_metadata,
    }

    safe_output = make_json_safe(output)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(safe_output, f, indent=2, ensure_ascii=False)


    print(f"PDF metadata saved to {OUTPUT_JSON}")
