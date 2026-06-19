import os
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import torch

import google.generativeai as genai
from transformers import AutoProcessor, AutoModelForVision2Seq


def is_low_variance(image, threshold=10):
    return np.array(image.convert("RGB")).std() < threshold


def is_flat_histogram(image, min_nonzero_bins=20):
    hist = image.convert("L").histogram()
    return sum(h > 0 for h in hist) < min_nonzero_bins


def is_bad_geometry(image):
    w, h = image.size
    ratio = w / h if h else 0
    return w < 100 or h < 100 or ratio > 10 or ratio < 0.1


def is_useful_image(image):
    return not (
        is_bad_geometry(image)
        or is_low_variance(image)
        or is_flat_histogram(image)
    )


def load_and_resize(path, max_size=384):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


def summarize_images_and_update_metadata(
    image_dir: str,
    image_metadata_file: str,
    output_summaries_file: str,
    use_gemini: bool = False,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    sleep_time: float = 0.3,
):
    """
    Summarizes images and updates image metadata with summaries.

    Returns:
        dict with stats
    """

    image_dir = Path(image_dir)

    if use_gemini:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        vision_model = genai.GenerativeModel("gemini-2.5-flash")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    if os.path.exists(output_summaries_file):
        with open(output_summaries_file, "r", encoding="utf-8") as f:
            summaries = json.load(f)
    else:
        summaries = {}

    generated = 0
    skipped = 0

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        image_id = img_path.stem

        if image_id in summaries and summaries[image_id].get("summary"):
            skipped += 1
            continue

        image = Image.open(img_path)

        if not is_useful_image(image):
            skipped += 1
            continue

        print(f"Summarizing: {img_path.name}")

        if use_gemini:
            response = vision_model.generate_content([
                "Describe this technical diagram concisely.", image
            ])
            summary = response.text
        else:
            resized = load_and_resize(img_path)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": resized},
                    {"type": "text", "text": (
                        "Describe this technical diagram:\n"
                        "- Type of visualization\n"
                        "- Key components and relationships\n"
                        "- Concise and technical"
                    )}
                ]
            }]

            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[prompt],
                images=[resized],
                return_tensors="pt"
            ).to(model.device)

            input_len = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, max_new_tokens=300)

            summary = processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

        summaries[image_id] = {
            "image_path": str(img_path),
            "summary": summary,
        }

        generated += 1
        time.sleep(sleep_time)

    with open(output_summaries_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    with open(image_metadata_file, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)

    updated = 0
    for item in image_metadata:
        image_id = item["image_id"]
        if image_id in summaries:
            item["summary"] = summaries[image_id]["summary"]
            updated += 1

    with open(image_metadata_file, "w", encoding="utf-8") as f:
        json.dump(image_metadata, f, indent=2, ensure_ascii=False)

    return {
        "generated": generated,
        "skipped": skipped,
        "metadata_updated": updated,
        "total_images": len(image_metadata),
    }
