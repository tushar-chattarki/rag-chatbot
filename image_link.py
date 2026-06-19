from collections import defaultdict
from pathlib import Path


def build_image_index(image_metadata):
    """
    Builds a lookup:
    (source_pdf, page) -> [image_dict, ...]
    """
    index = defaultdict(list)

    for img in image_metadata:
        try:
            key = (img["source_pdf"], img["page"])
            index[key].append(img)
        except KeyError:
            # skip malformed entries safely
            continue

    return index


def attach_images_to_nodes(text_nodes, image_index):
    """
    Attach image references to nodes via metadata only.
    Does NOT inject text.
    """
    attached = 0

    for node in text_nodes:
        source_pdf = Path(node.metadata.get("file_name", "")).name
        page = node.metadata.get("page")

        if not source_pdf or not page:
            continue

        key = (source_pdf, page)

        if key in image_index:
            node.metadata["images"] = image_index[key]
            attached += len(image_index[key])

    return attached


def enrich_nodes_with_image_summaries(text_nodes):
    """
    Inject image summaries into node text content.
    Safe if summaries are missing.
    """
    enriched_nodes = []

    for node in text_nodes:
        images = node.metadata.get("images", [])

        if not images:
            enriched_nodes.append(node)
            continue

        image_block = "\n\n[IMAGE CONTEXT]\n"

        for img in images:
            summary = img.get("summary")
            if summary:
                image_block += f"- {summary}\n"

        if image_block.strip() != "[IMAGE CONTEXT]":
            node.text = node.get_content() + image_block

        enriched_nodes.append(node)

    return enriched_nodes
