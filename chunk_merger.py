def merge_small_chunks_with_overlap(
    nodes,
    min_chars=800,
    max_chars=2000,
    overlap_chars=250,
):
    """
    Merge small markdown chunks into larger semantic chunks
    and add overlap between consecutive chunks.
    """

    merged = []
    buffer_node = None

    for node in nodes:
        text = node.get_content().strip()
        if not text:
            continue

        if buffer_node is None:
            buffer_node = node
            continue

        combined = buffer_node.get_content() + "\n\n" + text

        if len(combined) < min_chars:
            buffer_node.text = combined
            continue

        merged.append(buffer_node)
        buffer_node = node

    if buffer_node:
        merged.append(buffer_node)

    overlapped_nodes = []

    for i, node in enumerate(merged):
        text = node.get_content()

        if i == 0:
            overlapped_nodes.append(node)
            continue

        prev_text = merged[i - 1].get_content()

        overlap = prev_text[-overlap_chars:]

        node.text = overlap + "\n\n" + text
        overlapped_nodes.append(node)

    return overlapped_nodes
