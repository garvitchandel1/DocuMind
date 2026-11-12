
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """Create overlapping chunks without splitting words."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    # Combine page text while preserving its source locations.
    parts = []
    page_ranges = []
    position = 0

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        if parts:
            parts.append("\n\n")
            position += 2

        start = position
        parts.append(text)
        position += len(text)

        page_ranges.append({
            "start": start,
            "end": position,
            "metadata": page["metadata"],
        })

    full_text = "".join(parts)
    chunks = []
    start = 0

    while start < len(full_text):
        target_end = min(start + chunk_size, len(full_text))
        end = target_end

        if target_end < len(full_text):
            # Extend to the next whitespace boundary.
            boundary = full_text.find(" ", target_end)

            if boundary == -1:
                boundary = full_text.find("\n", target_end)

            if boundary != -1:
                end = boundary

        chunk_text = full_text[start:end].strip()

        if chunk_text:
            overlapping_pages = [
                page for page in page_ranges
                if page["start"] < end and page["end"] > start
            ]

            first_metadata = overlapping_pages[0]["metadata"]

            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "metadata": {
                    "source": first_metadata["source"],
                    "title": first_metadata.get("title"),
                    "pages": [
                        page["metadata"]["page"]
                        for page in overlapping_pages
                    ],
                },
            })

        if end == len(full_text):
            break

        # Start near the end of this chunk to preserve overlap.
        next_start = max(start + 1, end - overlap)

        # Move forward to a whitespace boundary.
        while (
            next_start < end
            and not full_text[next_start - 1].isspace()
        ):
            next_start += 1

        start = next_start

    return chunks
