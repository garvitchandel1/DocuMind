
def paragraph_chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """Create overlapping, paragraph-aware chunks with page citations."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    # Combine pages and retain each page's character offsets.
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

        page_start = position
        parts.append(text)
        position += len(text)

        page_ranges.append({
            "start": page_start,
            "end": position,
            "metadata": page["metadata"],
        })

    full_text = "".join(parts)

    if not full_text:
        return []

    chunks = []
    start = 0
    text_length = len(full_text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end

        
        if target_end < text_length:
            # Prefer paragraph boundaries, then sentence boundaries,
            # then word boundaries.
            minimum_end = start + int(chunk_size * 0.5)

            paragraph_end = full_text.rfind(
                "\n\n", start + 1, target_end
            )

            if paragraph_end >= minimum_end:
                end = paragraph_end
            else:
                sentence_end = max(
                    full_text.rfind(". ", start + 1, target_end),
                    full_text.rfind("? ", start + 1, target_end),
                    full_text.rfind("! ", start + 1, target_end),
                )

                if sentence_end >= minimum_end:
                    end = sentence_end + 1
                else:
                    end = target_end

                    while (
                        end < text_length
                        and not full_text[end].isspace()
                    ):
                        end += 1


        chunk_text = full_text[start:end].strip()

        if chunk_text:
            # Find all pages contributing actual text to this chunk.
            contributing_pages = [
                item for item in page_ranges
                if item["start"] < end and item["end"] > start
                and full_text[max(start, item["start"]):min(end, item["end"])].strip()
            ]

            first_metadata = contributing_pages[0]["metadata"]

            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "metadata": {
                    "source": first_metadata["source"],
                    "title": first_metadata.get("title"),
                    "pages": [
                        item["metadata"]["page"]
                        for item in contributing_pages
                    ],
                },
            })

        if end >= text_length:
            break

        # Target the configured overlap relative to the actual end.
        next_start = max(start + 1, end - overlap)

        # Advance to the beginning of a complete word.
        while (
            next_start < end
            and not full_text[next_start - 1].isspace()
        ):
            next_start += 1

        # Skip whitespace so chunks do not start with blank space.
        while (
            next_start < text_length
            and full_text[next_start].isspace()
        ):
            next_start += 1

        # Defensive guarantee against an infinite loop.
        if next_start <= start:
            raise RuntimeError("Chunker failed to advance.")

        start = next_start

    return chunks
