import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.paragraph_chunker import paragraph_chunk_pages
from src.ingestion.recursive_chunker import recursive_chunk_pages
PDF_PATH = Path("data/raw/machine_learning.pdf")

pages = parse_pdf(PDF_PATH)
#chunks = paragraph_chunk_pages(pages, chunk_size=1200, overlap=200)
chunks = recursive_chunk_pages(
    pages,
    chunk_size=1200,
    overlap=200,
)
lengths = [len(chunk["text"]) for chunk in chunks]

print(f"Pages extracted: {len(pages)}")
print(f"Chunks created: {len(chunks)}")
print(f"Average chunk length: {sum(lengths) / len(lengths):.0f}")
print(f"Shortest chunk: {min(lengths)}")
print(f"Longest chunk: {max(lengths)}")

print("\n--- First 3 chunks ---")
for chunk in chunks[:3]:
    print(f"\nChunk ID: {chunk['chunk_id']}")
    print(f"Pages: {chunk['metadata']['pages']}")
    print(f"Characters: {len(chunk['text'])}")
    print(chunk["text"][:300].replace("\n", " "))


print("\n--- Chunk boundary inspection ---")

for i in range(min(5, len(chunks) - 1)):
    current = chunks[i]["text"]
    following = chunks[i + 1]["text"]

    print(f"\nChunk {i} ending:")
    print(repr(current[-150:]))

    print(f"Chunk {i + 1} starting:")
    print(repr(following[:150]))


print("\n--- Page-level inspection ---")

for page in pages[:12]:
    print(f"\nPage {page['metadata']['page']}:")
    print(page["text"][:250].replace("\n", " "))


print("\n--- Actual overlap inspection ---")

for previous, current in zip(chunks[:5], chunks[1:6]):
    previous_text = previous["text"]
    current_text = current["text"]

    overlap_chars = 0
    max_overlap = min(len(previous_text), len(current_text))

    for size in range(1, max_overlap + 1):
        if previous_text.endswith(current_text[:size]):
            overlap_chars = size

    print(
        f"Chunks {previous['chunk_id']} -> {current['chunk_id']}: "
        f"{overlap_chars} matching characters"
    )