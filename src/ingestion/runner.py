from chunker import chunk_pages
from pdf_parser import parse_pdf



pages = parse_pdf("D:/Switch/Documind/data/raw/nasa_systems_engineering_handbook_0.pdf")

# chunks = chunk_pages(pages)

# print(f"Pages: {len(pages)}")
# print(f"Chunks: {len(chunks)}")

# for chunk in chunks[:3]:
#     print(chunk["metadata"])
#     print(chunk["text"][:200])
#     print("-" * 40)

from paragraph_chunker import chunk_pages

chunks = chunk_pages(pages)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print(chunk["metadata"])
    print("Length:", len(chunk["text"]))
    print(chunk["text"][:200])
    print("-" * 40)