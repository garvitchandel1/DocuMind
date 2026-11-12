import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.recursive_chunker import recursive_chunk_pages
from src.embeddings.embedder import TextEmbedder


PDF_PATH = Path("data/raw/Machine_learning.pdf")

# 1. Parse and chunk the document
pages = parse_pdf(PDF_PATH)
chunks = recursive_chunk_pages(
    pages,
    chunk_size=1200,
    overlap=200,
)

# 2. Initialise the embedding model
embedder = TextEmbedder()

# 3. Embed document chunks
texts = [chunk["text"] for chunk in chunks[:10]]
document_embeddings = embedder.embed_documents(texts)

# 4. Embed a query
query = "What is supervised machine learning?"
query_embedding = embedder.embed_query(query)

# 5. Inspect the results
print("Document embeddings shape:", document_embeddings.shape)
print("Query embedding shape:", query_embedding.shape)

print(
    "Document vector norm:",
    round(float(document_embeddings[0] @ document_embeddings[0]) ** 0.5, 4)
)

print(
    "Query vector norm:",
    round(float(query_embedding @ query_embedding) ** 0.5, 4)
)
