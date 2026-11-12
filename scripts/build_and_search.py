
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.recursive_chunker import recursive_chunk_pages
from src.embeddings.embedder import TextEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore


PDF_PATH = Path("data/raw/Machine_learning.pdf")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5


def main():

    # 1. Parse PDF
    print("Parsing PDF...")
    pages = parse_pdf(PDF_PATH)

    # 2. Chunk document
    print("Chunking document...")
    chunks = recursive_chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    print(f"Generated {len(chunks)} chunks.")

    # 3. Generate embeddings
    print("Loading embedding model...")
    embedder = TextEmbedder()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.embed_documents(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # 4. Build FAISS index
    print("Building FAISS index...")

    vector_store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add_documents(chunks, embeddings)

    print(f"Indexed {vector_store.index.ntotal} vectors.")

    # 5. Interactive semantic search
    print("\nDocument ready for searching!")

    while True:

        query = input("\nEnter your question (or 'exit'): ")

        if query.lower().strip() == "exit":
            break

        if not query.strip():
            continue

        query_embedding = embedder.embed_query(query)

        results = vector_store.search(
            query_embedding,
            top_k=TOP_K,
        )

        print("\n--- Search Results ---")

        for rank, result in enumerate(results, start=1):

            document = result["document"]
            metadata = document["metadata"]

            print(f"\nRank: {rank}")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {metadata['source']}")
            print(f"Pages: {metadata['pages']}")
            print(f"Text:\n{document['text'][:700]}")


if __name__ == "__main__":
    main()
