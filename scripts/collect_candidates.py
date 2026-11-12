
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.paragraph_chunker import paragraph_chunk_pages
from src.ingestion.recursive_chunker import recursive_chunk_pages
from src.embeddings.embedder import TextEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore


PDF_PATH = ROOT / "data" / "raw" / "Machine_learning.pdf"
QUERY_PATH = ROOT / "experiments" / "evaluation" / "queries.json"
RESULTS_DIR = ROOT / "experiments" / "evaluation" / "results"

TOP_K = 20
CHUNK_SIZE = 1200
OVERLAP = 200

CHUNKERS = {
    "fixed": chunk_pages,
    "paragraph": paragraph_chunk_pages,
    "recursive": recursive_chunk_pages,
}


def collect_candidates(strategy, pages, queries, embedder):

    chunker = CHUNKERS[strategy]

    # Generate chunks
    chunks = chunker(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    print(f"\nStrategy: {strategy}")
    print(f"Generated {len(chunks)} chunks")

    # Generate embeddings
    embeddings = embedder.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    # Build vector store
    store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    store.add_documents(chunks, embeddings)

    # Retrieve candidates
    candidate_results = {}

    for query_id, query_data in queries.items():

        query = query_data["query"]
        query_embedding = embedder.embed_query(query)

        results = store.search(
            query_embedding,
            top_k=TOP_K
        )

        candidate_results[query_id] = {
            "query": query,
            "results": []
        }

        for rank, result in enumerate(results, start=1):

            document = result["document"]

            candidate_results[query_id]["results"].append({
                "candidate_id": f"{strategy}:{document['chunk_id']}",
                "strategy": strategy,
                "rank": rank,
                "score": result["score"],
                "chunk_id": document["chunk_id"],
                "text": document["text"],
                "metadata": document["metadata"]
            })

    # Save results
    output_path = RESULTS_DIR / f"candidates_{strategy}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            candidate_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved candidates to: {output_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        choices=["fixed", "paragraph", "recursive"],
        required=True
    )

    args = parser.parse_args()

    # Load inputs
    with open(QUERY_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    pages = parse_pdf(PDF_PATH)

    print(f"Loaded {len(pages)} pages")

    # Initialize embedder once
    embedder = TextEmbedder()

    collect_candidates(
        args.strategy,
        pages,
        queries,
        embedder
    )


if __name__ == "__main__":
    main()
