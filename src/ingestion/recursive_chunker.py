
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def recursive_chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    documents = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata=page["metadata"].copy(),
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    split_documents = splitter.split_documents(documents)

    chunks = []

    for index, document in enumerate(split_documents):
        metadata = document.metadata

        chunks.append({
            "chunk_id": index,
            "text": document.page_content,
            "metadata": {
                "source": metadata["source"],
                "title": metadata.get("title"),
                "pages": [metadata["page"]],
                "start_index": metadata.get("start_index"),
            },
        })

    return chunks
