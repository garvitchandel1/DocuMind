
from pathlib import Path
import re
import pymupdf


def clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    # Join words split by soft hyphens across line breaks.
    text = text.replace("\u00ad\n", "")
    text = text.replace("\u00ad", "")

    # Normalize non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # Remove trailing whitespace from each line.
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Collapse excessive blank lines, preserving paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_pdf(file_path: str | Path) -> list[dict]:
    """Extract cleaned text and metadata from a PDF, page by page."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Expected a PDF file.")

    pages = []

    with pymupdf.open(file_path) as pdf:
        document_metadata = pdf.metadata or {}

        for page_number in range(len(pdf)):
            page = pdf[page_number]
            raw_text = str(page.get_text("text"))
            text = clean_text(raw_text)

            pages.append({
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "page": page_number+1,
                    "title": document_metadata.get("title"),
                },
            })

    return pages
