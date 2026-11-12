
import unittest

from src.ingestion.recursive_chunker import recursive_chunk_pages


def make_page(text, page_number):
    return {
        "text": text,
        "metadata": {
            "source": "test.pdf",
            "page": page_number,
            "title": "Test Manual",
        },
    }


class TestRecursiveChunker(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(recursive_chunk_pages([]), [])

    def test_invalid_parameters(self):
        pages = [make_page("Example text.", 1)]

        with self.assertRaises(ValueError):
            recursive_chunk_pages(pages, chunk_size=0)

        with self.assertRaises(ValueError):
            recursive_chunk_pages(pages, chunk_size=100, overlap=100)

        with self.assertRaises(ValueError):
            recursive_chunk_pages(pages, chunk_size=100, overlap=-1)

    def test_word_boundaries_and_coverage(self):
        words = [f"word{i:04d}" for i in range(500)]
        text = " ".join(words)

        chunks = recursive_chunk_pages(
            [make_page(text, 1)],
            chunk_size=120,
            overlap=20,
        )

        for chunk in chunks:
            self.assertRegex(chunk["text"], r"^word\d{4}")
            self.assertRegex(chunk["text"], r"word\d{4}$")

        recovered = {
            word
            for chunk in chunks
            for word in chunk["text"].split()
        }

        self.assertTrue(set(words).issubset(recovered))

    def test_page_provenance(self):
        pages = [
            make_page("Alpha beta gamma.", 1),
            make_page("Delta epsilon zeta.", 2),
        ]

        chunks = recursive_chunk_pages(
            pages, chunk_size=30, overlap=5
        )

        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["metadata"]["pages"], [1])

        self.assertTrue(
            any(2 in chunk["metadata"]["pages"] for chunk in chunks)
        )

    def test_overlap(self):
        words = [f"word{i:04d}" for i in range(100)]
        text = " ".join(words)

        chunks = recursive_chunk_pages(
            [make_page(text, 1)],
            chunk_size=120,
            overlap=20,
        )

        self.assertGreater(len(chunks), 1)

        for previous, current in zip(chunks, chunks[1:]):
            self.assertTrue(
                set(previous["text"].split())
                & set(current["text"].split())
            )


if __name__ == "__main__":
    unittest.main()
