
import unittest

from src.ingestion.paragraph_chunker import paragraph_chunk_pages


def make_page(text, page_number):
    return {
        "text": text,
        "metadata": {
            "source": "test.pdf",
            "page": page_number,
            "title": "Test Manual",
        },
    }


class TestParagraphChunker(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(paragraph_chunk_pages([]), [])

    def test_invalid_parameters(self):
        pages = [make_page("Example text.", 1)]

        with self.assertRaises(ValueError):
            paragraph_chunk_pages(pages, chunk_size=0)

        with self.assertRaises(ValueError):
            paragraph_chunk_pages(pages, chunk_size=100, overlap=100)

        with self.assertRaises(ValueError):
            paragraph_chunk_pages(pages, chunk_size=100, overlap=-1)

    def test_word_boundaries_and_coverage(self):
        words = [f"word{i:04d}" for i in range(500)]
        text = " ".join(words)

        chunks = paragraph_chunk_pages(
            [make_page(text, 1)],
            chunk_size=120,
            overlap=20,
        )

        # Every chunk starts and ends with a complete test word.
        for chunk in chunks:
            self.assertRegex(chunk["text"], r"^word\d{4}")
            self.assertRegex(chunk["text"], r"word\d{4}$")

        # Every original word must appear in at least one chunk.
        recovered = set(
            word
            for chunk in chunks
            for word in chunk["text"].split()
        )

        self.assertTrue(set(words).issubset(recovered))

    def test_page_provenance(self):
        pages = [
            make_page("Alpha beta gamma.", 1),
            make_page("Delta epsilon zeta.", 2),
        ]

        chunks = paragraph_chunk_pages(
            pages, chunk_size=30, overlap=5
        )

        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["metadata"]["pages"], [1])

        self.assertTrue(
            any(2 in chunk["metadata"]["pages"] for chunk in chunks)
        )

    def test_oversized_word(self):
        text = "a" * 150

        chunks = paragraph_chunk_pages(
            [make_page(text, 1)],
            chunk_size=100,
            overlap=10,
        )

        # Never split a word, even when it exceeds chunk_size.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)

    
    def test_chunks_have_configured_overlap(self):
        words = [f"word{i:04d}" for i in range(100)]
        text = " ".join(words)

        chunks = paragraph_chunk_pages(
            [make_page(text, 1)],
            chunk_size=120,
            overlap=20,
        )

        self.assertGreater(len(chunks), 1)

        for previous, current in zip(chunks, chunks[1:]):
            previous_words = previous["text"].split()
            current_words = current["text"].split()

            # There should be some shared words between adjacent chunks.
            self.assertTrue(
                set(previous_words) & set(current_words)
            )



if __name__ == "__main__":
    unittest.main()
