
def precision_at_k(retrieved_ids, qrels, k=5):
    retrieved = retrieved_ids[:k]

    judged = [
        doc_id for doc_id in retrieved
        if doc_id in qrels
    ]

    if not judged:
        return 0.0

    relevant = sum(
        1 for doc_id in judged
        if qrels[doc_id] >= 2
    )

    return relevant / len(judged)


def recall_at_k(retrieved_ids, qrels, k=5):
    retrieved = retrieved_ids[:k]

    relevant_ids = {
        doc_id for doc_id, score in qrels.items()
        if score >= 2
    }

    if not relevant_ids:
        return 0.0

    retrieved_relevant = sum(
        1 for doc_id in set(retrieved)
        if doc_id in relevant_ids
    )

    return retrieved_relevant / len(relevant_ids)


def reciprocal_rank_at_k(retrieved_ids, qrels, k=5):
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id not in qrels:
            continue

        if qrels[doc_id] >= 2:
            return 1 / rank

    return 0.0


def judgment_coverage_at_k(retrieved_ids, qrels, k=5):
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    judged = sum(
        1 for doc_id in retrieved
        if doc_id in qrels
    )

    return judged / len(retrieved)
