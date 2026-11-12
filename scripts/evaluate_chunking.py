import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "experiments" / "evaluation" / "results"
ASSESSMENTS_PATH = ROOT / "experiments" / "evaluation" / "assessments.json"

STRATEGIES = ["fixed", "paragraph", "recursive"]
K_VALUES = [5, 10, 20]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    assessments = load_json(ASSESSMENTS_PATH)

    # Load candidates and calculate raw utility scores.
    all_candidates = defaultdict(list)

    for strategy in STRATEGIES:
        candidate_data = load_json(
            RESULTS_DIR / f"candidates_{strategy}.json"
        )

        for query_id, query_data in candidate_data.items():
            for candidate in query_data["results"]:
                candidate_id = f"{strategy}:{candidate['chunk_id']}"

                if candidate_id not in assessments.get(query_id, {}):
                    raise ValueError(
                        f"Missing assessment: {query_id}/{candidate_id}"
                    )

                assessment = assessments[query_id][candidate_id]

                relevance = assessment["relevance"]
                information = assessment["relevant_information"]
                word_count = len(candidate["text"].split())

                raw_score = relevance * word_count * (information / 10)

                all_candidates[query_id].append({
                    "strategy": strategy,
                    "candidate_id": candidate_id,
                    "rank": candidate["rank"],
                    "word_count": word_count,
                    "relevance": relevance,
                    "relevant_information": information,
                    "raw_score": raw_score,
                })
        # Diagnostic analysis across all retrieved candidates.
    strategy_stats = defaultdict(list)

    for candidates in all_candidates.values():
        for candidate in candidates:
            strategy_stats[candidate["strategy"]].append(candidate)

    print("\nCHUNKING DIAGNOSTICS\n")

    for strategy in STRATEGIES:
        candidates = strategy_stats[strategy]
        total = len(candidates)

        avg_length = sum(
            c["word_count"] for c in candidates
        ) / total

        avg_relevance = sum(
            c["relevance"] for c in candidates
        ) / total

        avg_information = sum(
            c["relevant_information"] for c in candidates
        ) / total

        distribution = {
            score: sum(
                1 for c in candidates if c["relevance"] == score
            )
            for score in range(4)
        }

        print(f"\n{strategy.upper()} ({total} candidates)")
        print(f"Average word count: {avg_length:.1f}")
        print(f"Average relevance: {avg_relevance:.2f}/3")
        print(f"Average information: {avg_information:.2f}/10")
        print(f"Relevance distribution: {distribution}")
    # Normalize scores per query across all three strategies.
    for query_id, candidates in all_candidates.items():
        max_score = max(c["raw_score"] for c in candidates)

        for candidate in candidates:
            candidate["normalized_score"] = (
                candidate["raw_score"] / max_score
                if max_score > 0 else 0
            )
        # Rank-weighted relevance
    weighted_summary = {}

    for query_id, candidates in all_candidates.items():
        weighted_summary[query_id] = {}

        for strategy in STRATEGIES:
            strategy_candidates = sorted(
                [c for c in candidates if c["strategy"] == strategy],
                key=lambda c: c["rank"]
            )

            weighted_summary[query_id][strategy] = {}

            for k in K_VALUES:
                top_k = strategy_candidates[:k]

                weighted_sum = sum(
                    c["relevance"] / math.log2(c["rank"] + 1)
                    for c in top_k
                )

                weight_sum = sum(
                    1 / math.log2(c["rank"] + 1)
                    for c in top_k
                )

                weighted_summary[query_id][strategy][f"top_{k}"] = (
                    weighted_sum / weight_sum
                )

    print("\nRANK-WEIGHTED RELEVANCE (0–3)\n")

    for query_id, strategies in weighted_summary.items():
        print(f"\n{query_id}")
        print(f"{'Strategy':<15}{'Top 5':>10}{'Top 10':>10}{'Top 20':>10}")

        for strategy, scores in strategies.items():
            print(
                f"{strategy:<15}"
                f"{scores['top_5']:>10.4f}"
                f"{scores['top_10']:>10.4f}"
                f"{scores['top_20']:>10.4f}"
            )

    print("\nOVERALL RANK-WEIGHTED RELEVANCE\n")

    for strategy in STRATEGIES:
        print(f"\n{strategy.upper()}")

        for k in K_VALUES:
            key = f"top_{k}"
            average = sum(
                weighted_summary[q][strategy][key]
                for q in weighted_summary
            ) / len(weighted_summary)

            print(f"{key}: {average:.4f}")
    # Aggregate results by query and strategy.
    summary = {}

    for query_id, candidates in all_candidates.items():
        summary[query_id] = {}

        for strategy in STRATEGIES:
            strategy_candidates = sorted(
                [c for c in candidates if c["strategy"] == strategy],
                key=lambda c: c["rank"]
            )

            summary[query_id][strategy] = {}

            for k in K_VALUES:
                top_k = strategy_candidates[:k]

                avg_score = sum(
                    c["normalized_score"] for c in top_k
                ) / len(top_k)

                summary[query_id][strategy][f"top_{k}"] = round(
                    avg_score, 4
                )

    # Print results.
    print("\nNORMALIZED UTILITY SCORES\n")

    for query_id, strategies in summary.items():
        print(f"\n{query_id}")

        print(
            f"{'Strategy':<15}"
            f"{'Top 5':>10}"
            f"{'Top 10':>10}"
            f"{'Top 20':>10}"
        )

        for strategy, scores in strategies.items():
            print(
                f"{strategy:<15}"
                f"{scores['top_5']:>10.4f}"
                f"{scores['top_10']:>10.4f}"
                f"{scores['top_20']:>10.4f}"
            )

    output_path = RESULTS_DIR / "evaluation_summary.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()