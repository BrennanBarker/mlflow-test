"""Run a faithfulness judge over the generated dataset and score it with the Judge Evaluator."""
import argparse
import dataclasses
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import openai
import pandas as pd
from mlflow.genai.judges import CategoricalRating, is_grounded

from dataset.types import EvaluationExample
from faithfulness_judge import build_faithfulness_judge
from judge_evaluator import evaluate_judge_output

DATASET_PATH = "dataset/generated/examples.parquet"
RESULTS_DIR = "dataset/generated/results"


@dataclass
class NormalizedFeedback:
    """Backend-agnostic judge output: judge_evaluator only ever sees this shape."""
    value: bool
    rationale: str
    unknown: bool = False


def _custom_judge_fn(model: str) -> Callable[[EvaluationExample], NormalizedFeedback]:
    judge = build_faithfulness_judge(model=model)

    def run(example: EvaluationExample) -> NormalizedFeedback:
        feedback = judge(inputs=example.source, outputs=example.summary)
        return NormalizedFeedback(value=bool(feedback.value), rationale=feedback.rationale or "")

    return run


def _is_grounded_judge_fn(model: str) -> Callable[[EvaluationExample], NormalizedFeedback]:
    # is_grounded is RAG-shaped (request/response/context); we have no
    # "request", so we treat the source document as context and the summary
    # as the response being checked for support.
    def run(example: EvaluationExample) -> NormalizedFeedback:
        feedback = is_grounded(
            request="",
            response=example.summary,
            context=example.source,
            model=model,
        )
        # feedback.value is a CategoricalRating enum, not a real bool --
        # bool(CategoricalRating.NO) is True, so it must be compared explicitly.
        rating = feedback.value
        return NormalizedFeedback(
            value=rating == CategoricalRating.YES,
            rationale=feedback.rationale or "",
            unknown=rating == CategoricalRating.UNKNOWN,
        )

    return run


JUDGE_BACKENDS = {
    "custom": _custom_judge_fn,
    "is_grounded": _is_grounded_judge_fn,
}


def main(judge_kind: str, judge_model: str, split: str) -> None:
    df = pd.read_parquet(DATASET_PATH)

    if split == "opt":
        df = df[~df.held_out]
    elif split == "eval":
        df = df[df.held_out]

    judge_fn = JUDGE_BACKENDS[judge_kind](judge_model)
    client = openai.OpenAI()

    rows = []
    for row in df.itertuples():
        example = EvaluationExample(
            example_id=row.example_id,
            source=row.source,
            summary=row.summary,
            expected_faithful=row.expected_faithful,
            held_out=row.held_out,
            corruption_target=row.corruption_target,
            corruption_type=row.corruption_type,
            corruption_description=row.corruption_description,
        )

        feedback = judge_fn(example)
        verdict = evaluate_judge_output(example, feedback, client=client)
        if feedback.unknown:
            verdict.needs_human_review = True

        rows.append(
            {
                **dataclasses.asdict(example),
                "judge_value": feedback.value,
                "judge_rationale": feedback.rationale,
                **dataclasses.asdict(verdict),
            }
        )

    results = pd.DataFrame(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", judge_model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results_path = f"{RESULTS_DIR}/{judge_kind}_{safe_model}_{split}_{timestamp}.parquet"
    results.to_parquet(results_path, index=False)

    accuracy = (results.verdict == "success").mean()
    print(f"Judge: {judge_kind} ({judge_model}), split={split}")
    print(f"Overall accuracy: {accuracy:.2%} ({len(results)} examples)")
    print(f"Flagged for human review: {results.needs_human_review.sum()}")
    print()
    print(
        results.groupby(["corruption_target", "corruption_type"])["verdict"]
        .apply(lambda s: (s == "success").mean())
    )
    print(f"\nFull results written to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=list(JUDGE_BACKENDS), default="custom")
    parser.add_argument("--judge-model", default="openai:/gpt-5.4-mini-2026-03-17")
    parser.add_argument("--split", choices=["opt", "eval", "all"], default="all")
    args = parser.parse_args()
    main(args.judge, args.judge_model, args.split)
