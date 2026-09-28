"""Run a faithfulness judge over the generated dataset and score it with mlflow.genai.evaluate."""
from typing import Callable

import mlflow
import mlflow.genai as genai
import pandas as pd
from mlflow.genai.judges import CategoricalRating, is_grounded

from judge_eval.common.tracking import FAITHFULNESS_DATASET_NAME, JUDGE_EVALUATION_EXPERIMENT
from judge_eval.judges.faithfulness import build_faithfulness_judge

from .evaluator import FaithfulnessEvaluator, NormalizedFeedback


def _custom_judge_fn(model: str) -> Callable[[str, str], NormalizedFeedback]:
    judge = build_faithfulness_judge(model=model)

    def run(source: str, summary: str) -> NormalizedFeedback:
        feedback = judge(inputs=source, outputs=summary)
        return NormalizedFeedback(value=bool(feedback.value), rationale=feedback.rationale or "")

    return run


def _is_grounded_judge_fn(model: str) -> Callable[[str, str], NormalizedFeedback]:
    # is_grounded is RAG-shaped (request/response/context); we have no
    # "request", so we treat the source document as context and the summary
    # as the response being checked for support.
    def run(source: str, summary: str) -> NormalizedFeedback:
        feedback = is_grounded(request="", response=summary, context=source, model=model)
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


def _load_split(split: str) -> tuple["genai.datasets.EvaluationDataset", pd.DataFrame]:
    dataset = genai.get_dataset(name=FAITHFULNESS_DATASET_NAME)
    df = dataset.to_df()

    held_out = df["tags"].apply(lambda t: t.get("held_out") == "True")
    if split == "opt":
        df = df[~held_out]
    elif split == "eval":
        df = df[held_out]

    # Strip the dataset-registry bookkeeping columns (outputs, source*, etc.) --
    # an "outputs" column present alongside predict_fn is documented as
    # unsupported by mlflow.genai.evaluate, so only pass what it actually needs.
    return dataset, df[["inputs", "expectations", "tags"]]


def main(judge_kind: str, judge_model: str, split: str) -> None:
    dataset, eval_df = _load_split(split)

    judge_fn = JUDGE_BACKENDS[judge_kind](judge_model)

    def predict_fn(source: str, summary: str) -> NormalizedFeedback:
        return judge_fn(source, summary)

    scorer = FaithfulnessEvaluator()

    mlflow.set_experiment(JUDGE_EVALUATION_EXPERIMENT)
    with mlflow.start_run(run_name=f"{judge_kind}_{judge_model}_{split}"):
        mlflow.log_params(
            {
                "judge_kind": judge_kind,
                "judge_model": judge_model,
                "split": split,
                "referee_model": scorer.referee_model,
                "n_examples": len(eval_df),
                "dataset_id": dataset.dataset_id,
                "dataset_digest": dataset.digest,
            }
        )

        result = mlflow.genai.evaluate(data=eval_df, predict_fn=predict_fn, scorers=[scorer])

        # result.metrics is always populated straight from in-memory scorer output, so
        # it's the reliable source for the headline numbers. result.result_df additionally
        # requires every row's trace to have round-tripped through the tracking backend,
        # which is a taller order -- treat the per-category breakdown as best-effort.
        accuracy = result.metrics[f"{scorer.name}/mean"]
        n_needs_review = round(result.metrics["needs_human_review/mean"] * len(eval_df))

        mlflow.log_metrics(
            {"accuracy": accuracy, "needs_human_review_count": n_needs_review}
        )

        rdf = result.result_df
        value_col = f"{scorer.name}/value"

        run_id = mlflow.active_run().info.run_id

    print(f"Judge: {judge_kind} ({judge_model}), split={split}")
    print(f"Overall accuracy: {accuracy:.2%} ({len(eval_df)} examples)")
    print(f"Flagged for human review: {n_needs_review}")
    print()
    if rdf is not None:
        # corruption_target/type are tags (metadata), not expectations, so they
        # arrive as a raw dict in the "tags" column rather than exploded {name}/value
        # columns the way expectations and scorer feedback do.
        breakdown = rdf.assign(
            corruption_target=rdf["tags"].apply(lambda t: t.get("corruption_target", "unknown")),
            corruption_type=rdf["tags"].apply(lambda t: t.get("corruption_type", "unknown")),
        )
        print(breakdown.groupby(["corruption_target", "corruption_type"])[value_col].mean())
    else:
        print("(per-category breakdown unavailable -- inspect the run's traces in the MLflow UI)")
    print(f"\nMLflow run: {run_id} (experiment: {JUDGE_EVALUATION_EXPERIMENT})")
