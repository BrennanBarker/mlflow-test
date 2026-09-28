"""Generate the labeled faithfulness-judge evaluation dataset.

Samples gold (document, summary) pairs from XSum, augments each with a
verified corruption of the summary and of the source, and registers the
resulting examples (clean + corrupted) as an mlflow.genai evaluation dataset
(also written to data/generated/examples.parquet for manual inspection).
"""
import dataclasses
import os

import mlflow
import mlflow.genai as genai
import openai
import pandas as pd
from mlflow.exceptions import MlflowException

from judge_eval.common.models import CORRUPTION_MODEL, REFEREE_MODEL
from judge_eval.common.tracking import DATASET_GENERATION_EXPERIMENT, FAITHFULNESS_DATASET_NAME
from judge_eval.dataset.augment import augment_row

GOLD_PATH = "data/original/xsum/test.parquet"
OUTPUT_PATH = "data/generated/examples.parquet"

# Every HELD_OUT_MODULUS-th gold row (by id) is held out as the "eval" split,
# never touched during judge/prompt optimization -- i.e. a 1-in-5, ~20% split.
HELD_OUT_MODULUS = 5


def _to_records(out_df: pd.DataFrame) -> list[dict]:
    """Shape generated examples as mlflow.genai dataset records.

    expected_faithful/corruption_description are genuine expectations -- the
    harness's FaithfulnessEvaluator scorer reads them directly to grade the
    judge under test. corruption_target/type are never read by the scorer;
    they only drive the per-category breakdown, so they're tags (arbitrary
    metadata), not expectations.
    """
    records = []
    for row in out_df.itertuples():
        records.append(
            {
                "inputs": {"source": row.source, "summary": row.summary},
                "expectations": {
                    "expected_faithful": bool(row.expected_faithful),
                    "corruption_description": (
                        ""
                        if pd.isna(row.corruption_description)
                        else row.corruption_description
                    ),
                },
                "tags": {
                    "example_id": row.example_id,
                    "held_out": str(bool(row.held_out)),
                    "corruption_target": (
                        "clean" if pd.isna(row.corruption_target) else row.corruption_target
                    ),
                    "corruption_type": (
                        "clean" if pd.isna(row.corruption_type) else row.corruption_type
                    ),
                },
            }
        )
    return records


def _register_dataset(out_df: pd.DataFrame, experiment_id: str) -> "genai.datasets.EvaluationDataset":
    # Regenerating overwrites from scratch (matches the parquet file's existing
    # overwrite semantics) rather than accumulating stale records across runs.
    try:
        existing = genai.get_dataset(name=FAITHFULNESS_DATASET_NAME)
        genai.delete_dataset(dataset_id=existing.dataset_id)
    except MlflowException:
        pass

    dataset = genai.create_dataset(name=FAITHFULNESS_DATASET_NAME, experiment_id=experiment_id)
    dataset.merge_records(_to_records(out_df))
    return dataset


def main(n_samples: int) -> None:
    gold = pd.read_parquet(GOLD_PATH).sample(n_samples, random_state=0)

    client = openai.OpenAI()

    examples = []
    dropped = 0
    for row in gold.itertuples():
        held_out = int(row.id) % HELD_OUT_MODULUS == 0
        row_examples = augment_row(
            client,
            row.document,
            row.summary,
            example_id_prefix=str(row.id),
            held_out=held_out,
        )
        dropped += 2 - (len(row_examples) - 1)
        examples.extend(row_examples)

    out_df = pd.DataFrame([dataclasses.asdict(e) for e in examples])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)

    n_clean = int((out_df.expected_faithful).sum())
    n_corrupted = int((~out_df.expected_faithful).sum())
    n_held_out = int(out_df.held_out.sum())

    mlflow.set_experiment(DATASET_GENERATION_EXPERIMENT)
    with mlflow.start_run(run_name=f"n_samples={n_samples}") as run:
        mlflow.log_params(
            {
                "n_samples": n_samples,
                "held_out_modulus": HELD_OUT_MODULUS,
                "corruption_model": CORRUPTION_MODEL,
                "referee_model": REFEREE_MODEL,
                "gold_path": GOLD_PATH,
            }
        )
        mlflow.log_metrics(
            {
                "n_examples": len(out_df),
                "n_clean": n_clean,
                "n_corrupted": n_corrupted,
                "n_dropped": dropped,
                "n_held_out": n_held_out,
            }
        )

        dataset = _register_dataset(out_df, experiment_id=run.info.experiment_id)
        mlflow.log_params({"dataset_id": dataset.dataset_id, "dataset_digest": dataset.digest})

        run_id = run.info.run_id

    print(f"Wrote {len(out_df)} examples to {OUTPUT_PATH}")
    print(f"Registered as mlflow dataset '{FAITHFULNESS_DATASET_NAME}' ({dataset.dataset_id})")
    print(f"  clean: {n_clean}, corrupted: {n_corrupted}, dropped by verification: {dropped}")
    print(f"  held out: {n_held_out}, available for optimization: {len(out_df) - n_held_out}")
    print(out_df.groupby(["corruption_target", "corruption_type"]).size())
    print(f"\nMLflow run: {run_id} (experiment: {DATASET_GENERATION_EXPERIMENT})")
