"""Generate the labeled faithfulness-judge evaluation dataset.

Samples gold (document, summary) pairs from XSum, augments each with a
verified corruption of the summary and of the source, and writes the
resulting examples (clean + corrupted) to dataset/generated/examples.parquet.
"""
import argparse
import dataclasses
import os

import openai
import pandas as pd

from dataset.augment import augment_row

GOLD_PATH = "dataset/original/xsum/test.parquet"
OUTPUT_PATH = "dataset/generated/examples.parquet"


def main(n_samples: int) -> None:
    gold = pd.read_parquet(GOLD_PATH).sample(n_samples, random_state=0)

    client = openai.OpenAI()

    examples = []
    dropped = 0
    for row in gold.itertuples():
        held_out = int(row.id) % 5 == 0
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

    n_clean = (out_df.expected_faithful).sum()
    n_corrupted = (~out_df.expected_faithful).sum()
    n_held_out = out_df.held_out.sum()

    print(f"Wrote {len(out_df)} examples to {OUTPUT_PATH}")
    print(f"  clean: {n_clean}, corrupted: {n_corrupted}, dropped by verification: {dropped}")
    print(f"  held out: {n_held_out}, available for optimization: {len(out_df) - n_held_out}")
    print(out_df.groupby(["corruption_target", "corruption_type"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()
    main(args.n_samples)
