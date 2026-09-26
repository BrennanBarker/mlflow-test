import openai
import pandas as pd

from generate_corruption import generate_corruption
from make_example import make_corrupted_example
from validate import validate_corruption

client = openai.OpenAI()
corruption_models = {
    "summary": "gpt-5.4",
    "source": "gpt-5.4"
}

def augment_row(row):
    source, summary = row.source, row.summary

    for target in ("summary", "source"):
        corruption = generate_corruption(
            client,
            source,
            summary,
            target=target,
            model=corruption_models[target],
        )

        validate_corruption(
            source,
            summary,
            corruption,
            target=target,
        )

        example = make_corrupted_example(
            source,
            summary,
            corruption,
            target=target,
        )