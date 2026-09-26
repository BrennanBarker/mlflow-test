from typing import Literal

from .types import EvaluationExample, Corruption


def make_corrupted_example(
    source: str,
    summary: str,
    corruption: Corruption,
    *,
    target: Literal["source", "summary"],
) -> EvaluationExample:
    if target == "source":
        source = corruption.corrupted_document
    else:
        summary = corruption.corrupted_document

    return EvaluationExample(
        source=source,
        summary=summary,
        expected_faithful=False,
        corruption_target=target,
        corruption_type=corruption.corruption_type,
        corruption_description=corruption.corruption_description,
    )

def make_clean_example(
    source: str,
    summary: str,
) -> EvaluationExample:
    return EvaluationExample(
        source=source,
        summary=summary,
        expected_faithful=True,
    )
