from typing import Literal

import openai

from .generate_corruption import generate_corruption
from .verify import verify_corruption
from .types import Corruption, EvaluationExample
from models import CORRUPTION_MODEL, REFEREE_MODEL


def _validate_corruption(
    original_source: str,
    original_summary: str,
    corruption: Corruption,
    *,
    target: Literal["source", "summary"],
) -> None:
    if not corruption.corrupted_document.strip():
        raise ValueError("Corrupted document is empty.")

    original = original_summary if target == "summary" else original_source
    if corruption.corrupted_document == original:
        raise ValueError("Corruption made no change.")

    if not corruption.corruption_description.strip():
        raise ValueError("Missing corruption description.")


def _make_corrupted_example(
    source: str,
    summary: str,
    corruption: Corruption,
    *,
    target: Literal["source", "summary"],
    example_id: str,
    held_out: bool,
) -> EvaluationExample:
    if target == "source":
        source = corruption.corrupted_document
    else:
        summary = corruption.corrupted_document

    return EvaluationExample(
        example_id=example_id,
        source=source,
        summary=summary,
        expected_faithful=False,
        held_out=held_out,
        corruption_target=target,
        corruption_type=corruption.corruption_type,
        corruption_description=corruption.corruption_description,
    )


def _make_clean_example(
    source: str,
    summary: str,
    *,
    example_id: str,
    held_out: bool,
) -> EvaluationExample:
    return EvaluationExample(
        example_id=example_id,
        source=source,
        summary=summary,
        expected_faithful=True,
        held_out=held_out,
    )


def augment_row(
    client: openai.OpenAI,
    source: str,
    summary: str,
    *,
    example_id_prefix: str,
    held_out: bool,
) -> list[EvaluationExample]:
    """Build the clean example plus any successfully-verified corruptions for one gold row."""
    examples = [
        _make_clean_example(
            source,
            summary,
            example_id=f"{example_id_prefix}-clean",
            held_out=held_out,
        )
    ]

    for target in ("summary", "source"):
        try:
            corruption = generate_corruption(
                client,
                source,
                summary,
                target=target,
                model=CORRUPTION_MODEL,
            )

            _validate_corruption(
                source,
                summary,
                corruption,
                target=target,
            )

            verification = verify_corruption(
                client,
                source,
                summary,
                corruption,
                target=target,
                model=REFEREE_MODEL,
            )
        except Exception as e:
            print(f"[augment] skipping {target} corruption for {example_id_prefix}: {e}")
            continue

        if not verification.valid:
            print(
                f"[augment] dropping {target} corruption for {example_id_prefix}: "
                f"failed verification ({verification.rationale})"
            )
            continue

        examples.append(
            _make_corrupted_example(
                source,
                summary,
                corruption,
                target=target,
                example_id=f"{example_id_prefix}-{target}",
                held_out=held_out,
            )
        )

    return examples
