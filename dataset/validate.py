from typing import Literal

from .types import Corruption


def validate_corruption(
    original_source: str,
    original_summary: str,
    corruption: Corruption,
    *,
    target: Literal["source", "summary"],
) -> None:
    if not corruption.corrupted_document.strip():
        raise ValueError("Corrupted document is empty.")

    original = (
        original_summary
        if target == "summary"
        else original_source
    )

    if corruption.corrupted_document == original:
        raise ValueError("Corruption made no change.")

    if not corruption.corruption_description.strip():
        raise ValueError("Missing corruption description.")