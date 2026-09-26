from enum import StrEnum
from dataclasses import dataclass

from pydantic import BaseModel


class CorruptionType(StrEnum):
    UNSUPPORTED_FACT = "unsupported_fact"
    FACTUAL_ALTERATION = "factual_alteration"
    QUALIFICATION_CHANGE = "qualification_change"
    SUPPORT_REMOVAL = "support_removal"
    CONTRADICTION = "contradiction"


class Corruption(BaseModel):
    corrupted_document: str
    corruption_type: CorruptionType
    corruption_description: str


@dataclass
class EvaluationExample:
    source: str
    summary: str
    expected_faithful: bool

    # Metadata for analysis/debugging.
    corruption_target: str | None = None
    corruption_type: str | None = None
    corruption_description: str | None = None