"""Score a faithfulness judge's output against the labeled dataset produced by `judge-eval generate-dataset`.

Implements the four cases from README.md by comparing the judge's verdict
(True = judge says faithful) against EvaluationExample.expected_faithful.
"""
from dataclasses import dataclass
from typing import Literal

import openai
from mlflow.entities.assessment import Feedback

from judge_eval.common.models import REFEREE_MODEL
from judge_eval.common.structured_call import structured_call
from judge_eval.dataset.types import EvaluationExample, RationaleComparison

COMPARISON_PROMPT = """\
You are checking whether a faithfulness judge caught the *same* problem
that was deliberately induced in an evaluation example, or whether it
happened to flag the example as unfaithful for an unrelated reason.

You are given:
- the description of the corruption that was deliberately induced (ground truth)
- the judge's own rationale for why it marked the example unfaithful

Return matches=True only if the judge's rationale refers to the same
underlying claim/fact as the induced corruption. If the judge is pointing at
something else entirely, return matches=False.
"""


@dataclass
class EvaluatorVerdict:
    verdict: Literal["success", "failure"]
    needs_human_review: bool
    rationale: str


def compare_rationales(
    client: openai.OpenAI,
    corruption_description: str,
    judge_rationale: str,
    *,
    model: str,
) -> RationaleComparison:
    return structured_call(
        client,
        model=model,
        system_prompt=COMPARISON_PROMPT,
        user_content=(
            f"<induced_corruption>\n{corruption_description}\n</induced_corruption>\n\n"
            f"<judge_rationale>\n{judge_rationale}\n</judge_rationale>"
        ),
        response_model=RationaleComparison,
        tool_name="submit_comparison",
    )


def evaluate_judge_output(
    example: EvaluationExample,
    judge_feedback: Feedback,
    *,
    client: openai.OpenAI,
    referee_model: str = REFEREE_MODEL,
) -> EvaluatorVerdict:
    judge_says_faithful = bool(judge_feedback.value)
    label_says_faithful = example.expected_faithful

    if judge_says_faithful and not label_says_faithful:
        # Case 1: judge missed an induced corruption.
        return EvaluatorVerdict(
            verdict="failure",
            needs_human_review=False,
            rationale=example.corruption_description or "",
        )

    if not judge_says_faithful and label_says_faithful:
        # Case 2: judge flagged a clean example as unfaithful.
        return EvaluatorVerdict(
            verdict="failure",
            needs_human_review=True,
            rationale=judge_feedback.rationale or "",
        )

    if judge_says_faithful and label_says_faithful:
        # Case 3: judge correctly passed a clean example.
        return EvaluatorVerdict(
            verdict="success",
            needs_human_review=False,
            rationale=judge_feedback.rationale or "",
        )

    # Case 4: judge correctly flagged a corrupted example -- confirm it's for
    # the right reason.
    comparison = compare_rationales(
        client,
        example.corruption_description or "",
        judge_feedback.rationale or "",
        model=referee_model,
    )

    return EvaluatorVerdict(
        verdict="success" if comparison.matches else "failure",
        needs_human_review=not comparison.matches,
        rationale=comparison.rationale,
    )
