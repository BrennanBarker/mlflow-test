"""Score a faithfulness judge's output against the labeled dataset produced by `judge-eval generate-dataset`.

Implements the four cases from README.md by comparing the judge's verdict
(True = judge says faithful) against the labeled expectations. Judge-agnostic:
this only depends on the normalized judge output and the dataset's labels, not
on how the judge under test is built.
"""
from dataclasses import dataclass
from functools import lru_cache

import openai
from mlflow.entities import Feedback
from mlflow.genai.scorers import Scorer

from judge_eval.common.models import RATIONALE_COMPARISON_MODEL
from judge_eval.common.structured_call import structured_call
from judge_eval.dataset.types import RationaleComparison

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
class NormalizedFeedback:
    """Backend-agnostic judge output: FaithfulnessEvaluator only ever sees this shape."""
    value: bool
    rationale: str
    unknown: bool = False


# Unbounded: GEPA-style optimization re-runs the same dataset across many
# candidate prompts, and near-converged candidates often repeat a prior
# trial's rationale verbatim for a given example -- caching on the
# (description, rationale, model) triple skips a real referee call whenever
# that happens. Requires compare_rationales to build its own client rather
# than take one as an argument, since arguments must be hashable for
# lru_cache and an openai.OpenAI() instance isn't (and isn't stable across
# calls anyway).
@lru_cache(maxsize=None)
def compare_rationales(
    corruption_description: str,
    judge_rationale: str,
    *,
    model: str,
) -> RationaleComparison:
    return structured_call(
        openai.OpenAI(),
        model=model,
        system_prompt=COMPARISON_PROMPT,
        user_content=(
            f"<induced_corruption>\n{corruption_description}\n</induced_corruption>\n\n"
            f"<judge_rationale>\n{judge_rationale}\n</judge_rationale>"
        ),
        response_model=RationaleComparison,
        tool_name="submit_comparison",
    )


class FaithfulnessEvaluator(Scorer):
    """mlflow.genai.evaluate scorer: grades a judge's output against a labeled example.

    `referee_model` is a plain string field (not a stored client) so this
    stays JSON-dumpable, matching how MLflow's own built-in scorers hold
    `model: str` and resolve the actual call at invocation time rather than
    persisting a live client.
    """

    name: str = "faithfulness_evaluator"
    referee_model: str = RATIONALE_COMPARISON_MODEL

    def __call__(self, *, outputs: NormalizedFeedback, expectations: dict) -> list[Feedback]:
        judge_says_faithful = bool(outputs.value)
        judge_rationale = outputs.rationale or ""
        label_says_faithful = expectations["expected_faithful"]
        corruption_description = expectations.get("corruption_description") or ""

        if judge_says_faithful and not label_says_faithful:
            # Case 1: judge missed an induced corruption.
            return self._verdict(
                success=False, needs_review=outputs.unknown, rationale=corruption_description
            )

        if not judge_says_faithful and label_says_faithful:
            # Case 2: judge flagged a clean example as unfaithful.
            return self._verdict(success=False, needs_review=True, rationale=judge_rationale)

        if judge_says_faithful and label_says_faithful:
            # Case 3: judge correctly passed a clean example.
            return self._verdict(
                success=True, needs_review=outputs.unknown, rationale=judge_rationale
            )

        # Case 4: judge correctly flagged a corrupted example -- confirm it's
        # for the right reason.
        comparison = compare_rationales(
            corruption_description,
            judge_rationale,
            model=self.referee_model,
        )
        return self._verdict(
            success=comparison.matches,
            needs_review=(not comparison.matches) or outputs.unknown,
            rationale=comparison.rationale,
        )

    def _verdict(self, *, success: bool, needs_review: bool, rationale: str) -> list[Feedback]:
        return [
            Feedback(name=self.name, value=success, rationale=rationale),
            Feedback(name="needs_human_review", value=needs_review),
        ]
