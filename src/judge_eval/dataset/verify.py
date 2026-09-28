from typing import Literal

import openai

from judge_eval.common.structured_call import structured_call

from .types import Corruption, VerificationResult

VERIFICATION_PROMPTS = {
    "summary": """\
You are checking synthetic evaluation data for a faithfulness model judge.

You are given a source document, an original faithful summary, and a
"corrupted" version of the summary that was supposed to introduce exactly
ONE unsupported factual claim.

Confirm all of the following:
- The corrupted summary differs from the original by a single, minimal
  factual change (not a rewrite, not multiple changes).
- The changed claim is genuinely NOT supported by the source document.
- The rest of the corrupted summary remains faithful to the source.

Return valid=True only if all of these hold. Otherwise return valid=False
and explain which condition failed.
""",
    "source": """\
You are checking synthetic evaluation data for a faithfulness model judge.

You are given an original source document, a summary, and a "corrupted"
version of the source document that was supposed to remove support for
exactly ONE claim made in the (unchanged) summary.

Confirm all of the following:
- The corrupted source differs from the original by a single, minimal
  change (not a rewrite, not multiple changes).
- The identified summary claim is genuinely NOT supported by the corrupted
  source.
- The rest of the corrupted source remains otherwise faithful/plausible.

Return valid=True only if all of these hold. Otherwise return valid=False
and explain which condition failed.
""",
}


def verify_corruption(
    client: openai.OpenAI,
    source: str,
    summary: str,
    corruption: Corruption,
    *,
    target: Literal["source", "summary"],
    model: str,
) -> VerificationResult:
    return structured_call(
        client,
        model=model,
        system_prompt=VERIFICATION_PROMPTS[target],
        user_content=(
            f"<source>\n{source}\n</source>\n\n"
            f"<summary>\n{summary}\n</summary>\n\n"
            f"<corrupted_{target}>\n{corruption.corrupted_document}\n</corrupted_{target}>\n\n"
            f"<claimed_corruption_type>{corruption.corruption_type}</claimed_corruption_type>\n"
            f"<claimed_corruption_description>{corruption.corruption_description}</claimed_corruption_description>"
        ),
        response_model=VerificationResult,
        tool_name="submit_verification",
    )
