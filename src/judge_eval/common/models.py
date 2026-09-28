"""Model tiers for the faithfulness-judge pipeline.

CORRUPTION_MODEL and REFEREE_MODEL are the "oracle" roles: they generate
ground-truth corruptions and adjudicate the judge-under-test's output during
dataset generation, a rare, one-time step, so they should be the strongest
model we have. DEFAULT_JUDGE_MODEL is just a starting point for the judge
under test, the swept variable -- override it via --judge-model wherever the
CLI is invoked.

RATIONALE_COMPARISON_MODEL is a separate, cheaper tier for the harness's own
case-4 rationale comparison (FaithfulnessEvaluator, harness/evaluator.py):
unlike the oracle roles above, this fires on every corrupted example on every
evaluation run, and will fire far more under GEPA-style prompt optimization
(potentially ~max_metric_calls times), so cost matters more than for the
rare oracle calls.
"""

CORRUPTION_MODEL = "gpt-5.4"
REFEREE_MODEL = "gpt-5.4"
DEFAULT_JUDGE_MODEL = "openai:/gpt-5.4-mini-2026-03-17"
RATIONALE_COMPARISON_MODEL = "gpt-5.4-mini-2026-03-17"
