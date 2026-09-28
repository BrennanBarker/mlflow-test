"""Model tiers for the faithfulness-judge pipeline.

CORRUPTION_MODEL and REFEREE_MODEL are the "oracle" roles: they generate
ground-truth corruptions and adjudicate the judge-under-test's output, so
they should be the strongest model we have. DEFAULT_JUDGE_MODEL is just a
starting point for the judge under test, the swept variable -- override it
via --judge-model wherever the CLI is invoked.
"""

CORRUPTION_MODEL = "gpt-5.4"
REFEREE_MODEL = "gpt-5.4"
DEFAULT_JUDGE_MODEL = "openai:/gpt-5.4-mini-2026-03-17"
