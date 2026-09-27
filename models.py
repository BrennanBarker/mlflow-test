"""Model tiers for the faithfulness-judge pipeline.

CORRUPTION_MODEL and REFEREE_MODEL are the "oracle" roles: they generate
ground-truth corruptions and adjudicate the judge-under-test's output, so
they should be the strongest model we have. The judge itself is the swept
variable and is configured separately wherever it's built.
"""

CORRUPTION_MODEL = "gpt-5.4"
REFEREE_MODEL = "gpt-5.4"
