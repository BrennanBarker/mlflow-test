CORRUPTION_PROMPTS = {
    "summary": """\
You are generating synthetic evaluation data for a faithfulness model judge.

You are given a source document and a faithful summary of that document.

Your task is to introduce exactly ONE factual corruption into the summary.

The resulting summary must contain a factual claim that is not supported by
the source.

Make the smallest realistic change possible. Prefer subtle changes such as:

- adding a fact not present in the source
- changing a number, date, name, location, or other factual detail
- changing a relationship between entities
- changing the strength of a qualification or modality
- turning an unsupported inference into an assertion

Do not modify the source.
Do not introduce unrelated errors.
Keep the resulting summary otherwise faithful to the source.

Return:
- corrupted_document: the complete corrupted summary
- corruption_type: the type of corruption introduced
- corruption_description: a concise description of exactly what was changed
  and why the resulting claim is unsupported or incorrect

The corruption_description is ground truth for evaluating a separate
faithfulness judge. Do not discuss whether a judge would detect it.
""",

    "source": """\
You are generating synthetic evaluation data for a faithfulness model judge.

You are given a source document and a faithful summary of that document.

Your task is to modify ONLY the source document so that one factual claim
in the unchanged summary is no longer supported by the source.

The summary must remain exactly unchanged.

Make exactly ONE minimal, realistic change to the source.

Preferred corruptions include:

- removing the sentence or clause that supports a summary claim
- removing a specific fact, number, date, or entity
- changing a factual detail so that the source no longer supports the summary
- changing the source so that it contradicts the summary
- removing a qualification or context necessary to support the summary claim

Do not modify unrelated content.
Keep the resulting source plausible and natural.

It is acceptable, and desirable, for the summary's claim to remain true in
the real world. What matters is that the modified source no longer provides
adequate support for it.

Return:
- corrupted_document: the complete corrupted source
- corruption_type: the type of corruption introduced
- corruption_description: a concise description of exactly what was changed
  and why the resulting source no longer supports the summary

The corruption_description is ground truth for evaluating a separate
faithfulness judge. Do not discuss whether a judge would detect it.
"""
}


