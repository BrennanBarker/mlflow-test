from mlflow.genai.judges.prompts.summarization import SUMMARIZATION_PROMPT

from mlflow.genai.judges import make_judge

INSTRUCTIONS = """Consider the following source document and candidate summary.
You must decide whether the summary includes information not present in the source document.

## Instructions
First, read the document and summary carefully.
Second, evaluate faithfulness: check whether every concrete claim in the summary is supported by the document. Emphasize the accuracy of the main facts rather than the exact phrasing. If the summary contradicts the document or invents information, it fails.

Return a value of True only if the summary is faithful to the document (no hallucinations or contradictions).
If not, return False.

When producing your rationale, ensure that you include a concise description of any specific information about any unfaithful summary content, or any possibly unfaithful summary content, even if deciding the summary is faithful.

<document>{{ inputs }}</document>

<summary>{{ outputs }}</summary>
"""

judge = make_judge(
    name="summary_faithfulness",
    description="Check whether every concrete claim in a summary is supported by its source material.",
    model="openai:/gpt-5.4-mini-2026-03-17",
    feedback_value_type=bool,
    generate_rationale_first=True,
    instructions=INSTRUCTIONS
)

feedback = judge(inputs={"document": "test"}, outputs={"dang": "cool"})