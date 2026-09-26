from typing import Literal

import openai

from .types import Corruption
from .prompts import CORRUPTION_PROMPTS

CORRUPTION_TOOL = {
    "type": "function",
    "function": {
        "name": "create_corruption",
        "description": "Create exactly one controlled corruption.",
        "parameters": Corruption.model_json_schema(),
    },
}

def generate_corruption(
    client: openai.OpenAI,
    source: str,
    summary: str,
    *,
    target: Literal["source", "summary"],
    model: str,
) -> Corruption:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": CORRUPTION_PROMPTS[target],
            },
            {
                "role": "user",
                "content": (
                    f"<source>\n{source}\n</source>\n\n"
                    f"<summary>\n{summary}\n</summary>"
                ),
            },
        ],
        tools=[CORRUPTION_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "create_corruption"},
        },
    )

    tool_call = response.choices[0].message.tool_calls[0]

    corruption = Corruption.model_validate_json(tool_call.function.arguments)

    original = source if target == "source" else summary

    if corruption.corrupted_document == original:
        raise ValueError("Corruption made no change.")

    return corruption