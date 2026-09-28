"""Shared helper for forcing a structured, tool-called response out of an LLM.

Used by every "oracle"-role call in this pipeline (corruption generation,
corruption verification, judge-rationale comparison) so they share one
tool-schema-construction and parsing path instead of each reimplementing it.
"""
from typing import TypeVar

import openai
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def structured_call(
    client: openai.OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    response_model: type[ModelT],
    tool_name: str,
) -> ModelT:
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Report structured output matching {response_model.__name__}.",
            "parameters": response_model.model_json_schema(),
        },
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )

    tool_call = response.choices[0].message.tool_calls[0]

    return response_model.model_validate_json(tool_call.function.arguments)
