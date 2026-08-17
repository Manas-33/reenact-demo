"""A data-analyst agent (OpenAI SDK): inspect a table's schema, run read-only
SQL against it, and answer a question using only the figures the query returns.

Like the refund agent it knows nothing about reenact. It takes any object
exposing ``chat.completions.create(**kwargs)`` - a live ``openai.OpenAI`` client
when recording, a scripted fake in tests - plus an optional ``on_tool`` hook
reenact uses to record each tool call. Both tools are read-only, so a recorded
run replays offline with nothing re-executed. There is no mutating action here:
the analyst's angle is FAITHFULNESS - every number in the answer must come from
the query results, so an invented figure is what the gate catches.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from analyst_agent.fixtures import describe_schema, run_sql

MODEL = "gpt-4o"
MAX_TURNS = 6

SYSTEM_PROMPT = (
    "You are a data analyst. Answer the user's question about the sales data "
    "using ONLY the database. First call get_schema to see the table, then call "
    "run_query with a SQL SELECT to get the figures, then answer in one or two "
    "sentences. Every number in your answer must come from the query results - "
    "never estimate or invent a figure. Report the exact numbers the query returns."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Return the sales table's columns and types.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a read-only SQL SELECT and return the rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SQL SELECT statement.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
]

# Both tools only read data, so replay re-executes nothing - and the analyst has
# no mutating action at all. Its guarantee is faithfulness, not the refund
# agent's mutating-safety story.
TOOL_SIDE_EFFECTS: dict[str, str] = {
    "get_schema": "read_only",
    "run_query": "read_only",
}

# on_tool(name, arguments, result) - called after each tool runs, for the recorder.
ToolHook = Callable[[str, dict[str, Any], Any], None]


def get_schema() -> dict[str, Any]:
    """Return the sales table's schema (read-only)."""
    return describe_schema()


def run_query(sql: str) -> dict[str, Any]:
    """Run a read-only SELECT and return the resulting rows (read-only)."""
    return run_sql(sql)


def _execute(name: str, arguments: dict[str, Any]) -> Any:
    if name == "get_schema":
        return get_schema()
    if name == "run_query":
        return run_query(str(arguments.get("sql", "")))
    return {"error": f"unknown tool {name!r}"}


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Rebuild an assistant turn for the next request.

    Keeps only the fields the OpenAI API accepts back - the text content and any
    tool calls - dropping the null bookkeeping fields ``model_dump`` includes.
    """
    rebuilt: dict[str, Any] = {"role": "assistant"}
    content = message.get("content")
    if content:
        rebuilt["content"] = content
    tool_calls = message.get("tool_calls")
    if tool_calls:
        rebuilt["tool_calls"] = [
            {
                "id": call.get("id"),
                "type": "function",
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                },
            }
            for call in tool_calls
        ]
    return rebuilt


def run_analyst_agent(
    client: Any,
    question: str,
    on_tool: ToolHook | None = None,
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Answer ``question`` from the sales data and return the final reply text.

    Runs the OpenAI tool-use loop over ``client``: think -> call a tool -> think,
    until the model answers with no tool call. Each tool call is executed against
    the local fixtures and reported to ``on_tool`` (if given) for recording.

    ``system`` and ``tools`` default to the shipped prompt and tool set; a
    regression variant passes an override - a degraded prompt or a renamed tool - to
    seed a break the gate must catch.
    """
    active_system = system if system is not None else SYSTEM_PROMPT
    active_tools = tools if tools is not None else TOOLS
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": active_system},
        {"role": "user", "content": question},
    ]
    message: dict[str, Any] = {}
    for _ in range(MAX_TURNS):
        body: dict[str, Any] = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=active_tools,
        ).model_dump(mode="json")
        message = body["choices"][0]["message"]
        messages.append(_assistant_message(message))
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or ""
        for call in tool_calls:
            name = str(call["function"]["name"])
            raw_arguments = call["function"].get("arguments") or "{}"
            try:
                arguments: dict[str, Any] = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            result = _execute(name, arguments)
            if on_tool is not None:
                on_tool(name, arguments, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result),
                }
            )
    return message.get("content") or ""
