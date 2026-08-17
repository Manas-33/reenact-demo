"""Keyless tests for the analyst agent.

A scripted fake OpenAI client drives the *real* agent offline - no key, no
network - the same record-real / test-keyless split reenact itself uses. These
prove the loop's shape and the property the faithfulness criterion later guards:
the figure in the answer is one the query actually returned.
"""

from __future__ import annotations

import json
from typing import Any

import reenact
from reenact.schema import SideEffect, ToolCallEvent, Trajectory

from analyst_agent.agent import TOOL_SIDE_EFFECTS, run_analyst_agent
from analyst_agent.fixtures import describe_schema, run_sql


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _assistant_tool_turn(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def _assistant_text_turn(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


class _Response:
    def __init__(self, message: dict[str, Any]) -> None:
        self._message = message

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"choices": [{"message": self._message}]}


class _Completions:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = script
        self._turn = 0

    def create(self, **_kwargs: Any) -> _Response:
        message = self._script[self._turn]
        self._turn += 1
        return _Response(message)


class _Chat:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.completions = _Completions(script)


class _ScriptedClient:
    """A fake OpenAI client that returns pre-scripted turns in order."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.chat = _Chat(script)


def _record(script: list[dict[str, Any]], question: str) -> tuple[Trajectory, str]:
    client = _ScriptedClient(script)
    with reenact.recording(client) as rec:

        def on_tool(name: str, arguments: dict[str, Any], result: Any) -> None:
            rec.record_tool_call(
                name=name,
                arguments=arguments,
                result=result,
                side_effect=SideEffect(TOOL_SIDE_EFFECTS[name]),
            )

        answer = run_analyst_agent(client, question, on_tool=on_tool)
    return rec.trajectory, answer


def test_answer_number_is_grounded_in_query() -> None:
    # The property the faithfulness criterion guards: the figure the agent reports
    # is one the query actually returned. The West-region revenue really is 12,500,
    # so the anchor below is grounded, not hand-waved.
    west_sql = "SELECT SUM(revenue) AS total FROM sales WHERE region = 'West'"
    grounded = run_sql(west_sql)
    assert grounded["rows"][0]["total"] == 12500.0

    question = "What was our total revenue in the West region?"
    script = [
        _assistant_tool_turn(_tool_call("c1", "get_schema", {})),
        _assistant_tool_turn(_tool_call("c2", "run_query", {"sql": west_sql})),
        _assistant_text_turn("Total revenue in the West region was $12,500.00."),
    ]
    trajectory, answer = _record(script, question)

    kinds = [event.type for event in trajectory.events]
    assert kinds == ["llm_call", "tool_call", "llm_call", "tool_call", "llm_call"]
    tool_calls = [e for e in trajectory.events if isinstance(e, ToolCallEvent)]
    assert [e.name for e in tool_calls] == ["get_schema", "run_query"]
    # No mutating tool exists - both reads are read-only, so replay re-fires nothing.
    assert all(e.side_effect is SideEffect.READ_ONLY for e in tool_calls)
    assert "12,500" in answer


def test_run_query_is_read_only() -> None:
    # run_query answers SELECTs but refuses anything that could mutate the data.
    ok = run_sql("SELECT COUNT(*) AS n FROM sales")
    assert ok["rows"][0]["n"] == 6
    mutations = ("DROP TABLE sales", "DELETE FROM sales", "UPDATE sales SET units = 0")
    for mutation in mutations:
        assert "error" in run_sql(mutation)


def test_schema_lists_the_table() -> None:
    described = describe_schema()
    assert described["table"] == "sales"
    names = [column["name"] for column in described["columns"]]
    assert names == ["id", "region", "product", "units", "revenue", "quarter"]
