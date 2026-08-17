"""Keyless proof of the analyst recording setup.

No key, no network: a scripted fake OpenAI client drives the real analyst loop.
The variant registry is well-formed, the tool-rename swaps run_query for
execute_sql, and a scripted rename break records execute_sql (never run_query) so
the deterministic checks catch it. The faithfulness breaks (hallucinate /
cite-unfetched) are judged at measure time with a key.
"""

from __future__ import annotations

import json
from typing import Any

from reenact.evals import Check, answer_contains, called_tool
from reenact.schema import ToolCallEvent

from metrics.harness import DETERMINISTIC, JUDGE, score_recording
from metrics.record_regressions import _renamed_analyst_tools, record_analyst_variant
from metrics.variants import analyst_variants

_VALID_BASES = {"analyst-west-revenue", "analyst-top-product", "analyst-q4-revenue"}


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _tool_turn(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def _text_turn(text: str) -> dict[str, Any]:
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


class _ScriptedClient:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.chat = type("_Chat", (), {"completions": _Completions(script)})()


def test_analyst_registry_is_well_formed() -> None:
    variants = analyst_variants()
    breaks = [v for v in variants if v.kind == "break"]
    benign = [v for v in variants if v.kind == "benign"]
    assert len(breaks) == 7  # 4 deterministic + 3 judge
    assert len(benign) == 30

    names = [v.name for v in variants]
    assert len(names) == len(set(names))
    for variant in variants:
        assert variant.agent == "analyst"
        assert variant.base_scenario in _VALID_BASES

    deterministic = [b for b in breaks if b.column == DETERMINISTIC]
    judged = [b for b in breaks if b.column == JUDGE]
    assert len(deterministic) == 4
    assert len(judged) == 3
    assert all(b.name.startswith("cite-unfetched") for b in judged)


def test_renamed_analyst_tools_swaps_run_query() -> None:
    renamed = _renamed_analyst_tools(("run_query", "execute_sql"))
    assert renamed is not None
    names = [entry["function"]["name"] for entry in renamed]
    assert "run_query" not in names
    assert "execute_sql" in names
    assert "get_schema" in names
    assert _renamed_analyst_tools(None) is None


def test_rename_break_records_execute_sql_and_is_caught() -> None:
    variant = next(v for v in analyst_variants() if v.name == "rename-run-query")
    script = [
        _tool_turn(_tool_call("c1", "get_schema", {})),
        _tool_turn(
            _tool_call("c2", "execute_sql", {"sql": "SELECT SUM(revenue) FROM sales"})
        ),
        _text_turn("Total Q4 revenue was $15,400."),
    ]
    trajectory = record_analyst_variant(_ScriptedClient(script), variant)

    tool_names = [e.name for e in trajectory.events if isinstance(e, ToolCallEvent)]
    assert "execute_sql" in tool_names
    assert "run_query" not in tool_names

    deterministic_checks: list[Check] = [
        called_tool("get_schema"),
        called_tool("run_query"),
        answer_contains("15,400"),
    ]
    deterministic_failed, judge_failed = score_recording(
        trajectory, deterministic_checks
    )
    assert deterministic_failed is True  # run_query never called by name
    assert judge_failed is False
