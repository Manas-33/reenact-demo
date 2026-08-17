"""Keyless proof of the refund recording setup.

No key, no network: the variant registry is well-formed, the tool-rename builder
swaps only the named tool, and a scripted rename break is recorded and caught by
the base scenario's checks. Whether the *real* model breaks under each degraded
prompt is confirmed at record time; this proves the machinery around it.
"""

from __future__ import annotations

from typing import Any

from reenact.evals import Check, answer_contains, called_tool
from reenact.schema import ToolCallEvent

from metrics.harness import DETERMINISTIC, score_recording
from metrics.record_regressions import _renamed_tools, record_refund_variant
from metrics.variants import refund_variants

_VALID_BASES = {"refund-eligible", "refund-final-sale", "refund-past-window"}


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_use(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


class _Response:
    def __init__(self, content: list[dict[str, Any]]) -> None:
        self._content = content

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"role": "assistant", "content": self._content}


class _Messages:
    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self._script = script
        self._turn = 0

    def create(self, **_kwargs: Any) -> _Response:
        content = self._script[self._turn]
        self._turn += 1
        return _Response(content)


class _ScriptedClient:
    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self.messages = _Messages(script)


def test_refund_registry_is_well_formed() -> None:
    variants = refund_variants()
    breaks = [v for v in variants if v.kind == "break"]
    benign = [v for v in variants if v.kind == "benign"]
    assert len(breaks) == 6
    assert len(benign) == 30

    names = [v.name for v in variants]
    assert len(names) == len(set(names))  # unique names -> distinct cassette files
    for variant in variants:
        assert variant.agent == "refund"
        assert variant.base_scenario in _VALID_BASES
    for seeded in breaks:
        assert seeded.column == DETERMINISTIC  # refund has no judge criteria
    for clean in benign:
        assert clean.column == ""
        assert clean.system is not None  # a benign reword always overrides the prompt


def test_renamed_tools_swaps_only_the_named_tool() -> None:
    tools = _renamed_tools(("issue_refund", "process_refund"))
    assert tools is not None
    names = [tool["name"] for tool in tools]
    assert "issue_refund" not in names
    assert "process_refund" in names
    assert "get_order" in names and "check_policy" in names
    assert _renamed_tools(None) is None


def test_rename_break_records_and_is_caught() -> None:
    # A scripted model calls the renamed tool; the recording must show
    # 'process_refund', never 'issue_refund', so called_tool('issue_refund') fails.
    script = [
        [_tool_use("t1", "get_order", {"order_id": "1001"})],
        [_tool_use("t2", "check_policy", {"order_id": "1001"})],
        [_tool_use("t3", "process_refund", {"order_id": "1001", "amount": 79.99})],
        [_text_block("All set - the refund is on its way.")],
    ]
    variant = next(v for v in refund_variants() if v.name == "rename-issue-refund")
    trajectory = record_refund_variant(_ScriptedClient(script), variant)

    tool_names = [e.name for e in trajectory.events if isinstance(e, ToolCallEvent)]
    assert "process_refund" in tool_names
    assert "issue_refund" not in tool_names

    eligible_checks: list[Check] = [
        called_tool("get_order"),
        called_tool("check_policy"),
        called_tool("issue_refund"),
        answer_contains("refund"),
    ]
    deterministic_failed, judge_failed = score_recording(trajectory, eligible_checks)
    assert deterministic_failed is True  # issue_refund was never called by name
    assert judge_failed is False
