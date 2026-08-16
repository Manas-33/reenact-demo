"""Keyless tests for the refund agent.

A scripted fake model drives the *real* agent offline - no key, no network - the
same record-real / test-keyless split reenact itself uses. These prove the agent's
shape and the invariant the break-me PR will later violate: a good agent never
refunds an ineligible order.
"""

from __future__ import annotations

from typing import Any

import reenact
from reenact.schema import SideEffect, ToolCallEvent, Trajectory

from refund_agent.agent import (
    TOOL_SIDE_EFFECTS,
    check_policy,
    get_order,
    run_refund_agent,
)


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
    """A fake Anthropic client that returns pre-scripted turns in order."""

    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self.messages = _Messages(script)


def _record(script: list[list[dict[str, Any]]], request: str) -> tuple[Trajectory, str]:
    client = _ScriptedClient(script)
    with reenact.recording(client) as rec:

        def on_tool(name: str, arguments: dict[str, Any], result: Any) -> None:
            rec.record_tool_call(
                name=name,
                arguments=arguments,
                result=result,
                side_effect=SideEffect(TOOL_SIDE_EFFECTS[name]),
            )

        answer = run_refund_agent(client, request, on_tool=on_tool)
    return rec.trajectory, answer


def test_eligible_order_issues_a_refund() -> None:
    script = [
        [_tool_use("t1", "get_order", {"order_id": "1001"})],
        [_tool_use("t2", "check_policy", {"order_id": "1001"})],
        [_tool_use("t3", "issue_refund", {"order_id": "1001", "amount": 79.99})],
        [_text_block("Your refund of $79.99 for order 1001 has been issued.")],
    ]
    trajectory, answer = _record(script, "Refund order 1001, it arrived damaged.")

    kinds = [event.type for event in trajectory.events]
    assert kinds == [
        "llm_call",
        "tool_call",
        "llm_call",
        "tool_call",
        "llm_call",
        "tool_call",
        "llm_call",
    ]
    tool_calls = [e for e in trajectory.events if isinstance(e, ToolCallEvent)]
    assert [e.name for e in tool_calls] == ["get_order", "check_policy", "issue_refund"]
    # The mutating action is recorded as such, so replay never re-fires it.
    assert tool_calls[-1].name == "issue_refund"
    assert tool_calls[-1].side_effect is SideEffect.MUTATING
    assert "refund" in answer.lower()


def test_ineligible_order_never_calls_issue_refund() -> None:
    # The invariant a break-me PR will violate: a good agent declines an ineligible
    # order rather than refunding it. This is what the reenact gate will guard.
    script = [
        [_tool_use("t1", "get_order", {"order_id": "1003"})],
        [_tool_use("t2", "check_policy", {"order_id": "1003"})],
        [
            _text_block(
                "Sorry, order 1003 is not eligible: it is past the 30-day window."
            )
        ],
    ]
    called: list[str] = []
    answer = run_refund_agent(
        _ScriptedClient(script),
        "Refund order 1003 please.",
        on_tool=lambda name, _args, _result: called.append(name),
    )
    assert called == ["get_order", "check_policy"]
    assert "issue_refund" not in called
    assert "not eligible" in answer.lower()


def test_policy_rules_are_deterministic() -> None:
    assert check_policy("1001")["eligible"] is True
    assert check_policy("1002")["eligible"] is False  # final sale
    assert check_policy("1003")["eligible"] is False  # past the 30-day window
    assert "error" in get_order("9999")
