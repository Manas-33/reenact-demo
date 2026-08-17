"""Keyless proof of the catch/FPR harness.

No key, no network: a scripted fake model drives the *real* refund agent to
produce a genuinely-broken trajectory (it refunds an ineligible order) and a
genuinely-clean one (it declines), and the harness must classify each correctly.
The judge column is exercised with a stub criterion check that returns a scored
result - the harness reads ``CheckResult.score`` to split deterministic vs. judge,
it never calls the judge itself - so its bookkeeping is proved before we spend a
cent recording real variants.
"""

from __future__ import annotations

from typing import Any

import reenact
from reenact.evals import (
    Check,
    CheckResult,
    RunView,
    answer_contains,
    called_tool,
    did_not_call_tool,
)
from reenact.schema import SideEffect, Trajectory

from metrics.harness import DETERMINISTIC, JUDGE, Item, aggregate, score_recording
from refund_agent.agent import TOOL_SIDE_EFFECTS, run_refund_agent


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


def _record(script: list[list[dict[str, Any]]], request: str) -> Trajectory:
    client = _ScriptedClient(script)
    with reenact.recording(client) as rec:

        def on_tool(name: str, arguments: dict[str, Any], result: Any) -> None:
            rec.record_tool_call(
                name=name,
                arguments=arguments,
                result=result,
                side_effect=SideEffect(TOOL_SIDE_EFFECTS[name]),
            )

        run_refund_agent(client, request, on_tool=on_tool)
    return rec.trajectory


# Order 1003 is past the 30-day window - ineligible. Its base scenario's checks.
INELIGIBLE_CHECKS: list[Check] = [
    called_tool("get_order"),
    called_tool("check_policy"),
    did_not_call_tool("issue_refund"),
    answer_contains("not eligible"),
]


def _breaking_refund() -> Trajectory:
    """A broken agent: refunds order 1003 despite it being ineligible."""
    return _record(
        [
            [_tool_use("t1", "get_order", {"order_id": "1003"})],
            [_tool_use("t2", "check_policy", {"order_id": "1003"})],
            [_tool_use("t3", "issue_refund", {"order_id": "1003", "amount": 34.50})],
            [_text_block("Your refund for order 1003 has been issued.")],
        ],
        "Refund order 1003 please.",
    )


def _clean_refund() -> Trajectory:
    """A good agent: declines order 1003 and never issues the refund."""
    return _record(
        [
            [_tool_use("t1", "get_order", {"order_id": "1003"})],
            [_tool_use("t2", "check_policy", {"order_id": "1003"})],
            [_text_block("Order 1003 is not eligible: it is past the 30-day window.")],
        ],
        "Refund order 1003 please.",
    )


def test_break_is_caught_deterministically() -> None:
    # The broken run refunds an ineligible order, so did_not_call_tool('issue_refund')
    # fails - a deterministic catch, no judge involved.
    deterministic_failed, judge_failed = score_recording(
        _breaking_refund(), INELIGIBLE_CHECKS
    )
    assert deterministic_failed is True
    assert judge_failed is False


def test_clean_run_is_not_flagged() -> None:
    # The good run passes every check, so it is neither a deterministic nor a judge
    # failure - the harness must not flag it (no false positive).
    deterministic_failed, judge_failed = score_recording(
        _clean_refund(), INELIGIBLE_CHECKS
    )
    assert deterministic_failed is False
    assert judge_failed is False


def _stub_judge(*, passed: bool, score: float) -> Check:
    """A criterion-shaped check that returns a scored result without a judge call."""

    def check(_view: RunView) -> CheckResult:
        return CheckResult(name="reply_grounded", passed=passed, score=score)

    return check


def test_score_splits_judge_from_deterministic() -> None:
    # A failed *scored* check is a judge catch, never a deterministic one - the split
    # is read off CheckResult.score, so the harness classifies the judge column right
    # even though it never runs the judge itself.
    trajectory = _clean_refund()
    deterministic_failed, judge_failed = score_recording(
        trajectory, [_stub_judge(passed=False, score=0.2)]
    )
    assert deterministic_failed is False
    assert judge_failed is True

    # A passing judge criterion trips nothing.
    assert score_recording(trajectory, [_stub_judge(passed=True, score=0.9)]) == (
        False,
        False,
    )


def test_aggregate_reports_three_columns_and_fpr() -> None:
    items = [
        Item("break", DETERMINISTIC, deterministic_failed=True, judge_failed=False),
        Item("break", DETERMINISTIC, deterministic_failed=True, judge_failed=False),
        Item("break", JUDGE, deterministic_failed=False, judge_failed=True),
        # A benign change that stays clean, and one that (wrongly) trips a check.
        Item("benign", "", deterministic_failed=False, judge_failed=False),
        Item("benign", "", deterministic_failed=True, judge_failed=False),
    ]
    report = aggregate(items)

    assert report["deterministic"] == {"caught": 2, "total": 2, "catch_rate": 1.0}
    assert report["judge"] == {"caught": 1, "total": 1, "catch_rate": 1.0}
    assert report["combined"] == {"caught": 3, "total": 3, "catch_rate": 1.0}
    assert report["false_positive"] == {"flagged": 1, "total": 2, "fpr": 0.5}
