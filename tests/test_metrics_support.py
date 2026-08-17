"""Keyless proof of the support recording setup.

No key, no network: a scripted fake model drives the real LangGraph graph. The
variant registry is well-formed, the tool-rename swaps post_reply for send_message,
and a scripted rename break records send_message (never post_reply) so the
deterministic checks catch it. The invent-feature breaks are judged at measure
time with a key; here we only assert they are registered in the judge column.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from reenact.evals import Check, answer_contains, called_tool
from reenact.schema import ToolCallEvent

from metrics.harness import DETERMINISTIC, JUDGE, score_recording
from metrics.record_regressions import (
    _support_tools,
    record_support_variant,
    send_message,
)
from metrics.variants import support_variants

_VALID_BASES = {"support-42", "support-57", "support-63"}


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{name}"}


def _renaming_model() -> Any:
    """A fake model that searches, labels, then calls the *renamed* reply tool."""
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="", tool_calls=[_call("search_docs", query="password")]
                ),
                AIMessage(
                    content="",
                    tool_calls=[_call("label_ticket", ticket_id="42", label="account")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        _call("send_message", ticket_id="42", body="See Settings.")
                    ],
                ),
                AIMessage(content="I labeled #42 and sent the password reset steps."),
            ]
        )
    )


def test_support_registry_is_well_formed() -> None:
    variants = support_variants()
    breaks = [v for v in variants if v.kind == "break"]
    benign = [v for v in variants if v.kind == "benign"]
    assert len(breaks) == 7  # 4 deterministic + 3 judge
    assert len(benign) == 30

    names = [v.name for v in variants]
    assert len(names) == len(set(names))
    for variant in variants:
        assert variant.agent == "support"
        assert variant.base_scenario in _VALID_BASES

    deterministic = [b for b in breaks if b.column == DETERMINISTIC]
    judged = [b for b in breaks if b.column == JUDGE]
    assert len(deterministic) == 4
    assert len(judged) == 3
    assert all(b.name.startswith("invent-feature") for b in judged)


def test_support_tools_renames_post_reply() -> None:
    variant = next(v for v in support_variants() if v.name == "rename-post-reply")
    tools = _support_tools(variant)
    names = [t.name for t in tools]
    assert "post_reply" not in names
    assert "send_message" in names
    assert "search_docs" in names and "label_ticket" in names
    # A variant with no rename keeps the shipped tool set.
    plain = next(v for v in support_variants() if v.name == "skip-label")
    assert [t.name for t in _support_tools(plain)] == [t.name for t in tools[:2]] + [
        "label_ticket",
        "post_reply",
    ]


def test_rename_break_records_send_message_and_is_caught() -> None:
    variant = next(v for v in support_variants() if v.name == "rename-post-reply")
    trajectory = record_support_variant(
        _renaming_model(), variant, _support_tools(variant)
    )
    tool_names = [e.name for e in trajectory.events if isinstance(e, ToolCallEvent)]
    assert "send_message" in tool_names
    assert "post_reply" not in tool_names

    deterministic_checks: list[Check] = [
        called_tool("search_docs"),
        called_tool("label_ticket"),
        called_tool("post_reply"),
        answer_contains("password"),
    ]
    deterministic_failed, judge_failed = score_recording(
        trajectory, deterministic_checks
    )
    assert deterministic_failed is True  # post_reply never called by name
    assert judge_failed is False


def test_send_message_tool_is_mutating_shaped() -> None:
    # The renamed reply tool stands in for post_reply, so it must still "post".
    assert "posted reply" in send_message.invoke({"ticket_id": "42", "body": "hi"})
