"""Keyless tests for the support agent: a scripted fake model drives the real graph.

Proves the LangGraph RAG pipeline offline (no key): the graph runs, the callback
handler records LLM / tool / node events, side-effects are labeled, and the reply is
grounded in what `search_docs` returned - the same record-real / test-keyless split
reenact uses. (langgraph is a hard dependency of this repo, so no importorskip.)
"""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from reenact.evals import (
    RunView,
    Scenario,
    answer_contains,
    called_tool,
    no_mutating_tool_reexecuted,
    run_scenario,
)
from reenact.schema import SideEffect, ToolCallEvent

from support_agent.agent import support_trajectory
from support_agent.fixtures import ticket_text


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{name}"}


def _scripted_model() -> Any:
    """A fake model scripted to search, label, reply (grounded), then summarize."""
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        _call("search_docs", query="password reset link error")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[_call("label_ticket", ticket_id="42", label="account")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        _call(
                            "post_reply",
                            ticket_id="42",
                            body="Reset links expire in an hour; request a fresh one.",
                        )
                    ],
                ),
                AIMessage(
                    content=(
                        "I labeled ticket #42 as account and replied that the reset "
                        "link expires in an hour; reset from Settings > Security."
                    )
                ),
            ]
        )
    )


def test_records_all_surfaces() -> None:
    trajectory = support_trajectory(
        _scripted_model(), ticket_text("42"), name="support-42"
    )
    kinds = {event.type for event in trajectory.events}
    assert {"llm_call", "tool_call", "graph_node"} <= kinds
    tool_names = [e.name for e in trajectory.events if isinstance(e, ToolCallEvent)]
    assert tool_names == ["search_docs", "label_ticket", "post_reply"]


def test_side_effects_labeled_by_name() -> None:
    trajectory = support_trajectory(_scripted_model(), ticket_text("42"))
    effects = {
        e.name: e.side_effect for e in trajectory.events if isinstance(e, ToolCallEvent)
    }
    assert effects["search_docs"] is SideEffect.READ_ONLY
    assert effects["label_ticket"] is SideEffect.MUTATING
    assert effects["post_reply"] is SideEffect.MUTATING


def test_reply_is_grounded_in_the_retrieved_article() -> None:
    trajectory = support_trajectory(_scripted_model(), ticket_text("42"))
    assert "Settings > Security" in RunView(trajectory).final_answer


def test_eval_checks_pass_on_the_recording() -> None:
    trajectory = support_trajectory(
        _scripted_model(), ticket_text("42"), name="support-42"
    )
    scenario = Scenario(
        name="support-42",
        trajectory=trajectory,
        checks=[
            called_tool("search_docs"),
            called_tool("label_ticket"),
            called_tool("post_reply"),
            answer_contains("Settings"),
            no_mutating_tool_reexecuted(),
        ],
    )
    result = run_scenario(scenario)
    assert result.passed, result.failures
