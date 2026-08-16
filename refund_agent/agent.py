"""A refund agent (Anthropic SDK): look up an order, check the refund policy, and
issue a refund only when the order is eligible.

It knows nothing about reenact. It takes any object exposing
``messages.create(**kwargs)`` - a live ``anthropic.Anthropic`` client when recording,
a scripted fake in tests - plus an optional ``on_tool`` hook reenact uses to record
each tool call. The tools are pure functions over local fixtures, so a recorded run
replays offline with nothing re-executed (and the refund is never re-fired).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from refund_agent.fixtures import REFUND_POLICY, eligibility, lookup_order

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 512
MAX_TURNS = 6

SYSTEM_PROMPT = (
    "You are a refund assistant for an online store. For each request: look up the "
    "order with get_order, check eligibility with check_policy, and issue a refund "
    "with issue_refund ONLY if the order is eligible. If it is not eligible, do not "
    "issue a refund - politely decline and explain why, citing the policy."
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_order",
        "description": "Look up an order by its id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "check_policy",
        "description": "Check whether an order is eligible for a refund.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund. Call this only for an eligible order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["order_id", "amount"],
        },
    },
]

# Tool -> side-effect class, as plain strings (reenact maps these onto its SideEffect
# enum at record time). issue_refund is the mutating action that must never re-fire.
TOOL_SIDE_EFFECTS: dict[str, str] = {
    "get_order": "read_only",
    "check_policy": "read_only",
    "issue_refund": "mutating",
}

# on_tool(name, arguments, result) - called after each tool runs, for the recorder.
ToolHook = Callable[[str, dict[str, Any], Any], None]


def get_order(order_id: str) -> dict[str, Any]:
    """Look up an order (read-only)."""
    order = lookup_order(order_id)
    if order is None:
        return {"error": f"no order found with id {order_id!r}"}
    return order


def check_policy(order_id: str) -> dict[str, Any]:
    """Report whether an order is refundable, with the reason and the policy text."""
    order = lookup_order(order_id)
    if order is None:
        return {"error": f"no order found with id {order_id!r}"}
    eligible, reason = eligibility(order)
    return {
        "order_id": order_id,
        "eligible": eligible,
        "reason": reason,
        "policy": REFUND_POLICY,
    }


def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
    """Issue a refund (the mutating action). A stand-in for moving real money."""
    return {"refunded": True, "order_id": order_id, "amount": amount}


def _execute(name: str, arguments: dict[str, Any]) -> Any:
    if name == "get_order":
        return get_order(str(arguments.get("order_id", "")))
    if name == "check_policy":
        return check_policy(str(arguments.get("order_id", "")))
    if name == "issue_refund":
        return issue_refund(
            str(arguments.get("order_id", "")), float(arguments.get("amount", 0.0))
        )
    return {"error": f"unknown tool {name!r}"}


def _text(content: list[Any]) -> str:
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def run_refund_agent(client: Any, request: str, on_tool: ToolHook | None = None) -> str:
    """Handle a refund request and return the agent's final reply text.

    Runs the Anthropic tool-use loop over ``client``: think -> call a tool -> think,
    until the model answers without a tool call. Each tool call is executed against
    the local fixtures and reported to ``on_tool`` (if given) for recording.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": request}]
    content: list[Any] = []
    for _ in range(MAX_TURNS):
        body: dict[str, Any] = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ).model_dump(mode="json")
        content = body.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_uses:
            return _text(content)
        results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            name = str(tool_use.get("name", ""))
            arguments: dict[str, Any] = tool_use.get("input", {}) or {}
            result = _execute(name, arguments)
            if on_tool is not None:
                on_tool(name, arguments, result)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.get("id"),
                    "content": json.dumps(result),
                }
            )
        messages.append({"role": "user", "content": results})
    return _text(content)
