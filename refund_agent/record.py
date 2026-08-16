"""Record refund scenarios against the live Anthropic API into cassettes.

Run once with a real key to (re)generate the committed recordings under
``refund_agent/scenarios/``. The reenact gate replays them offline forever after -
no key, no network, and the refund is never re-fired.

    uv sync --extra record
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python refund_agent/record.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import reenact
from reenact.schema import SideEffect, Trajectory
from reenact.store import save_cassette

from refund_agent.agent import TOOL_SIDE_EFFECTS, run_refund_agent

SCENARIOS = Path(__file__).resolve().parent / "scenarios"

# A small, representative set: an eligible refund, and two declines (final sale and
# past the 30-day window). The eligible/ineligible split is what the break-me PR
# exploits - a prompt that drops the eligibility gate refunds the ineligible ones.
CASES: list[tuple[str, str]] = [
    ("eligible", "Please refund order 1001, the headphones arrived damaged."),
    ("final-sale", "I'd like a refund for order 1002, the festival ticket."),
    ("past-window", "Refund order 1003 please, the desk lamp stopped working."),
]


def record_case(client: Any, request: str) -> Trajectory:
    """Run one request through ``client``, recording its LLM and tool calls."""
    with reenact.recording(client) as rec:

        def on_tool(name: str, arguments: dict[str, Any], result: Any) -> None:
            rec.record_tool_call(
                name=name,
                arguments=arguments,
                result=result,
                side_effect=SideEffect(TOOL_SIDE_EFFECTS.get(name, "unknown")),
            )

        run_refund_agent(client, request, on_tool=on_tool)
    return rec.trajectory


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set - needed to record.")
    import anthropic

    client = anthropic.Anthropic()
    SCENARIOS.mkdir(parents=True, exist_ok=True)
    for name, request in CASES:
        trajectory = record_case(client, request)
        trajectory.name = f"refund-{name}"
        destination = SCENARIOS / f"{name}.json"
        save_cassette(trajectory, destination)
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
