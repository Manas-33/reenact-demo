"""Record analyst scenarios against the live OpenAI API into cassettes.

Run once with a real key to (re)generate the committed recordings under
``analyst_agent/scenarios/``. The reenact gate replays them offline forever
after - no key, no network, and no query re-executed.

    export OPENAI_API_KEY=sk-...
    uv run --extra record python -m analyst_agent.record
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import reenact
from reenact.schema import SideEffect, Trajectory
from reenact.store import save_cassette

from analyst_agent.agent import TOOL_SIDE_EFFECTS, run_analyst_agent

SCENARIOS = Path(__file__).resolve().parent / "scenarios"

# Three questions, each answerable from the sales table with one distinctive
# figure (12,500 / 180 / 15,400). The break-me PR (a model swap or a prompt edit)
# makes the agent state a number the query never returned - the faithfulness
# criterion catches that where an assertion on a keyword cannot.
CASES: list[tuple[str, str]] = [
    ("west-revenue", "What was our total revenue in the West region?"),
    ("top-product", "Which product sold the most units, and how many units was that?"),
    ("q4-revenue", "What was our total revenue in Q4?"),
]


def record_case(client: Any, question: str) -> Trajectory:
    """Run one question through ``client``, recording its LLM and tool calls."""
    with reenact.recording(client) as rec:

        def on_tool(name: str, arguments: dict[str, Any], result: Any) -> None:
            rec.record_tool_call(
                name=name,
                arguments=arguments,
                result=result,
                side_effect=SideEffect(TOOL_SIDE_EFFECTS.get(name, "unknown")),
            )

        run_analyst_agent(client, question, on_tool=on_tool)
    return rec.trajectory


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set - needed to record.")
    import openai

    client = openai.OpenAI()
    SCENARIOS.mkdir(parents=True, exist_ok=True)
    for name, question in CASES:
        trajectory = record_case(client, question)
        trajectory.name = f"analyst-{name}"
        destination = SCENARIOS / f"{name}.json"
        save_cassette(trajectory, destination)
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
