"""Record support scenarios against the live Anthropic API into cassettes.

Run once with a real key to (re)generate the committed recordings under
``support_agent/scenarios/``. The reenact gate replays them offline forever after.

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra record python -m support_agent.record
"""

from __future__ import annotations

import os
from pathlib import Path

from reenact.store import save_cassette

from support_agent.agent import TOOLS, support_trajectory
from support_agent.fixtures import ticket_text

SCENARIOS = Path(__file__).resolve().parent / "scenarios"
TICKET_IDS = ["42", "57", "63"]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set - needed to record.")
    from langchain_anthropic import ChatAnthropic

    # temperature=0 for a stable fixture (see the on-ramp notes on recording).
    model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0).bind_tools(TOOLS)
    SCENARIOS.mkdir(parents=True, exist_ok=True)
    for ticket_id in TICKET_IDS:
        trajectory = support_trajectory(
            model, ticket_text(ticket_id), name=f"support-{ticket_id}"
        )
        destination = SCENARIOS / f"ticket-{ticket_id}.json"
        save_cassette(trajectory, destination)
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
