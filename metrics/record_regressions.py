"""Record the regression corpus (break + benign variants) against the live APIs.

Run once, with keys, to (re)generate ``metrics/corpus/``. Refund + support record
on ANTHROPIC_API_KEY; analyst records on OPENAI_API_KEY. Every run here spends
tokens; the catch/FPR harness that reads the recordings never does.

    set -a; source .env; set +a          # both keys, from the gitignored .env
    uv run --extra record python -m metrics.record_regressions

Each recording is saved under ``corpus/<agent>/<break|benign>/<name>.json`` and
committed, redacted and keyless-replayable, so the numbers reproduce offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import reenact
from reenact.schema import SideEffect, Trajectory
from reenact.store import save_cassette

from metrics.variants import Variant, all_variants
from refund_agent.agent import TOOL_SIDE_EFFECTS, TOOLS, run_refund_agent

CORPUS = Path(__file__).resolve().parent / "corpus"


def _renamed_tools(
    rename: tuple[str, str] | None,
) -> list[dict[str, Any]] | None:
    """The refund tool set with one tool renamed - or ``None`` for the shipped set."""
    if rename is None:
        return None
    old, new = rename
    return [{**tool, "name": new} if tool["name"] == old else tool for tool in TOOLS]


def record_refund_variant(client: Any, variant: Variant) -> Trajectory:
    """Record one refund variant, applying its prompt / tool override."""
    tools = _renamed_tools(variant.rename)
    with reenact.recording(client) as rec:

        def on_tool(name: str, arguments: dict[str, Any], result: Any) -> None:
            rec.record_tool_call(
                name=name,
                arguments=arguments,
                result=result,
                side_effect=SideEffect(TOOL_SIDE_EFFECTS.get(name, "unknown")),
            )

        run_refund_agent(
            client,
            variant.scenario_input,
            on_tool=on_tool,
            system=variant.system,
            tools=tools,
        )
    return rec.trajectory


# agent -> (build a live client, record one variant with it). Support + analyst
# are wired in their own rungs; until then their variants are simply skipped.
RECORDERS: dict[str, Any] = {"refund": record_refund_variant}


def _anthropic_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


CLIENTS: dict[str, Any] = {"refund": _anthropic_client}


def main() -> None:
    variants = [v for v in all_variants() if v.agent in RECORDERS]
    agents = {v.agent for v in variants}
    needs_anthropic = "refund" in agents or "support" in agents
    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set - needed to record.")
    if "analyst" in agents and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set - needed to record the analyst.")

    clients = {agent: CLIENTS[agent]() for agent in agents}
    written = 0
    for variant in variants:
        trajectory = RECORDERS[variant.agent](clients[variant.agent], variant)
        trajectory.name = f"{variant.agent}-{variant.name}"
        destination = CORPUS / variant.agent / variant.kind / f"{variant.name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_cassette(trajectory, destination)
        written += 1
        print(f"  {destination.relative_to(CORPUS.parent)}")
    print(f"wrote {written} recordings to {CORPUS}")


if __name__ == "__main__":
    main()
