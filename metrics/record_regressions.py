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
from langchain_core.tools import tool
from reenact.schema import SideEffect, Trajectory
from reenact.store import save_cassette

from metrics.variants import Variant, all_variants
from refund_agent.agent import TOOL_SIDE_EFFECTS, TOOLS, run_refund_agent
from support_agent.agent import TOOLS as SUPPORT_TOOLS
from support_agent.agent import support_trajectory

CORPUS = Path(__file__).resolve().parent / "corpus"
IMPLEMENTED = ("refund", "support")


# --- Refund (Anthropic SDK) -------------------------------------------------

def _renamed_tools(rename: tuple[str, str] | None) -> list[dict[str, Any]] | None:
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


# --- Support (LangGraph RAG) ------------------------------------------------

@tool
def send_message(ticket_id: str, body: str) -> str:
    """Post a public reply on a ticket (mutating)."""
    return f"posted reply to ticket #{ticket_id}"


_RENAMED_SUPPORT_TOOLS = {"send_message": send_message}


def _support_tools(variant: Variant) -> list[Any]:
    """The support tool set with one tool renamed - or the shipped set."""
    if variant.rename is None:
        return SUPPORT_TOOLS
    old, new = variant.rename
    replacement = _RENAMED_SUPPORT_TOOLS[new]
    return [replacement if t.name == old else t for t in SUPPORT_TOOLS]


def record_support_variant(
    model: Any, variant: Variant, tools: list[Any]
) -> Trajectory:
    """Record one support variant over ``model`` (already bound to ``tools``)."""
    return support_trajectory(
        model, variant.scenario_input, system=variant.system, tools=tools
    )


# --- Recording ---------------------------------------------------------------

def _save(variant: Variant, trajectory: Trajectory) -> None:
    trajectory.name = f"{variant.agent}-{variant.name}"
    destination = CORPUS / variant.agent / variant.kind / f"{variant.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_cassette(trajectory, destination)
    print(f"  {destination.relative_to(CORPUS.parent)}")


def _record_refund(variants: list[Variant]) -> None:
    import anthropic

    client = anthropic.Anthropic()
    for variant in variants:
        _save(variant, record_refund_variant(client, variant))


def _record_support(variants: list[Variant]) -> None:
    from langchain_anthropic import ChatAnthropic

    for variant in variants:
        tools = _support_tools(variant)
        model = ChatAnthropic(
            model="claude-sonnet-4-5", temperature=0
        ).bind_tools(tools)
        _save(variant, record_support_variant(model, variant, tools))


BATCH = {"refund": _record_refund, "support": _record_support}


def main() -> None:
    variants = [v for v in all_variants() if v.agent in IMPLEMENTED]
    agents = {v.agent for v in variants}
    if agents & {"refund", "support"} and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set - needed to record.")
    if "analyst" in agents and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set - needed to record the analyst.")

    for agent in sorted(agents):
        batch = [v for v in variants if v.agent == agent]
        print(f"{agent}: recording {len(batch)} variants")
        BATCH[agent](batch)
    print(f"wrote {len(variants)} recordings to {CORPUS}")


if __name__ == "__main__":
    main()
