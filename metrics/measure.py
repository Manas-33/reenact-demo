"""Measure catch rate + false-positive rate over the recorded corpus.

Loads each agent's *real* ``suite.toml`` checks - so it measures the actual gate,
not a re-implementation - scores every recorded variant against its base
scenario's checks, and reports catch rate in three columns (deterministic, judge,
combined) paired with the false-positive rate over the benign changes.

The deterministic column and the FPR run offline; the judge column needs the
Anthropic judge that backs the ``reply_grounded`` / ``answer_grounded`` criteria,
so pass a ``judge_client`` to score it (and to load the support / analyst suites,
which declare criteria). Missed breaks and false positives are listed so a real
run can be inspected: a designed break that did not break, or a benign change
wrongly flagged, is ground truth to relabel - not a harness error.

    set -a; source .env; set +a
    uv run --extra record python -m metrics.measure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reenact.evals import Check, load_suite
from reenact.store import load_cassette

from metrics.harness import Item, aggregate, score_recording
from metrics.variants import all_variants

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(__file__).resolve().parent / "corpus"
RESULTS = Path(__file__).resolve().parent / "results.json"
AGENTS = ("refund", "support", "analyst")


def _suite_path(agent: str) -> Path:
    return REPO_ROOT / f"{agent}_agent" / "evals" / "suite.toml"


def _agent_checks(agent: str, judge_client: Any) -> dict[str, list[Check]]:
    """Map each of the agent's base scenarios to its committed suite checks."""
    scenarios = load_suite(_suite_path(agent), judge_client=judge_client)
    return {scenario.name: scenario.checks for scenario in scenarios}


def measure(
    agents: tuple[str, ...] = AGENTS,
    *,
    corpus: Path = CORPUS,
    judge_client: Any = None,
) -> dict[str, Any]:
    """Score every recorded variant and aggregate the catch / FPR report.

    Variants with no recording on disk yet are skipped, so this runs on a partial
    corpus. ``judge_client`` is required for any agent whose suite declares a
    criterion (support, analyst); ``refund`` scores offline.
    """
    items: list[Item] = []
    missed: list[dict[str, str]] = []
    false_positives: list[dict[str, str]] = []
    for agent in agents:
        checks_by_scenario = _agent_checks(agent, judge_client)
        for variant in (v for v in all_variants() if v.agent == agent):
            path = corpus / agent / variant.kind / f"{variant.name}.json"
            if not path.is_file():
                continue
            checks = checks_by_scenario[variant.base_scenario]
            deterministic_failed, judge_failed = score_recording(
                load_cassette(path), checks
            )
            items.append(
                Item(
                    variant.kind,
                    variant.column,
                    deterministic_failed,
                    judge_failed,
                    agent,
                    variant.name,
                )
            )
            if variant.kind == "break" and not (deterministic_failed or judge_failed):
                missed.append(
                    {"agent": agent, "name": variant.name, "column": variant.column}
                )
            elif variant.kind == "benign" and (deterministic_failed or judge_failed):
                false_positives.append({"agent": agent, "name": variant.name})

    report = aggregate(items)
    report["recordings_scored"] = len(items)
    report["missed_breaks"] = missed
    report["false_positives"] = false_positives
    return report


def main() -> None:
    import anthropic  # the judge is Anthropic; scoring the judge column needs a key

    report = measure(judge_client=anthropic.Anthropic())
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
