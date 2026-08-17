"""The catch-rate / false-positive-rate harness.

Each recorded variant is scored against its base scenario's checks. A BREAK is
caught when at least one check fails; a BENIGN change is a false positive when any
check fails. Results are split three ways by reading each ``CheckResult.score``:
an assertion leaves it ``None`` (a *deterministic* signal), the LLM judge fills it
(a *judge* signal). So a failed check with no score is a deterministic catch, and
a failed check with a score is a judge catch.

``score_recording`` and ``aggregate`` are pure over their inputs, so the whole
tally is provable offline with scripted trajectories - no key, no network - before
a single real recording is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reenact.evals import Check, RunView
from reenact.schema import Trajectory

# A break declares which column is meant to catch it; benign changes carry no column.
DETERMINISTIC = "deterministic"
JUDGE = "judge"


def score_recording(trajectory: Trajectory, checks: list[Check]) -> tuple[bool, bool]:
    """Run ``checks`` against ``trajectory``; return ``(det_failed, judge_failed)``.

    A failed assertion (``score is None``) sets the deterministic flag; a failed
    judge criterion (``score`` filled) sets the judge flag. A recording can trip
    both - a hallucinated figure fails an ``answer_contains`` assertion *and* a
    faithfulness criterion.
    """
    view = RunView(trajectory)
    deterministic_failed = False
    judge_failed = False
    for check in checks:
        result = check(view)
        if result.passed:
            continue
        if result.score is None:
            deterministic_failed = True
        else:
            judge_failed = True
    return deterministic_failed, judge_failed


@dataclass(frozen=True)
class Item:
    """One scored recording: what it was meant to be, and what the checks found."""

    kind: str  # "break" | "benign"
    column: str  # DETERMINISTIC | JUDGE for a break; "" for a benign change
    deterministic_failed: bool
    judge_failed: bool
    agent: str = ""
    name: str = ""

    @property
    def any_failed(self) -> bool:
        return self.deterministic_failed or self.judge_failed


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _column(items: list[Item], column: str) -> dict[str, Any]:
    """Catch tally for the break variants meant to be caught by ``column``."""
    variants = [it for it in items if it.kind == "break" and it.column == column]
    caught = sum(
        1
        for it in variants
        if (it.deterministic_failed if column == DETERMINISTIC else it.judge_failed)
    )
    total = len(variants)
    return {"caught": caught, "total": total, "catch_rate": _rate(caught, total)}


def aggregate(items: list[Item]) -> dict[str, Any]:
    """Three-column catch rate plus the false-positive rate over the benign items."""
    breaks = [it for it in items if it.kind == "break"]
    benign = [it for it in items if it.kind == "benign"]
    combined_caught = sum(1 for it in breaks if it.any_failed)
    flagged = sum(1 for it in benign if it.any_failed)
    return {
        "deterministic": _column(items, DETERMINISTIC),
        "judge": _column(items, JUDGE),
        "combined": {
            "caught": combined_caught,
            "total": len(breaks),
            "catch_rate": _rate(combined_caught, len(breaks)),
        },
        "false_positive": {
            "flagged": flagged,
            "total": len(benign),
            "fpr": _rate(flagged, len(benign)),
        },
    }
