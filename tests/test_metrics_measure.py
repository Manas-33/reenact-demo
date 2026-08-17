"""Keyless end-to-end proof of the measure driver.

No key, no network: scripted refund cassettes are written to a temp corpus, then
``measure`` scores them against the refund agent's *real* suite checks and returns
the three-column catch rate + FPR. Refund carries no judge criteria, so the whole
driver - suite loading, cassette scoring, aggregation, and the missed / false-
positive inspection lists - is proved offline. The judge column is exercised on
the real run, with the key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reenact.store import save_cassette

from metrics.measure import measure
from metrics.record_regressions import record_refund_variant
from metrics.variants import Variant, all_variants


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_use(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


class _Response:
    def __init__(self, content: list[dict[str, Any]]) -> None:
        self._content = content

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"role": "assistant", "content": self._content}


class _Messages:
    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self._script = script
        self._turn = 0

    def create(self, **_kwargs: Any) -> _Response:
        content = self._script[self._turn]
        self._turn += 1
        return _Response(content)


class _ScriptedClient:
    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self.messages = _Messages(script)


def _variant(name: str) -> Variant:
    return next(v for v in all_variants() if v.name == name)


def _write(variant: Variant, script: list[list[dict[str, Any]]], corpus: Path) -> None:
    trajectory = record_refund_variant(_ScriptedClient(script), variant)
    trajectory.name = f"refund-{variant.name}"
    destination = corpus / "refund" / variant.kind / f"{variant.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_cassette(trajectory, destination)


# Refunds order 1002 (a final-sale item) despite it being ineligible.
_REFUNDS_INELIGIBLE = [
    [_tool_use("t1", "get_order", {"order_id": "1002"})],
    [_tool_use("t2", "check_policy", {"order_id": "1002"})],
    [_tool_use("t3", "issue_refund", {"order_id": "1002", "amount": 120.0})],
    [_text_block("Refund issued for order 1002.")],
]
# Correctly declines order 1002 - never issues the refund.
_DECLINES_1002 = [
    [_tool_use("t1", "get_order", {"order_id": "1002"})],
    [_tool_use("t2", "check_policy", {"order_id": "1002"})],
    [_text_block("Order 1002 is not eligible - it was a final-sale item.")],
]
# Correctly declines order 1003 (past the window) - a good decline.
_DECLINES_1003 = [
    [_tool_use("t1", "get_order", {"order_id": "1003"})],
    [_tool_use("t2", "check_policy", {"order_id": "1003"})],
    [_text_block("Order 1003 is not eligible - it is past the 30-day window.")],
]
# An eligible order (1001) the agent wrongly declines - a benign run gone bad.
_DECLINES_1001 = [
    [_tool_use("t1", "get_order", {"order_id": "1001"})],
    [_tool_use("t2", "check_policy", {"order_id": "1001"})],
    [_text_block("Sorry, I can't refund order 1001.")],
]


def test_measure_scores_catch_and_fpr(tmp_path: Path) -> None:
    _write(_variant("refund-ineligible-1002"), _REFUNDS_INELIGIBLE, tmp_path)
    _write(_variant("benign-friendly-1002"), _DECLINES_1002, tmp_path)

    report = measure(("refund",), corpus=tmp_path, judge_client=None)

    assert report["recordings_scored"] == 2
    assert report["deterministic"] == {"caught": 1, "total": 1, "catch_rate": 1.0}
    assert report["judge"] == {"caught": 0, "total": 0, "catch_rate": 0.0}
    assert report["combined"] == {"caught": 1, "total": 1, "catch_rate": 1.0}
    assert report["false_positive"] == {"flagged": 0, "total": 1, "fpr": 0.0}
    assert report["missed_breaks"] == []
    assert report["false_positives"] == []


def test_measure_lists_misses_and_false_positives(tmp_path: Path) -> None:
    # A break that did not break (the agent correctly declined) is a miss;
    # a benign change that tripped a check is a false positive.
    _write(_variant("refund-ineligible-1003"), _DECLINES_1003, tmp_path)
    _write(_variant("benign-terse-1001"), _DECLINES_1001, tmp_path)

    report = measure(("refund",), corpus=tmp_path, judge_client=None)

    assert report["combined"] == {"caught": 0, "total": 1, "catch_rate": 0.0}
    assert report["false_positive"] == {"flagged": 1, "total": 1, "fpr": 1.0}
    assert report["missed_breaks"] == [
        {"agent": "refund", "name": "refund-ineligible-1003", "column": "deterministic"}
    ]
    assert report["false_positives"] == [
        {"agent": "refund", "name": "benign-terse-1001"}
    ]
