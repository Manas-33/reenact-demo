"""Fixtures for the analyst agent: a small in-memory sales table.

A deterministic stand-in for a real analytics warehouse. The data lives in a
throwaway in-memory SQLite database rebuilt on every call, so a query can never
mutate anything and the same SQL always returns the same rows - which is what
lets a recorded run replay offline forever. The figures are chosen to be
distinctive (West revenue 12,500 / top product 180 units / Q4 revenue 15,400),
so a hallucinated number in an answer is plainly ungrounded.
"""

from __future__ import annotations

import sqlite3
from typing import Any

TABLE = "sales"

_SCHEMA_SQL = (
    "CREATE TABLE sales (\n"
    "  id INTEGER PRIMARY KEY,\n"
    "  region TEXT NOT NULL,\n"
    "  product TEXT NOT NULL,\n"
    "  units INTEGER NOT NULL,\n"
    "  revenue REAL NOT NULL,\n"
    "  quarter TEXT NOT NULL\n"
    ")"
)

# (id, region, product, units, revenue, quarter). Sums are engineered to be
# distinct so each scenario's answer is one unambiguous figure:
#   West revenue  = 5400 + 7100          = 12,500
#   Widget units  = 120 + 60             = 180  (the top product by units)
#   Q4 revenue    = 6800 + 8600          = 15,400
_ROWS: list[tuple[int, str, str, int, float, str]] = [
    (1, "West", "Widget", 120, 5400.00, "Q1"),
    (2, "West", "Gadget", 45, 7100.00, "Q2"),
    (3, "East", "Widget", 60, 4200.00, "Q1"),
    (4, "East", "Gizmo", 30, 9300.00, "Q3"),
    (5, "North", "Gadget", 75, 6800.00, "Q4"),
    (6, "South", "Gizmo", 40, 8600.00, "Q4"),
]


def _connect() -> sqlite3.Connection:
    """Build a fresh in-memory database seeded with the fixture rows."""
    connection = sqlite3.connect(":memory:")
    connection.execute(_SCHEMA_SQL)
    connection.executemany(
        "INSERT INTO sales (id, region, product, units, revenue, quarter) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        _ROWS,
    )
    connection.commit()
    return connection


def describe_schema() -> dict[str, Any]:
    """Return the table's structure: its DDL and typed column list (read-only)."""
    connection = _connect()
    try:
        info = connection.execute(f"PRAGMA table_info({TABLE})").fetchall()
    finally:
        connection.close()
    columns = [{"name": row[1], "type": row[2]} for row in info]
    return {"table": TABLE, "columns": columns, "ddl": _SCHEMA_SQL}


def run_sql(sql: str) -> dict[str, Any]:
    """Run a single read-only ``SELECT`` and return its columns and rows.

    Only one ``SELECT``/``WITH`` statement is allowed; anything else is refused,
    so the tool can never mutate the data. The database is rebuilt per call from
    the fixed rows, so results are fully deterministic.
    """
    statement = sql.strip().rstrip(";").strip()
    leading = statement.split(None, 1)[0].lower() if statement else ""
    if leading not in {"select", "with"}:
        return {"error": "only read-only SELECT queries are allowed"}
    if ";" in statement:
        return {"error": "only a single statement is allowed"}
    connection = _connect()
    try:
        cursor = connection.execute(statement)
        columns = [description[0] for description in cursor.description or []]
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    except sqlite3.Error as error:
        return {"error": f"query failed: {error}"}
    finally:
        connection.close()
    return {"columns": columns, "rows": rows, "row_count": len(rows)}
