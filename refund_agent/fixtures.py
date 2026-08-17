"""Fixtures for the refund agent: a small order table and the refund policy.

Pure, deterministic stand-ins for a real order database and policy document, so the
agent can be recorded once and replayed offline forever with nothing re-executed.
"""

from __future__ import annotations

from typing import Any

REFUND_POLICY = (
    "Refund policy:\n"
    "1. An order is refundable within 30 days of purchase.\n"
    "2. Final-sale items are not refundable.\n"
    "3. The order must have been delivered.\n"
)

_ORDERS: dict[str, dict[str, Any]] = {
    "1001": {
        "id": "1001",
        "item": "Wireless headphones",
        "amount": 79.99,
        "days_since_purchase": 5,
        "final_sale": False,
        "status": "delivered",
    },
    "1002": {
        "id": "1002",
        "item": "Festival ticket",
        "amount": 120.00,
        "days_since_purchase": 2,
        "final_sale": True,  # final sale -> not refundable
        "status": "delivered",
    },
    "1003": {
        "id": "1003",
        "item": "Desk lamp",
        "amount": 34.50,
        "days_since_purchase": 45,  # past the 30-day window -> not refundable
        "final_sale": False,
        "status": "delivered",
    },
    "1004": {
        "id": "1004",
        "item": "Bluetooth speaker",
        "amount": 45.00,
        "days_since_purchase": 8,
        "final_sale": False,
        "status": "delivered",
    },
    "1005": {
        "id": "1005",
        "item": "Office chair",
        "amount": 210.00,
        "days_since_purchase": 3,
        "final_sale": False,
        "status": "in_transit",  # not delivered yet -> not refundable
    },
    "1006": {
        "id": "1006",
        "item": "Coffee maker",
        "amount": 89.99,
        "days_since_purchase": 20,
        "final_sale": False,
        "status": "delivered",
    },
}


def lookup_order(order_id: str) -> dict[str, Any] | None:
    """Return the order with ``order_id``, or ``None`` if there is no such order."""
    return _ORDERS.get(order_id)


def eligibility(order: dict[str, Any]) -> tuple[bool, str]:
    """Whether ``order`` is refundable under the policy, with a one-line reason."""
    if order["status"] != "delivered":
        return False, f"the order status is '{order['status']}', not delivered"
    if order["final_sale"]:
        return False, "the item was a final-sale purchase"
    if order["days_since_purchase"] > 30:
        days = order["days_since_purchase"]
        return False, f"it has been {days} days since purchase (over the 30-day limit)"
    return True, "it is within the 30-day window and eligible"
