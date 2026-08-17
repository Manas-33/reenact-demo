"""The support agent's fixture world: a small help-center KB and a ticket queue.

The tools read and write this fixture, not a live help desk - a demo must never
post a real reply, and substituting the mutating tools on replay is the guarantee
being shown off. Fixed data keeps recordings deterministic.

The articles are also the *ground truth* for the reply: a good answer stays within
what they say. Note what they deliberately rule out (reset is web-only; no paid tier
raises the rate limit) - that is what a model swap that invents features contradicts.
"""

DOCS: dict[str, str] = {
    "password-reset.md": (
        "Password reset: open Settings > Security and click Reset Password. A reset "
        "link is emailed and expires in one hour. If the link errors it has usually "
        "expired - request a fresh one. Password reset is available on the web only."
    ),
    "billing.md": (
        "Billing: invoices are issued monthly. A duplicate charge is refunded within "
        "five business days to the original card. Ask for the invoice id before "
        "escalating a billing dispute."
    ),
    "rate-limits.md": (
        "API rate limits: 100 requests per minute per key. A 429 response means the "
        "limit was exceeded; back off and retry with exponential delay. There is no "
        "higher tier that raises this limit."
    ),
    "exports.md": (
        "Data export: an admin can export a workspace to CSV from Settings > Export. "
        "Exports run nightly and email a download link when they are ready."
    ),
}

TICKETS: dict[str, dict[str, str]] = {
    "42": {
        "subject": "Password reset link doesn't work",
        "body": "I click the reset link in my email but it just shows an error page.",
    },
    "57": {
        "subject": "Charged twice this month",
        "body": "My card was billed two times for a single subscription this month.",
    },
    "63": {
        "subject": "Getting 429 errors from the API",
        "body": "All of a sudden every request to the API fails with a 429.",
    },
}


def ticket_text(ticket_id: str) -> str:
    """Render a ticket as the prompt the agent answers."""
    ticket = TICKETS[ticket_id]
    return f"Ticket #{ticket_id}: {ticket['subject']}\n\n{ticket['body']}"
