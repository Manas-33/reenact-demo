"""The regression corpus: break and benign variants per agent.

A BREAK is a small edit that should fail one of an agent's checks, so the gate
catches it: a degraded system prompt (drop a step, decline everything) or a
renamed tool. A BENIGN change rewords the shipped prompt without changing what the
agent does, so every check still passes and the gate stays quiet. Each variant
names the base scenario whose checks apply, so the harness scores it correctly.

Whether a designed break *actually* breaks - and a benign one stays clean - is
verified at record time by inspecting the recording; the model, not this file, has
the final say. This is the corpus of *intended* labels the recording confirms.
"""

from __future__ import annotations

from dataclasses import dataclass

from metrics.harness import DETERMINISTIC, JUDGE


@dataclass(frozen=True)
class Variant:
    """One recorded variant of an agent: what it is, and what should catch it."""

    agent: str  # "refund" | "support" | "analyst"
    name: str  # unique within the agent
    kind: str  # "break" | "benign"
    column: str  # DETERMINISTIC | JUDGE for a break; "" for a benign change
    base_scenario: str  # suite scenario whose checks apply
    scenario_input: str  # the request/ticket/question to send
    system: str | None = None  # system-prompt override (None = shipped prompt)
    rename: tuple[str, str] | None = None  # (old_tool, new_tool) for a rename break
    note: str = ""


# --- Refund agent (Anthropic SDK) -------------------------------------------

_DROP_ELIGIBILITY = (
    "You are a refund assistant for an online store. The customer is always right: "
    "look up the order with get_order and issue a refund with issue_refund for "
    "whatever the customer asks. Do not deny refunds."
)
_SKIP_POLICY = (
    "You are a refund assistant. Look up the order with get_order and, if the "
    "request seems reasonable, issue the refund with issue_refund. There is no need "
    "to check the refund policy."
)
_STRICT_DECLINE = (
    "You are a refund assistant. Refunds are almost never allowed. Look up the order "
    "and check the policy, then decline the request and explain that it is not "
    "eligible. Do not issue refunds."
)
_NO_TOOLS = (
    "You are a refund assistant. Answer the customer's refund request directly from "
    "your own judgement. Do not use any tools."
)

_REFUND_BENIGN_PROMPTS = {
    "friendly": (
        "You are a warm, friendly refund assistant for an online store. For each "
        "request: look up the order with get_order, check eligibility with "
        "check_policy, and issue a refund with issue_refund ONLY if the order is "
        "eligible. If it is not eligible, kindly decline and explain why, citing the "
        "policy. Keep a warm, reassuring tone throughout."
    ),
    "verbose": (
        "You are a refund assistant for an online store. For each request: look up "
        "the order with get_order, check eligibility with check_policy, and issue a "
        "refund with issue_refund ONLY if the order is eligible. If it is not "
        "eligible, do not issue a refund - decline and explain in detail which "
        "policy rule applies and why, quoting the relevant clause."
    ),
    "terse": (
        "You are a refund assistant. Steps: get_order, then check_policy, then "
        "issue_refund only if eligible; otherwise decline. Reply in one short "
        "sentence, citing the policy when you decline."
    ),
    "reordered": (
        "You are a refund assistant for an online store. Only issue a refund with "
        "issue_refund when the order is eligible - and to know that, always check "
        "eligibility with check_policy first, after looking the order up with "
        "get_order. When an order is not eligible, decline politely and cite the "
        "policy rather than issuing a refund."
    ),
    "double-check": (
        "You are a refund assistant for an online store. For each request: look up "
        "the order with get_order, check eligibility with check_policy, and issue a "
        "refund with issue_refund ONLY if the order is eligible. Before you reply, "
        "double-check that your decision matches the eligibility result. If it is not "
        "eligible, do not issue a refund - decline and explain why, citing the policy."
    ),
}

# (order id, base scenario whose checks apply, the customer request)
_REFUND_ORDERS = [
    ("1001", "refund-eligible", "Refund order 1001, the headphones arrived damaged."),
    ("1002", "refund-final-sale", "Refund order 1002, the festival ticket."),
    ("1003", "refund-past-window", "Refund order 1003, the desk lamp broke."),
    ("1004", "refund-eligible", "Refund order 1004, the bluetooth speaker is faulty."),
    ("1005", "refund-final-sale", "Refund order 1005, the office chair."),
    ("1006", "refund-eligible", "Refund order 1006, the coffee maker leaks."),
]


def _refund_breaks() -> list[Variant]:
    return [
        Variant(
            "refund", "refund-ineligible-1002", "break", DETERMINISTIC,
            "refund-final-sale",
            "I'd like a refund for order 1002, the festival ticket.",
            system=_DROP_ELIGIBILITY,
            note="drops the eligibility gate -> refunds a final-sale order",
        ),
        Variant(
            "refund", "refund-ineligible-1003", "break", DETERMINISTIC,
            "refund-past-window",
            "Refund order 1003 please, the desk lamp stopped working.",
            system=_DROP_ELIGIBILITY,
            note="drops the eligibility gate -> refunds a past-window order",
        ),
        Variant(
            "refund", "skip-policy", "break", DETERMINISTIC,
            "refund-eligible",
            "Please refund order 1001, the headphones arrived damaged.",
            system=_SKIP_POLICY,
            note="never calls check_policy",
        ),
        Variant(
            "refund", "rename-issue-refund", "break", DETERMINISTIC,
            "refund-eligible",
            "Please refund order 1001, the headphones arrived damaged.",
            rename=("issue_refund", "process_refund"),
            note="issue_refund renamed -> called_tool('issue_refund') fails",
        ),
        Variant(
            "refund", "decline-eligible", "break", DETERMINISTIC,
            "refund-eligible",
            "Please refund order 1001, the headphones arrived damaged.",
            system=_STRICT_DECLINE,
            note="declines a valid refund -> issue_refund never called",
        ),
        Variant(
            "refund", "no-tools", "break", DETERMINISTIC,
            "refund-eligible",
            "Please refund order 1001, the headphones arrived damaged.",
            system=_NO_TOOLS,
            note="answers without tools -> get_order never called",
        ),
    ]


def _refund_benign() -> list[Variant]:
    out: list[Variant] = []
    for key, prompt in _REFUND_BENIGN_PROMPTS.items():
        for order_id, base, request in _REFUND_ORDERS:
            out.append(
                Variant(
                    "refund", f"benign-{key}-{order_id}", "benign", "",
                    base, request, system=prompt,
                    note=f"{key} reword of the shipped prompt",
                )
            )
    return out


def refund_variants() -> list[Variant]:
    """The refund agent's break + benign corpus (6 breaks, 30 benign)."""
    return _refund_breaks() + _refund_benign()


# --- Support agent (LangGraph RAG) ------------------------------------------

_SUPPORT_SKIP_LABEL = (
    "You are a support assistant for a SaaS product. For the ticket: search the help "
    "center for relevant articles and post a short reply grounded in them - do not "
    "invent features the articles do not state. Finish with a one-sentence summary. "
    "There is no need to categorize or label the ticket."
)
_SUPPORT_SKIP_REPLY = (
    "You are a support assistant for a SaaS product. For the ticket: search the help "
    "center and apply exactly one category label (account, billing, or api). Do not "
    "post a public reply - a human agent will follow up. Finish with a summary."
)
_SUPPORT_SKIP_SEARCH = (
    "You are a support assistant for a SaaS product. Answer the ticket from your own "
    "product knowledge - no need to search the help center. Apply exactly one category "
    "label (account, billing, or api) and post a short reply. Finish with a summary."
)
_SUPPORT_INVENT = (
    "You are a support assistant for a SaaS product. For the ticket: search the help "
    "center, apply exactly one category label (account, billing, or api), and post a "
    "helpful reply. Go above and beyond: suggest premium tiers, upgrades, mobile apps, "
    "and other fixes that would delight the customer, whether or not the articles "
    "mention them. Finish with a one-sentence summary."
)

_SUPPORT_BENIGN_PROMPTS = {
    "friendly": (
        "You are a warm, friendly support assistant for a SaaS product. For the "
        "ticket: search the help center, apply exactly one category label (account, "
        "billing, or api), and post a short reply grounded in the articles you "
        "retrieved - do not invent features the articles do not state. Finish with a "
        "one-sentence summary, and keep a reassuring tone."
    ),
    "verbose": (
        "You are a support assistant for a SaaS product. For the ticket: search the "
        "help center, apply exactly one category label (account, billing, or api), and "
        "post a thorough reply that walks through the steps, grounded strictly in the "
        "retrieved articles - do not invent anything they do not state. Finish with a "
        "one-sentence summary."
    ),
    "terse": (
        "You are a support assistant. Steps: search the help center, apply one label "
        "(account, billing, or api), post a reply grounded in the docs (never invent "
        "features), then summarize in one sentence."
    ),
    "reordered": (
        "You are a support assistant for a SaaS product. Always ground your reply in "
        "the help-center articles - never invent features, tiers, or policies they do "
        "not state. To do that, first search the help center, then apply exactly one "
        "category label (account, billing, or api), then post a short grounded reply, "
        "and finish with a one-sentence summary."
    ),
    "structured": (
        "You are a support assistant for a SaaS product. For the ticket: search the "
        "help center, apply exactly one category label (account, billing, or api), and "
        "post a reply grounded in the retrieved docs - never invent features they do "
        "not state. Structure the reply as a brief greeting then the fix. "
        "Finish with a one-sentence summary of what you did."
    ),
}

_SUPPORT_PASSWORD = "Ticket #42: my password reset link shows an error when I click it."
_SUPPORT_BILLING = "Ticket #57: my monthly billing shows two charges for one plan."
_SUPPORT_RATELIMIT = "Ticket #63: every API request suddenly fails with a 429 error."

# (base scenario whose checks apply, the ticket) - two per category, DOCS-answerable.
_SUPPORT_TICKETS = [
    ("support-42", _SUPPORT_PASSWORD),
    ("support-42", "Ticket #71: I forgot my password; no reset email arrives."),
    ("support-57", _SUPPORT_BILLING),
    ("support-57", "Ticket #72: a billing question - an invoice looks duplicated."),
    ("support-63", _SUPPORT_RATELIMIT),
    ("support-63", "Ticket #73: my API keeps hitting a 429 rate limit."),
]


def _support_breaks() -> list[Variant]:
    return [
        Variant(
            "support", "skip-label", "break", DETERMINISTIC, "support-42",
            _SUPPORT_PASSWORD, system=_SUPPORT_SKIP_LABEL,
            note="never calls label_ticket",
        ),
        Variant(
            "support", "skip-reply", "break", DETERMINISTIC, "support-42",
            _SUPPORT_PASSWORD, system=_SUPPORT_SKIP_REPLY,
            note="never calls post_reply",
        ),
        Variant(
            "support", "skip-search", "break", DETERMINISTIC, "support-42",
            _SUPPORT_PASSWORD, system=_SUPPORT_SKIP_SEARCH,
            note="never calls search_docs",
        ),
        Variant(
            "support", "rename-post-reply", "break", DETERMINISTIC, "support-42",
            _SUPPORT_PASSWORD, rename=("post_reply", "send_message"),
            note="post_reply renamed -> called_tool('post_reply') fails",
        ),
        Variant(
            "support", "invent-feature-42", "break", JUDGE, "support-42",
            _SUPPORT_PASSWORD, system=_SUPPORT_INVENT,
            note="invents an undocumented feature -> reply_grounded fails",
        ),
        Variant(
            "support", "invent-feature-57", "break", JUDGE, "support-57",
            _SUPPORT_BILLING, system=_SUPPORT_INVENT,
            note="invents an undocumented billing policy",
        ),
        Variant(
            "support", "invent-feature-63", "break", JUDGE, "support-63",
            _SUPPORT_RATELIMIT, system=_SUPPORT_INVENT,
            note="invents a premium tier the rate-limit doc rules out",
        ),
    ]


def _support_benign() -> list[Variant]:
    out: list[Variant] = []
    for key, prompt in _SUPPORT_BENIGN_PROMPTS.items():
        for index, (base, ticket) in enumerate(_SUPPORT_TICKETS):
            out.append(
                Variant(
                    "support", f"benign-{key}-{index}", "benign", "",
                    base, ticket, system=prompt,
                    note=f"{key} reword of the shipped prompt",
                )
            )
    return out


def support_variants() -> list[Variant]:
    """The support corpus (4 deterministic + 3 judge breaks, 30 benign)."""
    return _support_breaks() + _support_benign()


# --- Analyst agent (OpenAI SDK, read-only, faithfulness) --------------------

_ANALYST_SKIP_SCHEMA = (
    "You are a data analyst. Answer using the database: call run_query with a SQL "
    "SELECT to get the figures, then answer. No need to inspect the schema first. "
    "Every number in your answer must come from the query results."
)
_ANALYST_SKIP_QUERY = (
    "You are a data analyst. Answer the user's question about the sales data from "
    "your own knowledge of the business - no need to query the database. Give your "
    "best answer in one or two sentences."
)
_ANALYST_HALLUCINATE = (
    "You are a data analyst. Answer the user's question about the sales data. Call "
    "get_schema and run_query, but if the exact figure is unclear just give your best "
    "estimate - a confident round number reads better than hedging. Answer in one or "
    "two sentences."
)
_ANALYST_CITE_UNFETCHED = (
    "You are a data analyst. Answer using the database: call get_schema, run_query, "
    "then state the figure. Make the answer richer with helpful extra context - "
    "related totals, comparisons, and trends - estimating those extras yourself where "
    "the query did not return them."
)

_ANALYST_BENIGN_PROMPTS = {
    "friendly": (
        "You are a warm, friendly data analyst. Answer the user's question about the "
        "sales data using ONLY the database: first call get_schema, then run_query "
        "with a SQL SELECT, then answer in one or two sentences. Every number must "
        "come from the query results - never estimate or invent a figure."
    ),
    "verbose": (
        "You are a data analyst. Answer the question about the sales data using ONLY "
        "the database: call get_schema, then run_query with a SQL SELECT, then explain "
        "the answer and how the query produced it. Every number must come from the "
        "query results - never estimate or invent a figure."
    ),
    "terse": (
        "You are a data analyst. Steps: get_schema, run_query (a SQL SELECT), then "
        "answer in one short sentence. Every number must come from the query results; "
        "never estimate or invent."
    ),
    "reordered": (
        "You are a data analyst. Every number in your answer must come from the query "
        "results - never estimate or invent one. To get them: first call get_schema, "
        "then run_query with a SQL SELECT, then answer in one or two sentences."
    ),
    "structured": (
        "You are a data analyst. Answer the question about the sales data using ONLY "
        "the database: call get_schema, then run_query with a SQL SELECT, then give a "
        "one-line answer followed by the figure. Every number must come from the query "
        "results - never estimate or invent a figure."
    ),
}

_ANALYST_WEST = "What was our total revenue in the West region?"
_ANALYST_TOP = "Which product sold the most units, and how many units was that?"
_ANALYST_Q4 = "What was our total revenue in Q4?"

# (base scenario whose checks apply, the question) - two phrasings per figure.
_ANALYST_QUESTIONS = [
    ("analyst-west-revenue", _ANALYST_WEST),
    ("analyst-west-revenue", "How much revenue did the West region bring in overall?"),
    ("analyst-top-product", _ANALYST_TOP),
    ("analyst-top-product", "What is our best-selling product by units, and how many?"),
    ("analyst-q4-revenue", _ANALYST_Q4),
    ("analyst-q4-revenue", "How much total revenue did we make in Q4?"),
]


def _analyst_breaks() -> list[Variant]:
    return [
        Variant(
            "analyst", "skip-schema", "break", DETERMINISTIC, "analyst-q4-revenue",
            _ANALYST_Q4, system=_ANALYST_SKIP_SCHEMA, note="never calls get_schema",
        ),
        Variant(
            "analyst", "skip-query", "break", DETERMINISTIC, "analyst-q4-revenue",
            _ANALYST_Q4, system=_ANALYST_SKIP_QUERY, note="never calls run_query",
        ),
        Variant(
            "analyst", "rename-run-query", "break", DETERMINISTIC, "analyst-q4-revenue",
            _ANALYST_Q4, rename=("run_query", "execute_sql"),
            note="run_query renamed -> called_tool('run_query') fails",
        ),
        Variant(
            "analyst", "hallucinate-figure", "break", DETERMINISTIC,
            "analyst-q4-revenue", _ANALYST_Q4, system=_ANALYST_HALLUCINATE,
            note="estimates a figure -> the distinctive number drops from the answer",
        ),
        Variant(
            "analyst", "cite-unfetched-west", "break", JUDGE, "analyst-west-revenue",
            _ANALYST_WEST, system=_ANALYST_CITE_UNFETCHED,
            note="states the figure but fabricates extra numbers",
        ),
        Variant(
            "analyst", "cite-unfetched-top", "break", JUDGE, "analyst-top-product",
            _ANALYST_TOP, system=_ANALYST_CITE_UNFETCHED,
            note="fabricates extra numbers alongside the real one",
        ),
        Variant(
            "analyst", "cite-unfetched-q4", "break", JUDGE, "analyst-q4-revenue",
            _ANALYST_Q4, system=_ANALYST_CITE_UNFETCHED,
            note="fabricates extra numbers alongside the real one",
        ),
    ]


def _analyst_benign() -> list[Variant]:
    out: list[Variant] = []
    for key, prompt in _ANALYST_BENIGN_PROMPTS.items():
        for index, (base, question) in enumerate(_ANALYST_QUESTIONS):
            out.append(
                Variant(
                    "analyst", f"benign-{key}-{index}", "benign", "",
                    base, question, system=prompt,
                    note=f"{key} reword of the shipped prompt",
                )
            )
    return out


def analyst_variants() -> list[Variant]:
    """The analyst corpus (4 deterministic + 3 judge breaks, 30 benign)."""
    return _analyst_breaks() + _analyst_benign()


def all_variants() -> list[Variant]:
    """Every agent's break + benign variants, across the three frameworks."""
    return refund_variants() + support_variants() + analyst_variants()
