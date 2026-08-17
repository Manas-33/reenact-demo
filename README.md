# reenact-demo

Example LLM agents, each gated by reenact regression tests. Every pull request
replays the committed scenarios offline and fails only if an agent's behavior
regressed - so a prompt edit or a model swap that quietly breaks an agent is caught
before it merges, not after a customer complains.

Three agents, one per SDK, each showing a different kind of break reenact catches:

| Agent | Stack | Tools | The break it catches |
|---|---|---|---|
| [refund-agent](#refund-agent) | Anthropic SDK | `get_order`, `check_policy`, `issue_refund` | refunds an order that is not eligible |
| [support-agent](#support-agent) | LangGraph | `search_docs`, `label_ticket`, `post_reply` | invents a feature the help center never documents |
| [analyst-agent](#analyst-agent) | OpenAI SDK | `get_schema`, `run_query` | reports a number the query never returned |

Each pull request runs one gate job per agent (a matrix), so a change that breaks one
agent shows that agent's check red while the other two stay green.

## Run the tests

Keyless - a scripted model drives each real agent offline, no key and no network.

    uv sync
    uv run pytest

## refund-agent

A refund assistant built on the Anthropic SDK. It looks up an order, checks the
refund policy, and issues a refund only when the order is eligible.

- `refund_agent/agent.py` - the agent. Three tools: `get_order` and `check_policy`
  (read-only) and `issue_refund` (the mutating action, never re-fired on replay).
- `refund_agent/fixtures.py` - a small order table and the refund policy.
- `refund_agent/record.py` - records scenarios into `scenarios/*.json` (needs a key).

Record real scenarios (needs an Anthropic key), then derive the suite and baseline:

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra record python -m refund_agent.record
    uv run reenact suggest refund_agent/scenarios/ -o refund_agent/evals/suite.toml
    uv run reenact eval refund_agent/evals/suite.toml --write-baseline refund_agent/evals/baseline.json

## support-agent

A help-desk assistant built on LangGraph. It searches the help center, labels the
ticket, and posts a reply - grounded in the articles it retrieved, not in invented
features.

- `support_agent/agent.py` - the agent. Tools `search_docs` and `read_doc` (read-only),
  `label_ticket` and `post_reply` (mutating, never re-fired on replay).
- `support_agent/fixtures.py` - a small help-center knowledge base and a few tickets.
- `support_agent/record.py` - records scenarios (needs a key).

Its suite adds a `reply_grounded` criterion - a model-judged check that catches a reply
inventing a feature, which a keyword assertion cannot. Record and baseline (needs an
Anthropic key to record, and again for the criterion at baseline time):

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra record python -m support_agent.record
    uv run reenact suggest support_agent/scenarios/ -o support_agent/evals/suite.toml
    uv run reenact eval support_agent/evals/suite.toml --write-baseline support_agent/evals/baseline.json

## analyst-agent

A data analyst built on the OpenAI SDK. It inspects a table's schema, runs a read-only
SQL query, and answers using only the figures the query returned.

- `analyst_agent/agent.py` - the agent. Two tools, both read-only: `get_schema` and
  `run_query` (a `SELECT` over the fixtures). There is no mutating tool - the analyst's
  guarantee is faithfulness: every number in the answer must come from the data.
- `analyst_agent/fixtures.py` - a small sales table in throwaway in-memory SQLite.
- `analyst_agent/record.py` - records scenarios (needs an OpenAI key).

Its suite adds an `answer_grounded` criterion - a model-judged check that catches an
answer stating a number the query never returned, which a keyword assertion cannot.
Record with an OpenAI key, then derive the suite and baseline (the criterion is judged
by an Anthropic model, so the baseline write needs an Anthropic key too):

    export OPENAI_API_KEY=sk-...
    uv run --extra record python -m analyst_agent.record
    uv run reenact suggest analyst_agent/scenarios/ -o analyst_agent/evals/suite.toml
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run reenact eval analyst_agent/evals/suite.toml --write-baseline analyst_agent/evals/baseline.json
