# reenact-demo

Example LLM agents, each gated by reenact regression tests. Every pull request
replays the committed scenarios offline and fails only if an agent's behavior
regressed - so a prompt edit or a model swap that quietly breaks an agent is caught
before it merges, not after a customer complains.

## refund-agent

A refund assistant built on the Anthropic SDK. It looks up an order, checks the
refund policy, and issues a refund only when the order is eligible.

- `refund_agent/agent.py` - the agent. Three tools: `get_order` and `check_policy`
  (read-only) and `issue_refund` (the mutating action, never re-fired on replay).
- `refund_agent/fixtures.py` - a small order table and the refund policy.
- `refund_agent/record.py` - records scenarios into `scenarios/*.json` (needs a key).
- `tests/` - keyless tests: a scripted model drives the real agent offline.

### Run the tests

    uv sync
    uv run pytest

### Record real scenarios (needs an Anthropic key)

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra record python -m refund_agent.record

Then derive the eval suite from a recording and set the baseline:

    uv run reenact suggest refund_agent/scenarios/eligible.json -o refund_agent/evals/suite.toml
    uv run reenact eval refund_agent/evals/suite.toml --write-baseline refund_agent/evals/baseline.json
