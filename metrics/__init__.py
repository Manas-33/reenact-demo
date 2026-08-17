"""Regression-detection metrics for the demo agents.

Separate from the three shipped agents and their launch-facing break-me PRs: this
package measures how well the reenact gate *catches* seeded regressions across all
three agents (catch rate) without flagging harmless changes (false-positive rate).

A break is a small edit that should fail one of an agent's checks; a benign change
should pass them all. The harness runs each recorded variant against its base
scenario's checks and reports catch rate in three columns - deterministic
(assertion checks), judge (LLM-scored criteria), and combined - paired with the
false-positive rate over the benign changes.
"""
