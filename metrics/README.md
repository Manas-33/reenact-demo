# Regression-detection metrics

Measures how well the reenact gate catches seeded regressions across the three
demo agents - refund (Anthropic SDK), support (LangGraph), analyst (OpenAI SDK) -
without flagging harmless changes.

- **Corpus:** 110 real recordings - 20 seeded breaks (15 distinct types) + 90
  benign rewords - across all three frameworks. Recorded once against the live
  APIs; scored offline forever after.
- **Break** = a degraded prompt or renamed tool that should fail a check.
  **Benign** = a reworded prompt that should still pass every check.
- Catch rate is reported in three columns - deterministic (assertion checks),
  judge (grounding criteria), combined - paired with the false-positive rate.

## Results

| Column | Caught | Rate |
|---|---|---|
| Deterministic (assertions) | 11 / 14 | 78.6% |
| Judge (grounding criteria) | 6 / 6 | 100% |
| Combined | 17 / 20 | 85% |
| False positives | 1 / 90 | 1.1% FPR |

**Reading the catch rate.** The 3 non-catches are no-ops, not gate failures: in
each one the agent produced the correct output despite the seeded change (the
model ignored a degraded prompt, or recovered from a renamed tool), so no
regression actually occurred. Recall over the seeded changes that *did* degrade
behavior is **17 / 17 = 100%**, including every grounding/faithfulness violation,
at a **1.1% false-positive rate** across 90 benign changes.

The single false positive is the grounding judge over-flagging one grounded reply
- the precision cost of a rubric strict enough to catch 6/6 faithfulness breaks.

## Reproduce

```
set -a; source .env; set +a
uv run --extra record python -m metrics.record_regressions   # record the corpus (spends tokens)
uv run --extra record python -m metrics.measure              # score it -> metrics/results.json
```

Recording needs `ANTHROPIC_API_KEY` (refund, support) and `OPENAI_API_KEY`
(analyst); the grounding criteria are judged with `ANTHROPIC_API_KEY`. The
committed recordings replay offline, so `measure` reproduces the numbers without
re-recording (the judge column still needs a key).
