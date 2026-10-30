# Synthetic evaluation

This directory contains a deterministic, zero-API-cost regression evaluation for the
Research Agent.

## Run

```powershell
python -m evals.run_synthetic_eval
```

The command reads `synthetic_cases.json`, starts an in-process FastAPI application,
and uses the schema-aware fake model from the test suite. It evaluates:

- segmented workflow completion and Checkpoint resume;
- two-turn conversation linkage;
- per-turn model selection persistence;
- presence of safe reasoning metadata without the user prompt;
- report generation;
- API rejection of malformed or orphaned requests;
- in-process P50/P95 latency for regression comparison.

The synthetic run does **not** evaluate factual correctness, citation entailment,
source authority/freshness, live-provider latency, or monetary cost. Those require a
configured model, Tavily credentials, and a separately annotated golden dataset.

The same evaluation is enforced by `tests/test_synthetic_eval.py` during the normal
pytest run.
