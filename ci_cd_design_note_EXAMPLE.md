# CI/CD Design Note: Defect Model Pipeline
*Model deliverable example (Week 9)*

## Purpose
Replace the manual "train in a notebook and deploy by hand" flow with an automated,
re-runnable pipeline that retrains safely whenever code or data changes, and only
deploys a model that is demonstrably at least as good as the one in production.

## Pipeline
Five stages, defined once in `mlpipe.py` and run identically in the notebook, the
tests, and CI:

1. **ingest** the incoming training batch.
2. **validate** it against a data contract (a gate).
3. **train** a scikit-learn pipeline (scaling, one-hot encoding, logistic regression).
4. **evaluate** on a fixed golden evaluation set.
5. **promote** the model only if it clears the gate (a gate).

## The two gates
- **Validation gate.** Checks required columns, null rate, numeric ranges, and allowed
  categories. A malformed batch is blocked before any training happens, and in CI this
  fails the build.
- **Promotion gate.** A challenger becomes champion only if its AUC clears an absolute
  threshold **and** beats the current champion. Otherwise it is rejected and the
  champion stays. This is what protects production from a batch that passes validation
  but produces a worse model (for example, drifted labels).

These are complementary. Validation catches data that is broken; the promotion gate
catches data that is intact but wrong.

## Experiment tracking and registry
Every run appends a row to `runs.csv` (run id, rows, metrics, decision, version,
reason). Every trained model is written to a versioned registry
(`model_vN.joblib`) with its metrics recorded in `manifest.json`, and `champion.json`
names the model currently in production. Any decision can be traced back to an exact
model artifact.

## CI
On every push to main and every pull request, the CI job:
1. installs pinned dependencies from `requirements.txt`,
2. runs the test suite (`pytest`),
3. runs the pipeline on the latest data.

A rejected challenger does not fail the build; blocked data does.

## CD
A deploy step promotes the champion to the serving endpoint (the Vertex AI endpoint
from Week 7) only when the CI job is green and the run promoted a new challenger.
Rollback is a pointer change: set the champion alias back to the previous version.

## Reproducibility
Fixed random seeds for the split and the model, pinned dependencies, a fixed golden
evaluation set so every challenger is judged on identical data, and versioned
artifacts. Re-running the pipeline on the same inputs yields the same decision.

## Triggers and cadence
Run on code changes (push, PR) and on a schedule when new data lands. Each run
retrains and re-evaluates; promotion always goes through the gate. Retrain immediately
if monitoring shows the input distribution has shifted.

## Mapping to Google Cloud
The stages map to Vertex AI Pipelines components with a conditional deploy step; the
registry and champion alias map to the Vertex AI Model Registry; `runs.csv` maps to
Vertex AI Experiments; and CI maps to Cloud Build triggers or GitHub Actions invoking
gcloud.

## Limitations
Synthetic tabular data and a single metric (AUC). In production, gate on several
metrics plus slice-level checks (per machine, per shift), add data-drift monitoring on
the live inputs, and require human sign-off for large promotions.
