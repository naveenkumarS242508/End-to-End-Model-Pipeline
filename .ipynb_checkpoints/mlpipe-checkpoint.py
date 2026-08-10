"""mlpipe.py - a small production-style ML pipeline.

Stages: ingest -> validate (gate) -> train -> evaluate -> promote (gate), with a
versioned model registry, a champion/challenger promotion rule, and an experiment
log. Imported by the notebook, the tests, and CI, so the same code runs everywhere.
"""
from __future__ import annotations
import os, json, csv, hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

NUMERIC = ["avg_vibration", "avg_temp_c", "throughput_units",
           "prior_defect_rate", "machine_age_years"]
CATEGORICAL = ["maker", "shift"]
TARGET = "high_defect"
EVAL_SEED = 42          # fixed split for the common evaluation set

# Data contract the validation gate enforces.
SCHEMA = {
    "required": NUMERIC + CATEGORICAL + [TARGET],
    "ranges": {"avg_vibration": (0, 20), "avg_temp_c": (0, 120),
               "throughput_units": (0, 2000), "prior_defect_rate": (0, 1),
               "machine_age_years": (0, 60)},
    "allowed": {"shift": {"day", "night"}, TARGET: {0, 1}},
    "max_null_rate": 0.02,
}


# ---- Stage 1: ingest ----
def ingest(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---- Stage 2: validate (a gate) ----
def validate(df: pd.DataFrame, schema: dict = SCHEMA) -> tuple[bool, list[str]]:
    issues = []
    for col in schema["required"]:
        if col not in df.columns:
            issues.append(f"missing column: {col}")
    if issues:
        return False, issues                      # cannot check further without columns

    for col in schema["required"]:
        null_rate = df[col].isna().mean()
        if null_rate > schema["max_null_rate"]:
            issues.append(f"{col} null rate {null_rate:.2%} exceeds "
                          f"{schema['max_null_rate']:.0%}")
    for col, (lo, hi) in schema["ranges"].items():
        if df[col].min() < lo or df[col].max() > hi:
            issues.append(f"{col} out of range [{lo}, {hi}]")
    for col, allowed in schema["allowed"].items():
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            issues.append(f"{col} has unexpected values: {sorted(bad)}")
    return (len(issues) == 0), issues


# ---- Stage 3: train ----
def train(df: pd.DataFrame, seed: int = 9) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(df[NUMERIC + CATEGORICAL], df[TARGET])
    return model


# ---- Stage 4: evaluate ----
def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"auc": round(float(roc_auc_score(y, proba)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4)}


# ---- Stage 5: promote (a gate) ----
def promote(challenger: dict, champion: dict | None, threshold: float) -> tuple[str, str]:
    """Decide whether the challenger becomes the new champion.

    Rule: it must clear the absolute quality bar AND beat the current champion.
    """
    if challenger["auc"] < threshold:
        return "reject", f"AUC {challenger['auc']} below threshold {threshold}"
    if champion is not None and challenger["auc"] <= champion["auc"]:
        return "reject", (f"AUC {challenger['auc']} does not beat champion "
                          f"{champion['auc']}")
    return "promote", "clears threshold and beats champion"


# ---- Registry helpers ----
def load_champion(registry_dir: str) -> dict | None:
    p = os.path.join(registry_dir, "champion.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _register(model: Pipeline, metrics: dict, registry_dir: str) -> int:
    os.makedirs(registry_dir, exist_ok=True)
    man_path = os.path.join(registry_dir, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"versions": []}
    version = len(manifest["versions"]) + 1
    joblib.dump(model, os.path.join(registry_dir, f"model_v{version}.joblib"))
    manifest["versions"].append({"version": version, "metrics": metrics,
                                 "created": datetime.now(timezone.utc).isoformat()})
    json.dump(manifest, open(man_path, "w"), indent=2)
    return version


def _set_champion(version: int, metrics: dict, registry_dir: str) -> None:
    json.dump({"version": version, "metrics": metrics},
              open(os.path.join(registry_dir, "champion.json"), "w"), indent=2)


def log_run(runs_csv: str, record: dict) -> None:
    exists = os.path.exists(runs_csv)
    with open(runs_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(record.keys()))
        if not exists:
            w.writeheader()
        w.writerow(record)


# ---- Orchestration ----
def run_pipeline(
    train_path,
    eval_path,
    seed=9,
    threshold=0.70
):
    # 1. Ingest
    train_df = ingest(train_path)
    eval_df = ingest(eval_path)

    # 2. Validate
    ok, issues = validate(train_df, SCHEMA)

    if not ok:
        record = {
            "train_path": train_path,
            "auc": None,
            "accuracy": None,
            "f1": None,
            "decision": "blocked",
            "issues": issues,
            "version": None,
            "champion": get_champion(),
        }

        append_run(record)
        return record

    # 3. Train
    model = train(train_df, seed)

    # 4. Evaluate on golden set
    metrics = evaluate(model, eval_df)

    # 5. Register challenger
    version = register_model(model, metrics)

    challenger = {
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
    }

    # 6. Get current champion
    champion = get_champion()

    # 7. Promotion gate
    promoted, reason = promote(
        challenger,
        champion,
        threshold
    )

    if promoted:
        decision = "promote"
        set_champion(version, metrics)
    else:
        decision = "reject"

    # 8. Build run record
    record = {
        "train_path": train_path,
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "decision": decision,
        "reason": reason,
        "version": version,
        "champion": get_champion(),
    }

    # 9. Log run
    append_run(record)

    return record


if __name__ == "__main__":
    import shutil
    shutil.rmtree("registry", ignore_errors=True)
    if os.path.exists("runs.csv"):
        os.remove("runs.csv")
    r = run_pipeline("data/train.csv", "data/eval.csv", "registry", "runs.csv")
    print(r["decision"], r["auc"])
