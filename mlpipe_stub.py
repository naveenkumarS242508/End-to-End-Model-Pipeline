"""mlpipe.py (STARTER STUB) - complete the TODOs, then rename this file to mlpipe.py.

A small production-style ML pipeline: ingest -> validate (gate) -> train -> evaluate
-> promote (gate), with a versioned registry and an experiment log. The registry
helpers and logging are given; you implement the stages and the two gates.
"""
from __future__ import annotations
import os, json, csv
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

SCHEMA = {
    "required": NUMERIC + CATEGORICAL + [TARGET],
    "ranges": {"avg_vibration": (0, 20), "avg_temp_c": (0, 120),
               "throughput_units": (0, 2000), "prior_defect_rate": (0, 1),
               "machine_age_years": (0, 60)},
    "allowed": {"shift": {"day", "night"}, TARGET: {0, 1}},
    "max_null_rate": 0.02,
}


def ingest(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate(df: pd.DataFrame, schema: dict = SCHEMA) -> tuple[bool, list[str]]:
    """Return (ok, issues). Check required columns, null rate, ranges, allowed values.
    TODO: build a list of issue strings; ok is True only when it is empty."""
    raise NotImplementedError


def train(df: pd.DataFrame, seed: int = 9) -> Pipeline:
    """TODO: a scikit-learn Pipeline (StandardScaler on NUMERIC, OneHotEncoder on
    CATEGORICAL, then LogisticRegression(random_state=seed)); fit and return it."""
    raise NotImplementedError


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    """TODO: return {"auc":..., "accuracy":..., "f1":...} on df."""
    raise NotImplementedError


def promote(challenger: dict, champion: dict | None, threshold: float) -> tuple[str, str]:
    """TODO: return ("promote"|"reject", reason). Promote only if the challenger clears
    the threshold AND beats the champion (or there is no champion yet)."""
    raise NotImplementedError


# ---- Registry helpers (given) ----
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


def run_pipeline(train_path: str, eval_path: str, registry_dir: str, runs_csv: str,
                 seed: int = 9, threshold: float = 0.80,
                 run_id: str | None = None) -> dict:
    """TODO: orchestrate the stages.
    1. ingest + validate the training file; if invalid, log decision="blocked" and return.
    2. ingest the golden eval file; train; evaluate.
    3. _register the model; compare against load_champion via promote().
    4. if promoted, _set_champion; log the run; return the record.
    """
    raise NotImplementedError
