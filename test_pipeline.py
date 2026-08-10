"""Tests for the ML pipeline. Run with: pytest -q

These guard the two things that keep a pipeline safe: the validation gate rejects bad
data, and the promotion gate never lets a worse model reach production. They also pin
reproducibility.
"""
import pandas as pd
import pytest

import mlpipe

TRAIN = "data/train.csv"
EVAL = "data/eval.csv"
INVALID = "data/train_invalid.csv"


# ---- validation gate ----
def test_validate_passes_clean_data():
    ok, issues = mlpipe.validate(mlpipe.ingest(TRAIN))
    assert ok and issues == []


def test_validate_catches_out_of_range():
    df = mlpipe.ingest(TRAIN).copy()
    df.loc[df.index[0], "avg_vibration"] = 999.0
    ok, issues = mlpipe.validate(df)
    assert not ok and any("avg_vibration" in i for i in issues)


def test_validate_catches_missing_column():
    df = mlpipe.ingest(TRAIN).drop(columns=["maker"])
    ok, issues = mlpipe.validate(df)
    assert not ok and any("maker" in i for i in issues)


def test_validate_catches_bad_category():
    df = mlpipe.ingest(TRAIN).copy()
    df.loc[df.index[0], "shift"] = "twilight"
    ok, issues = mlpipe.validate(df)
    assert not ok and any("shift" in i for i in issues)


# ---- training is reproducible ----
def test_train_is_reproducible():
    df = mlpipe.ingest(TRAIN)
    ev = mlpipe.ingest(EVAL)
    a = mlpipe.evaluate(mlpipe.train(df, seed=9), ev)
    b = mlpipe.evaluate(mlpipe.train(df, seed=9), ev)
    assert a == b


# ---- promotion gate ----
def test_promote_accepts_better():
    decision, _ = mlpipe.promote({"auc": 0.90}, {"auc": 0.85}, threshold=0.80)
    assert decision == "promote"


def test_promote_rejects_worse_than_champion():
    decision, _ = mlpipe.promote({"auc": 0.82}, {"auc": 0.88}, threshold=0.80)
    assert decision == "reject"


def test_promote_rejects_below_threshold():
    decision, _ = mlpipe.promote({"auc": 0.70}, None, threshold=0.80)
    assert decision == "reject"


def test_promote_first_model_when_no_champion():
    decision, _ = mlpipe.promote({"auc": 0.85}, None, threshold=0.80)
    assert decision == "promote"


# ---- end to end ----
def test_pipeline_runs_and_promotes_first_model(tmp_path):
    reg = str(tmp_path / "registry")
    runs = str(tmp_path / "runs.csv")
    rec = mlpipe.run_pipeline(TRAIN, EVAL, reg, runs, run_id="t1")
    assert rec["decision"] == "promote" and rec["version"] == 1
    assert mlpipe.load_champion(reg)["version"] == 1


def test_pipeline_blocks_invalid_batch(tmp_path):
    reg = str(tmp_path / "registry")
    runs = str(tmp_path / "runs.csv")
    rec = mlpipe.run_pipeline(INVALID, EVAL, reg, runs, run_id="t2")
    assert rec["decision"] == "blocked"
    assert mlpipe.load_champion(reg) is None       # nothing promoted from bad data
