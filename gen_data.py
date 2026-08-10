"""Generate the datasets the Week 9 pipeline uses.

A fixed golden EVAL set, plus several TRAIN files that stand for real-world data
pulls the pipeline has to judge:
  - train.csv          a clean baseline batch
  - train_v2.csv       a larger clean batch (a better model, should be promoted)
  - train_drifted.csv  a batch with label noise (passes validation, worse model, rejected)
  - train_invalid.csv  a malformed batch (blocked by the validation gate)
"""
import os
import numpy as np, pandas as pd

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(9)
machines = {"M-01": ("Hokuto", 6), "M-02": ("Hokuto", 4), "M-03": ("Brunner", 11),
            "M-04": ("Brunner", 3), "M-05": ("Voss", 5), "M-06": ("Voss", 2),
            "M-07": ("Brunner", 12), "M-08": ("Hokuto", 4)}
mids = list(machines)


def make(n):
    rows = []
    for _ in range(n):
        mid = rng.choice(mids); maker, age = machines[mid]
        shift = rng.choice(["day", "night"])
        vib = rng.normal(2.8 + age * 0.12, 0.5)
        temp = rng.normal(58 + age * 0.6, 3.0)
        thr = rng.normal(520 - age * 4, 40)
        prior = float(np.clip(rng.normal(0.05 + age * 0.004 + (shift == "night") * 0.02, 0.02), 0, 0.4))
        risk = (0.9 * (vib - 3.0) + 0.05 * (temp - 60) + 3.0 * (prior - 0.06)
                + 0.6 * (shift == "night") + 0.4 * (mid == "M-07") + rng.normal(0, 0.6))
        rows.append([mid, maker, shift, round(vib, 2), round(temp, 1),
                     round(thr, 1), round(prior, 3), age, int(risk > 1.0)])
    return pd.DataFrame(rows, columns=["machine_id", "maker", "shift", "avg_vibration",
                        "avg_temp_c", "throughput_units", "prior_defect_rate",
                        "machine_age_years", "high_defect"])


pool = make(1800)
eval_df = pool.iloc[:300].reset_index(drop=True)
rest = pool.iloc[300:].reset_index(drop=True)

train = rest.iloc[:700].reset_index(drop=True)
train_v2 = rest.reset_index(drop=True)

drifted = train_v2.copy()
flip = rng.random(len(drifted)) < 0.30
drifted.loc[flip, "high_defect"] = 1 - drifted.loc[flip, "high_defect"]

invalid = train.copy()
invalid.loc[invalid.index[:20], "avg_vibration"] = 999.0
invalid.loc[invalid.index[20:25], "shift"] = "twilight"

eval_df.to_csv("data/eval.csv", index=False)
train.to_csv("data/train.csv", index=False)
train_v2.to_csv("data/train_v2.csv", index=False)
drifted.to_csv("data/train_drifted.csv", index=False)
invalid.to_csv("data/train_invalid.csv", index=False)

for name, d in [("eval", eval_df), ("train", train), ("train_v2", train_v2),
                ("train_drifted", drifted), ("train_invalid", invalid)]:
    print(f"{name:14s} rows={len(d):5d}  pos_rate={d.high_defect.mean():.3f}")
