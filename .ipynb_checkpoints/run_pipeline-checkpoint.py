"""CLI entry point for CI: run the pipeline on a training batch and a golden eval set.

Exit code 1 if the data is blocked by validation (a bad batch should fail the build);
a 'reject' decision is the gate working as intended, so it exits 0.
"""
import argparse
import mlpipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--registry", default="registry")
    p.add_argument("--runs", default="runs.csv")
    p.add_argument("--threshold", type=float, default=0.80)
    a = p.parse_args()
    rec = mlpipe.run_pipeline(a.train, a.eval, a.registry, a.runs, threshold=a.threshold)
    print(f"decision={rec['decision']} auc={rec['auc']} version={rec['version']} "
          f"reason={rec['reason']}")
    raise SystemExit(1 if rec["decision"] == "blocked" else 0)


if __name__ == "__main__":
    main()
