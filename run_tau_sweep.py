"""
Sensitivity sweep for the semantic-aware temperature coefficient tau.

For each tau value it launches a full training run of ORCA-GCN and records the
final test metrics into a CSV. tau = 1.0 corresponds to the plain softmax(1-cos)
form written in the paper (i.e. the "no temperature" variant), so a single sweep
covers both the sensitivity analysis and the ablation that removes tau.

Usage:
    python run_tau_sweep.py --data FB15k-237 --gpu 0
    python run_tau_sweep.py --data FB15k-237 --gpu 0 --taus 0.2 0.4 0.6 0.8 1.0
"""

import argparse
import os
import re
import subprocess
import csv

# The final block printed by main.py looks like:
#   Test MRR: 0.363
#   , MR: ...
#   , H@10: 0.544
#   , H@3: 0.398
#   , H@1: 0.272
_FLOAT = r"([-+]?\d*\.\d+|\d+)"


def parse_final_test(log_text):
    """Extract MRR/H@1/H@3/H@10 from the *last* 'Test MRR' block in the log."""
    idx = log_text.rfind("Test MRR")
    if idx == -1:
        return None
    tail = log_text[idx:]
    out = {}
    for key, pat in [
        ("MRR", r"Test MRR:\s*" + _FLOAT),
        ("H@1", r"H@1:\s*" + _FLOAT),
        ("H@3", r"H@3:\s*" + _FLOAT),
        ("H@10", r"H@10:\s*" + _FLOAT),
    ]:
        m = re.search(pat, tail)
        out[key] = float(m.group(1)) if m else None
    return out


def main():
    parser = argparse.ArgumentParser(description="tau sensitivity sweep")
    parser.add_argument("--data", default="FB15k-237", help="Dataset name")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--score_func", default="conve")
    parser.add_argument("--opn", default="ccorr")
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="tau values to sweep (include 1.0 for the no-temperature case).",
    )
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Extra args passed straight through to main.py.")
    args = parser.parse_args()

    os.makedirs("sweep_logs", exist_ok=True)
    rows = []

    for tau in args.taus:
        log_path = os.path.join("sweep_logs", f"{args.data}_tau{tau}.log")
        cmd = [
            "python", "main.py",
            "--data", args.data,
            "--score_func", args.score_func,
            "--opn", args.opn,
            "--gpu", str(args.gpu),
            "--tau", str(tau),
        ] + args.extra

        print(f"\n=== tau={tau} ===\n{' '.join(cmd)}")
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logf.write(proc.stdout)

        metrics = parse_final_test(proc.stdout) or {}
        row = {"tau": tau, **{k: metrics.get(k) for k in ["MRR", "H@1", "H@3", "H@10"]}}
        rows.append(row)
        print(f"tau={tau} -> {row}")

    out_csv = f"tau_sensitivity_{args.data}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tau", "MRR", "H@1", "H@3", "H@10"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary to {out_csv}")


if __name__ == "__main__":
    main()