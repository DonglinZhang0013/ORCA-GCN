"""
One-stop reproduction command list for ORCA-GCN.

Each entry maps to a table/figure in the paper. Run everything, a single group,
or just print the commands.

Examples:
    python run_commands.py --gpu 0                 # run all groups sequentially
    python run_commands.py --gpu 0 --only table2   # run one group
    python run_commands.py --dry_run               # just print the commands

Prerequisites:
    * FB15k-237/  contains graph_with_olliverRicci.npy + olliverRicci.npy
    * WN18RR/     same two files (generate once via:
                      python scripts/build_wn18rr_orc_npy.py )
"""

import argparse
import subprocess

BASE = ["python", "main.py", "--score_func", "conve", "--opn", "ccorr"]


def build(gpu):
    def m(*extra):
        return BASE + ["--gpu", str(gpu)] + list(extra)

    groups = {
        # ---- Table 1: main results ----
        "table1": [
            ("Table 1 | FB15k-237 | ORCA-GCN", m("--data", "FB15k-237", "--tau", "0.6")),
            ("Table 1 | WN18RR    | ORCA-GCN", m("--data", "WN18RR",    "--tau", "0.6")),
        ],
        # ---- Table 2: ablation study (FB15k-237) ----
        "table2": [
            ("Table 2 | ORCA-GCN (full)",      m("--data", "FB15k-237", "--tau", "0.6")),
            ("Table 2 | w/o Geometry-Aware",   m("--data", "FB15k-237", "--tau", "0.6", "--uniform_geometry")),
            ("Table 2 | w/o Semantic-Aware",   m("--data", "FB15k-237", "--tau", "0")),
            ("Table 2 | w/ Random Curvature",  m("--data", "FB15k-237", "--tau", "0.6", "--random_curvature")),
        ],
        # ---- tau sensitivity (Section 4.3). tau=1.0 == plain softmax(1-cos) ----
        "tau": [
            ("tau sweep | FB15k-237", ["python", "run_tau_sweep.py", "--data", "FB15k-237",
                                       "--gpu", str(gpu), "--taus", "0.2", "0.4", "0.6", "0.8", "1.0"]),
            ("tau sweep | WN18RR",    ["python", "run_tau_sweep.py", "--data", "WN18RR",
                                       "--gpu", str(gpu), "--taus", "0.2", "0.4", "0.6", "0.8", "1.0"]),
        ],
        # ---- Table 3: edge pruning by ORC (base CompGCN aggregation) ----
        # Table 3 uses the base CompGCN backbone; within this repo that is the
        # both-modules-off config (--uniform_geometry --tau 0). Adjust to your
        # exact base-CompGCN setup if it differs.
        "table3": [
            (f"Table 3 | FB15k-237 | k={int(r*100)}%",
             m("--data", "FB15k-237", "--uniform_geometry", "--tau", "0", "--prune_topk_orc", str(r)))
            for r in [0.0, 0.2, 0.4, 0.6]
        ] + [
            (f"Table 3 | WN18RR    | k={int(r*100)}%",
             m("--data", "WN18RR", "--uniform_geometry", "--tau", "0", "--prune_topk_orc", str(r)))
            for r in [0.0, 0.2, 0.4, 0.6]
        ],
    }
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--only", choices=["table1", "table2", "tau", "table3"], default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    groups = build(args.gpu)
    names = [args.only] if args.only else ["table1", "table2", "tau", "table3"]

    for name in names:
        for label, cmd in groups[name]:
            print(f"\n### {label}\n{' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd)


if __name__ == "__main__":
    main()