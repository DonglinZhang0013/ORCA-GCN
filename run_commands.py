"""One-stop reproduction command list for ORCA-GCN.

Each entry maps to a table/figure in the paper. Run all groups, one group,
or print commands without executing them.

Examples:
    python run_commands.py --gpu 0
    python run_commands.py --gpu 0 --only table2
    python run_commands.py --dry_run

Prerequisites:
    FB15k-237/ and WN18RR/ contain aligned graph_with_olliverRicci.npy
    and olliverRicci.npy arrays.
"""
import argparse
import subprocess

BASE = ["python", "main.py", "--score_func", "conve", "--opn", "ccorr"]


def build(gpu):
    def m(*extra):
        return BASE + ["--gpu", str(gpu)] + list(extra)

    groups = {
        "table1": [
            ("Table 1 | FB15k-237 | ORCA-GCN", m("--data", "FB15k-237", "--tau", "0.6")),
            ("Table 1 | WN18RR | ORCA-GCN", m("--data", "WN18RR", "--tau", "0.6")),
        ],
        "table2": [
            ("Table 2 | ORCA-GCN (full)", m("--data", "FB15k-237", "--tau", "0.6")),
            ("Table 2 | w/o Geometry-Aware", m("--data", "FB15k-237", "--tau", "0.6", "--uniform_geometry")),
            ("Table 2 | w/o Semantic-Aware", m("--data", "FB15k-237", "--tau", "0")),
            ("Table 2 | w/ Random Curvature", m("--data", "FB15k-237", "--tau", "0.6", "--random_curvature")),
        ],
        "tau": [
            ("Tau sweep | FB15k-237", ["python", "run_tau_sweep.py", "--data", "FB15k-237", "--gpu", str(gpu), "--taus", "0.2", "0.4", "0.6", "0.8", "1.0"]),
            ("Tau sweep | WN18RR", ["python", "run_tau_sweep.py", "--data", "WN18RR", "--gpu", str(gpu), "--taus", "0.2", "0.4", "0.6", "0.8", "1.0"]),
        ],
        "table3": [
            (f"Table 3 | FB15k-237 | k={int(r * 100)}%", m("--data", "FB15k-237", "--uniform_geometry", "--tau", "0", "--prune_topk_orc", str(r)))
            for r in [0.0, 0.2, 0.4, 0.6]
        ] + [
            (f"Table 3 | WN18RR | k={int(r * 100)}%", m("--data", "WN18RR", "--uniform_geometry", "--tau", "0", "--prune_topk_orc", str(r)))
            for r in [0.0, 0.2, 0.4, 0.6]
        ],
    }
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only", choices=["table1", "table2", "tau", "table3"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    groups = build(args.gpu)
    names = [args.only] if args.only else ["table1", "table2", "tau", "table3"]
    for name in names:
        for label, cmd in groups[name]:
            print(f"\n### {label}\n{' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
