"""
Export the training edge list with the top-k% highest-ORC edges removed.

This is the edge-selection step behind Table 3 / Figure 5 (removing redundant
high-curvature edges). The resulting <data>/filtered_edges_top{ratio}.txt can be
fed to a base model as its aggregation graph.

For running the pruning *inside* this codebase directly, use instead:
    python main.py --data FB15k-237 --prune_topk_orc 0.4 ...

Usage:
    python get_deleted_train_based_on_orc.py --data FB15k-237 --ratio 0.4
"""

import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="FB15k-237", help="Dataset folder name")
    p.add_argument("--ratio", type=float, default=0.4,
                   help="Fraction (0-1) of highest-ORC edges to remove")
    args = p.parse_args()

    edge_index_type = np.load(f"./{args.data}/graph_with_olliverRicci.npy")
    orc = np.load(f"./{args.data}/olliverRicci.npy")

    id2ent = {i: l.strip() for i, l in enumerate(open(f"./{args.data}/entities.txt"))}
    id2rel = {i: l.strip() for i, l in enumerate(open(f"./{args.data}/relations.txt"))}

    num_edges = orc.shape[0]
    k = int(num_edges * args.ratio)
    gate = np.partition(orc, -k)[-k] if k > 0 else np.inf  # k-th largest ORC

    out_path = f"./{args.data}/filtered_edges_top{args.ratio}.txt"
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for col in range(num_edges):
            if orc[col] < gate:
                head = id2ent[int(edge_index_type[0][col])]
                tail = id2ent[int(edge_index_type[1][col])]
                relation = id2rel[int(edge_index_type[2][col])]
                f.write(f"{head}\t{relation}\t{tail}\n")
                kept += 1
    print(f"gate (k-th largest ORC): {gate}")
    print(f"removed {num_edges - kept} / {num_edges} edges, kept {kept} -> {out_path}")


if __name__ == "__main__":
    main()