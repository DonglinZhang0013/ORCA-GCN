#!/usr/bin/env python3
"""Summarize controlled top-k pruning runtime benchmarks."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=41504)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    path = args.project_root / "results" / "time_cost.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing runtime file: {path}")
    df = pd.read_csv(path)
    required = {"timestamp", "status", "dataset", "seed", "max_epochs",
                "uniform_geometry", "tau", "prune_topk_orc", "edge_count_raw",
                "edge_count_after_pruning", "num_edges", "avg_epoch_train_sec",
                "avg_valid_eval_sec", "avg_best_test_eval_sec"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"time_cost.csv is missing columns: {missing}")
    mask = (df["status"].eq("COMPLETED") & df["dataset"].eq(args.dataset)
            & df["seed"].eq(args.seed) & df["max_epochs"].eq(args.epochs)
            & as_bool(df["uniform_geometry"]) & df["tau"].fillna(-1).eq(0)
            & df["prune_topk_orc"].isin([0.0, 0.2, 0.4, 0.6]))
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError("No completed benchmark rows matched the requested configuration.")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = (out.sort_values("timestamp").groupby("prune_topk_orc", as_index=False)
           .tail(1).sort_values("prune_topk_orc"))
    if set(out["prune_topk_orc"]) != {0.0, 0.2, 0.4, 0.6}:
        raise RuntimeError(f"Expected four pruning levels; found {sorted(out['prune_topk_orc'].tolist())}")
    baseline = float(out.loc[out["prune_topk_orc"].eq(0.0), "avg_epoch_train_sec"].iloc[0])
    out["pruned_percent"] = (100 * out["prune_topk_orc"]).round().astype(int)
    out["remaining_forward_edges_percent"] = 100 * out["edge_count_after_pruning"] / out["edge_count_raw"]
    out["train_time_reduction_percent"] = 100 * (baseline - out["avg_epoch_train_sec"]) / baseline
    cols = ["pruned_percent", "edge_count_after_pruning", "num_edges",
            "remaining_forward_edges_percent", "avg_epoch_train_sec",
            "train_time_reduction_percent", "avg_valid_eval_sec", "avg_best_test_eval_sec"]
    result = out[cols].copy()
    output_path = args.project_root / "results" / f"pruning_runtime_summary_{args.dataset}_seed{args.seed}_{args.epochs}ep.csv"
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved: {output_path}")
    print("Use these rows for runtime and edge-count analysis only; retain full-run MRR/Hits values in the paper.")

if __name__ == "__main__":
    main()
