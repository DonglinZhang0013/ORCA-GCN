# Reviewer-requested runtime analyses

The two CSV files report controlled 30-epoch pruning runtime benchmarks using
seed 41504. Configurations for one dataset were executed sequentially on the
same machine and GPU. These short runs are used only for runtime and edge-count
analysis; the manuscript's pruning MRR/Hits values come from the full-training
experiments.

Run from the repository root:

    bash scripts/run_pruning_runtime_benchmark.sh FB15k-237 0 41504 30
    bash scripts/run_pruning_runtime_benchmark.sh WN18RR 0 41504 30

The runtime benchmark measures the CompGCN backbone under curvature-guided
pruning. It is not a direct acceleration measurement of the complete ORCA-GCN
architecture.
