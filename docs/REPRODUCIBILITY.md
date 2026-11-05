# Reproducibility Guide

## Manuscript-to-code map

| Manuscript component | Repository entry point |
|---|---|
| Main ORCA-GCN training | main.py |
| Geometry-aware and semantic-divergence aggregation | models.py |
| KG graph construction and data loading | data_loader.py |
| ORC recomputation | scripts/compute_orc.py |
| Data and ORC-array validation | scripts/check_data.py |
| Tau sensitivity | run_tau_sweep.py |
| Curvature-guided pruning | get_deleted_train_based_on_orc.py, examples/run_pruning.sh |
| Controlled pruning-runtime benchmark | scripts/run_pruning_runtime_benchmark.sh |

## Evaluation protocol

The evaluation code performs head and tail prediction and applies filtered ranking against known true triples. The manuscript reports MRR, Hits@1, Hits@3, and Hits@10.

For the principal CompGCN/ORCA-GCN comparison, the manuscript reports mean and sample standard deviation over five runs, with checkpoint selection based on validation MRR before test evaluation.

**Important release note:** the manuscript's aggregate five-run statistics are reported in the paper. The currently documented single-run commands use seed 41504 as an example/default entry point. The five individual seed identifiers and per-run metric logs are not specified in this repository note; they should be added only from the original experiment logs, not reconstructed or guessed.

## Main run commands

    python main.py --data FB15k-237 --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0
    python main.py --data WN18RR    --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0

For CPU, use --gpu -1. Install the PyTorch and DGL builds compatible with the local CUDA version before installing the remaining requirements.

## Data and ORC

The repository's data checker verifies entity/relation identifiers, NumPy array shapes, forward-triple multiset consistency, edge-array/ORC alignment, finite curvature values, and the self-loop convention:

    python scripts/check_data.py --data FB15k-237
    python scripts/check_data.py --data WN18RR

ORC is computed offline and reused during training. The bundled preprocessing implementation uses exact local optimal transport through ot.emd, with a 3,000-neighbor local-support safeguard. This safeguard is inactive on WN18RR and affects approximately 0.51% of FB15k-237 entity-pair edge computations.

## Runtime measurements

Use the same machine and GPU for all runtime comparisons. The pruning-runtime benchmark uses controlled 30-epoch runs on the CompGCN backbone. Its runtime measurements are distinct from the full-training pruning performance experiments.

    bash scripts/run_pruning_runtime_benchmark.sh FB15k-237 0 41504 30
    bash scripts/run_pruning_runtime_benchmark.sh WN18RR 0 41504 30

## Scope and limitations

The experiments cover FB15k-237 and WN18RR. The repository's preprocessing and runtime analyses do not replace evaluation on a third substantially larger KG and do not establish web-scale scalability. Approximate transport, graph sampling, partitioning, and distributed preprocessing are possible future directions, not evaluated results.
