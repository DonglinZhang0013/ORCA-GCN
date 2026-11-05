# Reproducibility guide

## Paper-to-command map

| Manuscript item | Repository entry point |
|---|---|
| Main ORCA-GCN training | main.py |
| Geometry-aware and semantic-divergence aggregation | models.py |
| KG graph construction and data loading | data_loader.py |
| ORC recomputation | scripts/compute_orc.py |
| Dataset and ORC-array validation | scripts/check_data.py |
| Tau sensitivity | run_tau_sweep.py |
| Curvature-guided pruning | get_deleted_train_based_on_orc.py and examples/run_pruning.sh |
| Controlled pruning-runtime benchmark | scripts/run_pruning_runtime_benchmark.sh |

## Evaluation protocol

Evaluation performs tail prediction (h,r,?) and head prediction (?,r,t), then averages both directions. Known true triples from the train, validation, and test splits are filtered before ranking. The manuscript reports MRR, Hits@1, Hits@3, and Hits@10.

## Main repeated-run results

The final manuscript reports five-run mean and sample standard deviation for CompGCN and ORCA-GCN on FB15k-237 and WN18RR. Checkpoints are selected using validation MRR before test evaluation.

The aggregate values are reproduced in results/paper_summary/main_results_five_run_summary.csv. This summary is a transcription of the final manuscript's Table 1, not a substitute for the five original per-run logs. The individual seed identifiers and raw per-run logs should be added only when available from the original experiment records; they must not be inferred from the default command-line seed.

## Example single-run commands

    python main.py --data FB15k-237 --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0
    python main.py --data WN18RR --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0

For CPU, use --gpu -1. Install PyTorch and DGL builds compatible with the local CUDA version before installing requirements.txt.

## Data validation

    python scripts/check_data.py --data FB15k-237
    python scripts/check_data.py --data WN18RR

The checker verifies entity/relation identifiers, NumPy array shapes, forward-triple multiset consistency, edge-array/ORC alignment, finite curvature values, and the self-loop convention.

## ORC preprocessing

ORC is computed offline and reused during training. Inspect the preprocessing plan without writing arrays:

    python scripts/compute_orc.py --data FB15k-237 --dry_run

To recompute arrays (this overwrites existing arrays only when --overwrite is supplied):

    python scripts/compute_orc.py --data FB15k-237 --alpha 0 --proc 16 --overwrite

The bundled implementation uses ot.emd for local optimal transport and includes a local support cap of 3,000 weighted neighbors. This safeguard is inactive on WN18RR and affects about 0.51% of FB15k-237 entity-pair edge computations.

## Ablation and sensitivity examples

    bash examples/run_ablation.sh FB15k-237 0
    python run_tau_sweep.py --data FB15k-237 --gpu 0 --taus 0.2 0.4 0.6 0.8 1.0

## Pruning and runtime

    bash examples/run_pruning.sh FB15k-237 0

The controlled 30-epoch runtime script uses a project-root environment variable. From the repository root:

    PROJECT_ROOT="$PWD" bash scripts/run_pruning_runtime_benchmark.sh FB15k-237 0 41504 30
    PROJECT_ROOT="$PWD" bash scripts/run_pruning_runtime_benchmark.sh WN18RR 0 41504 30

The pruning-runtime benchmark measures the CompGCN backbone under curvature-guided pruning. Its runtime measurements are separate from the full-training link-prediction results and should not be described as direct acceleration measurements of the complete ORCA-GCN architecture.

## Outputs

A normal run writes a best checkpoint and validation-improving metric blocks under the dataset directory. With --profile_runtime, structured timing information is appended to results/time_cost.csv. Tau sweeps write sweep_logs/ and tau_sensitivity_<DATASET>.csv.

## Scalability boundary

Cosine dissimilarity is computed only for existing directed message edges, requiring O(|E|d) time and O(|E|) attention storage rather than a dense O(|V|^2) entity-similarity matrix. The experiments cover FB15k-237 and WN18RR and do not establish web-scale scalability. Sampling, partitioning, approximate transport, or distributed preprocessing are potential future directions, not evaluated results.
