# Reproducibility guide

## Paper-to-command map

| Manuscript item | Command |
|---|---|
| Main ORCA-GCN result, FB15k-237 | python main.py --data FB15k-237 --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0 |
| Main ORCA-GCN result, WN18RR | python main.py --data WN18RR --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0 |
| CompGCN-style both-modules-off baseline | add --uniform_geometry --tau 0 |
| Without geometry-aware layer | add --uniform_geometry |
| Without semantic-divergence layer | set --tau 0 |
| Random-curvature control | add --random_curvature |
| Tau sensitivity | python run_tau_sweep.py ... |
| Top-k curvature pruning | add --uniform_geometry --tau 0 --prune_topk_orc K |
| Training/evaluation timing | add --profile_runtime |
| Controlled pruning runtime | bash scripts/run_pruning_runtime_benchmark.sh DATASET GPU SEED EPOCHS |

## Evaluation protocol

Evaluation performs both tail prediction (h,r,?) and head prediction (?,r,t)
and averages the two directions. All other known true triples from train,
validation, and test are filtered before ranking. Reported metrics are MR,
MRR, Hits@1, Hits@3, and Hits@10.

## Five-run results

The final manuscript reports five-run means and sample standard deviations
for CompGCN and ORCA-GCN on FB15k-237 and WN18RR. Checkpoints are selected
using validation MRR before test evaluation. Aggregate values from Table 1
are provided in results/paper_summary/main_results_five_run_summary.csv.

This CSV is a transcription of the manuscript's aggregate results, not the
five original per-run logs. Individual seed identifiers and raw run logs
should only be added from the original experiment records; they must not be
inferred from the example/default seed.

By default, the training script seeds NumPy and PyTorch CPU RNGs to preserve
the original training path. --deterministic additionally seeds CUDA and
enables deterministic cuDNN behavior, but can change numerical results and
speed.

## Outputs

A normal run writes a checkpoint at <DATASET>/comp_link_<DATASET>, validation-
improving result blocks at <DATASET>/results_<DATASET>.txt, and final metrics
to standard output.

With --profile_runtime, one row is appended to results/time_cost.csv,
including average epoch training time, validation time, best-checkpoint test
time, final test time, graph edge counts, seed, and ablation settings.

Tau sweeps write sweep_logs/ and tau_sensitivity_<DATASET>.csv.

## Runtime measurements

Use the same machine and GPU for all configurations in a runtime comparison.
The profiling code synchronizes CUDA around timed regions. Total fitting time
should not be compared without noting that early stopping can occur at
different epochs; average per-epoch training time is the cleaner recurrent-cost
measure.

## Scalability boundary

Cosine dissimilarity is computed only for existing directed message edges.
It therefore costs O(|E|d), not O(|V|^2 d). The implementation does not
construct a full entity-by-entity similarity matrix. Web-scale KGs may still
need neighborhood sampling, partitioning, or distributed training.

## Exact versus approximate ORC

For the reported FB15k-237 and WN18RR arrays, the bundled implementation uses
ot.emd for local optimal transport and evaluates every non-loop directed
entity-pair edge. It does not use global edge sampling, ATD, Sinkhorn transport,
or graph coarsening.

A local support cap of 3,000 weighted neighbors remains enabled. It is inactive
on WN18RR and affects about 0.51% of FB15k-237 entity-pair edge computations.
Larger KGs may require approximate transport, sampling, partitioning, or
distributed preprocessing; those strategies are not evaluated in the current
paper.
