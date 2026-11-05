# ORCA-GCN

Official code and data package for:

> **Beyond Geometric Blindness: Leveraging Ollivier–Ricci Curvature for Effective Knowledge Graph Completion**

ORCA-GCN operationalizes edge-level Ollivier–Ricci curvature inside
relation-aware knowledge-graph message passing. The method does **not** redefine
graph curvature. Its contribution is the use of a fixed edge-geometric prior in
the first aggregation layer, followed by an edge-wise representation-divergence
signal in the second layer.

## Architecture

1. **Geometry-aware layer.** For every directed message edge, the loader converts
   ORC into destination-neighborhood attention using `softmax(1 - kappa)`.
   Lower-curvature bridge-like edges receive larger relative weights, while
   high-curvature locally redundant edges receive smaller weights.
2. **Semantic-divergence layer.** The second layer computes
   `tau * (1 - cosine(h_source, h_target))` only on existing message edges and
   applies another destination-neighborhood softmax. Relation information remains
   in the CompGCN message composition; the cosine score itself is node-level.
3. **Decoder.** Updated entity and relation representations are scored by ConvE.

## Repository layout

```text
ORCA-GCN/
├── README.md
├── requirements.txt
├── requirements-preprocess.txt
├── main.py
├── models.py
├── data_loader.py
├── utils.py
├── run_commands.py
├── run_tau_sweep.py
├── get_deleted_train_based_on_orc.py
├── FB15k-237/
├── WN18RR/
│   └── raw/graph_with_olliverRicci.gml
├── paperGraphRicciCurvature/
├── scripts/
│   ├── check_data.py
│   ├── compute_orc.py
│   ├── build_wn18rr_orc_npy.py
│   ├── organize_data.py
│   ├── run_pruning_runtime_benchmark.sh
│   └── summarize_pruning_runtime.py
├── examples/
├── docs/
└── results/reviewer_analysis/
```

The two benchmark directories already contain aligned, precomputed ORC arrays,
so model training does not require recomputing curvature.

## 1. Environment

Recommended compatible stack:

- Python 3.9
- PyTorch 1.12.1
- DGL 0.9.1
- NumPy 1.x

Install PyTorch and DGL builds matching the local CUDA version first, then:

```bash
pip install -r requirements.txt
```

Weights & Biases is optional and disabled unless `--use_wandb` is supplied.

For optional ORC recomputation:

```bash
pip install -r requirements-preprocess.txt
```

## 2. Validate the supplied datasets

```bash
python scripts/check_data.py --data FB15k-237
python scripts/check_data.py --data WN18RR
```

Expected training/ORC counts:

| Dataset | Entities | Relations | Forward training triples |
|---|---:|---:|---:|
| FB15k-237 | 14,541 | 237 | 272,115 |
| WN18RR | 40,943 | 11 | 86,835 |

Detailed file semantics, parallel-edge handling, inverse-edge reuse, and the
self-loop convention are documented in [`docs/DATA_AND_ORC.md`](docs/DATA_AND_ORC.md).

## 3. Quick start

```bash
bash examples/quick_start.sh 0
```

Equivalent direct command:

```bash
python main.py \
  --data FB15k-237 \
  --score_func conve \
  --opn ccorr \
  --tau 0.6 \
  --seed 41504 \
  --gpu 0
```

For CPU, set `--gpu -1`.

## 4. Reproduce manuscript experiments

Print all commands without running:

```bash
python run_commands.py --dry_run
```

Run one experiment group:

```bash
python run_commands.py --gpu 0 --only table1
python run_commands.py --gpu 0 --only table2
python run_commands.py --gpu 0 --only tau
python run_commands.py --gpu 0 --only table3
```

### Main ORCA-GCN results

```bash
python main.py --data FB15k-237 --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0
python main.py --data WN18RR    --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0
```

### Ablations

```bash
# Full ORCA-GCN
python main.py --data FB15k-237 --tau 0.6 --seed 41504 --gpu 0

# Without geometry-aware attention
python main.py --data FB15k-237 --tau 0.6 --uniform_geometry --seed 41504 --gpu 0

# Without semantic-divergence attention
python main.py --data FB15k-237 --tau 0 --seed 41504 --gpu 0

# Random-curvature control
python main.py --data FB15k-237 --tau 0.6 --random_curvature --seed 41504 --gpu 0
```

`tau=1.0` is the unscaled `softmax(1-cos)` form; `tau=0` yields uniform
second-layer attention.

### Tau sensitivity

```bash
python run_tau_sweep.py --data FB15k-237 --gpu 0 --taus 0.2 0.4 0.6 0.8 1.0
python run_tau_sweep.py --data WN18RR    --gpu 0 --taus 0.2 0.4 0.6 0.8 1.0
```

### Top-k ORC pruning

The pruning study applies the base both-modules-off configuration and removes
the highest-curvature forward edges from the aggregation graph:

```bash
bash examples/run_pruning.sh FB15k-237 0
```

A single 40% run is:

```bash
python main.py \
  --data FB15k-237 \
  --uniform_geometry \
  --tau 0 \
  --prune_topk_orc 0.4 \
  --seed 41504 \
  --gpu 0
```

To export the retained forward triples for another model:

```bash
python get_deleted_train_based_on_orc.py --data FB15k-237 --ratio 0.4
```

## 5. Training and evaluation timing

Add `--profile_runtime` to any run:

```bash
python main.py \
  --data FB15k-237 \
  --tau 0.6 \
  --seed 41504 \
  --gpu 0 \
  --profile_runtime
```

CUDA is synchronized around timed regions. The run appends a row to
`results/time_cost.csv` with:

- average training time per epoch;
- average validation evaluation time;
- average test time at validation improvements;
- final test evaluation time;
- total fitting time;
- forward and directed message-edge counts;
- seed and ablation settings.

Controlled pruning benchmark:

```bash
bash scripts/run_pruning_runtime_benchmark.sh FB15k-237 0 41504 30
bash scripts/run_pruning_runtime_benchmark.sh WN18RR    0 41504 30
```

The reviewer-requested benchmark summaries are included under
`results/reviewer_analysis/`.

## 6. ORC preprocessing

Precomputed ORC is supplied and reused throughout training. To validate the
construction without overwriting files:

```bash
python scripts/compute_orc.py --data FB15k-237 --dry_run
```

To recompute:

```bash
python scripts/compute_orc.py \
  --data FB15k-237 \
  --alpha 0 \
  --proc 16 \
  --overwrite
```

The script:

1. reads original directed forward training triples;
2. excludes entity self-loops from transport;
3. collapses parallel relation-specific triples by ordered entity pair;
4. assigns collapsed edge weight `1 / multiplicity`;
5. computes ORC;
6. maps pair-level ORC back to every forward triple;
7. assigns self-loop triples `+1.0`;
8. writes aligned NumPy arrays and a JSON timing report.

Inverse message edges are generated during data loading and inherit the forward
ORC value. ORC is never recomputed per epoch.

### Transport solver used in the reported preprocessing

The bundled curvature implementation directly calls `ot.emd` for the local
Earth Mover's Distance problem. The reported arrays were not produced through
edge subsampling, graph coarsening, ATD, or Sinkhorn transport.

The implementation retains `nbr_topk=3000` as a safeguard for unusually large
local probability supports. It is inactive on WN18RR. On FB15k-237, it is
triggered for two high-in-degree source entities and affects 1,240 of the
245,072 non-loop entity-pair edge computations (about 0.51%). All other edge
computations use their full local neighborhood support.

These details matter because preprocessing time depends on local support sizes
and shortest-path/transport structure, not only on the number of KG triples.
The supplied timings are benchmark-specific full-graph measurements and should
not be interpreted as evidence of linear scaling to web-scale KGs.

For WN18RR, the bundled raw GML can also be converted back into aligned arrays:

```bash
python scripts/build_wn18rr_orc_npy.py
```

## 7. Output files

| Output | Description |
|---|---|
| `<DATASET>/comp_link_<DATASET>` | Best checkpoint selected by validation MRR |
| `<DATASET>/results_<DATASET>.txt` | Validation-improving metric blocks |
| `results/time_cost.csv` | Optional structured runtime log |
| `sweep_logs/*.log` | Tau-sweep stdout |
| `tau_sensitivity_<DATASET>.csv` | Tau-sweep metric summary |
| `<DATASET>/filtered_edges_top<RATIO>.txt` | Optional exported pruned triples |
| `<DATASET>/orc_preprocessing_timing.json` | Optional ORC preprocessing breakdown |

## 8. Complexity and scalability

The semantic-divergence layer gathers source and destination representations
only for existing directed message edges. Its extra computation is
`O(|E|d)` and its attention storage is linear in `|E|`; no `|V| x |V|`
similarity matrix is formed.

The included experiments use medium-scale benchmarks. Larger KGs may require
sampling, partitioning, or distributed training. Curvature-guided pruning is an
optional way to reduce recurrent aggregation cost; it is not claimed to prove
web-scale scalability on its own.

## 9. Reproducibility notes

- Evaluation uses filtered head and tail ranking and averages both directions.
- Point-estimate runs use the stated seed; multi-seed statistics must only be
  reported after all runs finish.
- Runtime configurations must be compared on the same machine and GPU.
- Total fitting time is affected by different early-stopping epochs; average
  per-epoch time is the cleaner recurrent-cost comparison.
- `--deterministic` is opt-in because deterministic CUDA behavior can change
  both speed and numerical results relative to the original runs.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for a paper-to-command
map and further details.

## 10. Key arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--data` | `FB15k-237` | Dataset directory |
| `--seed` | `41504` | Run seed |
| `--tau` | `0.6` | Second-layer divergence-logit scale |
| `--uniform_geometry` | off | Replace ORC attention with uniform first-layer weights |
| `--random_curvature` | off | Replace ORC with uniform noise in `[-2,1]` |
| `--prune_topk_orc` | `0.0` | Fraction of highest-ORC forward edges removed |
| `--eval_every` | `5` | Validation interval in epochs |
| `--patience` | `40` | Early-stop patience in validation checks |
| `--profile_runtime` | off | Write synchronized runtime statistics |
| `--use_wandb` | off | Enable W&B logging |
| `--gpu` | `0` | GPU ID; `-1` uses CPU |

## 11. Troubleshooting

**DGL/CUDA mismatch:** install PyTorch first, then a DGL wheel built for the
same CUDA version.

**Missing ORC arrays:** run `python scripts/check_data.py --data DATASET`.
For WN18RR, `scripts/build_wn18rr_orc_npy.py` can reconstruct them from the
bundled raw GML.

**Out of memory during evaluation:** reduce `--batch`; evaluation scores all
candidate entities and can be more memory-intensive than graph aggregation.

**Different result after enabling `--deterministic`:** this option changes the
CUDA execution path and was not used for the original point estimates.