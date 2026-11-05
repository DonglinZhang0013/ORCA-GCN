# ORCA-GCN

Official implementation and reproducibility resources for:

**Beyond Geometric Blindness: Leveraging Ollivier–Ricci Curvature for Effective Knowledge Graph Completion**

ORCA-GCN incorporates edge-level Ollivier–Ricci curvature (ORC) into relation-aware message aggregation for knowledge graph completion (KGC). It combines a first-stage geometry-aware aggregation with a second-stage node-representation-divergence aggregation.

## Method at a glance

1. **Geometry-aware aggregation:** ORC values are converted into destination-neighborhood attention weights using softmax(1 - kappa). Lower-curvature edges receive larger relative weights.
2. **Semantic-divergence aggregation:** cosine dissimilarity is computed only on existing directed message-passing edges and normalized within each destination neighborhood. Relation-aware composition remains part of the message function.
3. **KGC scoring:** entity and relation representations are used by the ConvE decoder.

The method does not introduce a new general theory of graph curvature. Its contribution is a KGC-oriented use of edge-level geometry within relation-aware message passing.

## Repository contents

- main.py, models.py, data_loader.py, utils.py: model and training implementation.
- scripts/: ORC preprocessing, data validation, and runtime utilities.
- examples/: quick-start, ablation, and pruning commands.
- docs/: data-format and reproducibility notes.
- results/reviewer_analysis/: controlled pruning-runtime summaries.
- FB15k-237/ and WN18RR/: benchmark data and aligned precomputed ORC arrays (where included in the repository release).

## Environment

The provided requirements describe a compatible environment based on Python 3.9, PyTorch 1.12.1, and DGL 0.9.1. Install PyTorch and DGL builds compatible with your CUDA version first, then install the remaining dependencies:

    pip install -r requirements.txt

Optional dependencies for recomputing ORC are listed separately:

    pip install -r requirements-preprocess.txt

## Validate data and ORC arrays

    python scripts/check_data.py --data FB15k-237
    python scripts/check_data.py --data WN18RR

Expected forward-training counts:

| Dataset | Entities | Relations | Training triples |
|---|---:|---:|---:|
| FB15k-237 | 14,541 | 237 | 272,115 |
| WN18RR | 40,943 | 11 | 86,835 |

See docs/DATA_AND_ORC.md.

## Quick start

    bash examples/quick_start.sh 0

Equivalent command:

    python main.py --data FB15k-237 --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0

Use --gpu -1 for CPU execution.

## Main experiments and ablations

Main model runs:

    python main.py --data FB15k-237 --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0
    python main.py --data WN18RR    --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu 0

Ablation examples on FB15k-237:

    # Full ORCA-GCN
    python main.py --data FB15k-237 --tau 0.6 --seed 41504 --gpu 0

    # Without geometry-aware attention
    python main.py --data FB15k-237 --tau 0.6 --uniform_geometry --seed 41504 --gpu 0

    # Without semantic-divergence attention
    python main.py --data FB15k-237 --tau 0 --seed 41504 --gpu 0

    # Random-curvature control
    python main.py --data FB15k-237 --tau 0.6 --random_curvature --seed 41504 --gpu 0

The manuscript's principal CompGCN/ORCA-GCN comparison reports five-run means and sample standard deviations. The released command examples provide individual-run entry points; do not infer or invent the five experimental seeds from the default seed alone. See docs/REPRODUCIBILITY.md for this distinction.

## ORC preprocessing

Precomputed ORC arrays are reused during training. To inspect the preprocessing graph construction without overwriting data:

    python scripts/compute_orc.py --data FB15k-237 --dry-run

To recompute arrays, install the optional preprocessing dependencies and run:

    python scripts/compute_orc.py --data FB15k-237 --alpha 0 --proc 16 --overwrite

The preprocessing maps parallel relation-specific triples to weighted directed entity-pair edges, computes pair-level ORC, maps values back to the forward triples, and assigns the documented +1.0 convention to relation-specific self-loop triples. Inverse message edges inherit the corresponding forward-edge curvature.

## Runtime and pruning analysis

Add --profile_runtime to a training command to record synchronized runtime measurements:

    python main.py --data FB15k-237 --tau 0.6 --seed 41504 --gpu 0 --profile_runtime

The controlled pruning runtime benchmark is run with:

    bash scripts/run_pruning_runtime_benchmark.sh FB15k-237 0 41504 30
    bash scripts/run_pruning_runtime_benchmark.sh WN18RR 0 41504 30

Important: the reported 30-epoch pruning-runtime benchmark measures the CompGCN backbone under curvature-guided edge pruning. Its runtime results are separate from the full-training link-prediction results; they should not be described as a direct acceleration measurement of the complete ORCA-GCN architecture.

## Reproducibility and limitations

The repository documents the reported preprocessing method, local support safeguard, data-array alignment, evaluation protocol, and runtime measurements. The current paper evaluates FB15k-237 and WN18RR; these experiments do not establish web-scale scalability. Larger KGs may require sampling, partitioning, approximate transport, or distributed preprocessing, which are not evaluated in the paper.
