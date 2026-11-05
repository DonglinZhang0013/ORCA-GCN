"""ORCA-GCN — training and evaluation entry point.

This is a structural refactor of the original training script. The training
protocol and all metric computations are byte-for-byte equivalent to the
original: same loss (BCE), optimizer (Adam), filtered ranking evaluation,
validation cadence, early-stopping rule, checkpoint path, and result-file
format. Only the code organization, logging, and configurability changed.

Reproducibility note: by default only NumPy and torch CPU RNGs are seeded
(exactly as before). Pass --deterministic to additionally seed CUDA and enable
cuDNN determinism; this changes results relative to the original runs, so it is
opt-in.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Dict

import numpy as np
import torch as th
from torch import nn, optim
from tqdm import tqdm

from data_loader import Data
from models import ORCAGCN_ConvE
from utils import in_out_norm

os.environ.setdefault("WANDB_MODE", "offline")

KS = (1, 3, 10)  # Hits@k reported


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed RNGs. Default matches the original script (NumPy + torch)."""
    np.random.seed(seed)
    th.manual_seed(seed)
    if deterministic:
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def get_device(gpu: int) -> str:
    """Resolve the requested device, falling back to CPU when needed."""
    if gpu >= 0 and th.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def sync_device(device: str) -> None:
    """Synchronize CUDA before wall-clock timing; no-op on CPU."""
    if str(device).startswith("cuda") and th.cuda.is_available():
        th.cuda.synchronize()


class WandbLogger:
    """Thin wrapper: logs to Weights & Biases if enabled, else a no-op."""

    def __init__(self, enabled: bool, args: argparse.Namespace):
        self.run = None
        if enabled:
            try:
                import wandb

                self.run = wandb.init(
                    project="orca", name=f"tau={args.tau}_{args.dataset}"
                )
            except Exception as exc:  # pragma: no cover
                print(f"[wandb] disabled ({exc})")

    def log(self, data: Dict, step: int) -> None:
        if self.run is not None:
            self.run.log(data, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


# --------------------------------------------------------------------------- #
# Evaluation (filtered ranking) — numerically identical to the original
# --------------------------------------------------------------------------- #
@th.no_grad()
def _rank_one_direction(model, graph, device, data_iter, split, mode) -> Dict:
    """Filtered ranks for one direction (tail or head). Returns summed stats."""
    model.eval()
    results: Dict[str, float] = {}
    loader = data_iter[f"{split}_{mode}"]
    for batch in tqdm(loader, desc=f"{split},{mode}"):
        triple, label = batch[0].to(device), batch[1].to(device)
        sub, rel, obj = triple[:, 0], triple[:, 1], triple[:, 2]

        pred = model(graph, sub, rel)
        b_range = th.arange(pred.size(0), device=device)
        target_pred = pred[b_range, obj]
        # filtered setting: mask all known-true entities, then restore target
        pred = th.where(label.bool(), -th.ones_like(pred) * 1e7, pred)
        pred[b_range, obj] = target_pred

        ranks = 1 + th.argsort(
            th.argsort(pred, dim=1, descending=True), dim=1, descending=False
        )[b_range, obj]
        ranks = ranks.float()

        results["count"] = th.numel(ranks) + results.get("count", 0.0)
        results["mr"] = th.sum(ranks).item() + results.get("mr", 0.0)
        results["mrr"] = th.sum(1.0 / ranks).item() + results.get("mrr", 0.0)
        for k in KS:
            results[f"hits@{k}"] = th.numel(ranks[ranks <= k]) + results.get(
                f"hits@{k}", 0.0
            )
    return results


@th.no_grad()
def evaluate(model, graph, device, data_iter, split: str = "valid") -> Dict:
    """Average tail (left) and head (right) predictions into final metrics."""
    left = _rank_one_direction(model, graph, device, data_iter, split, "tail")
    right = _rank_one_direction(model, graph, device, data_iter, split, "head")
    count = float(left["count"])

    res: Dict[str, float] = {}
    res["left_mr"] = round(left["mr"] / count, 5)
    res["left_mrr"] = round(left["mrr"] / count, 5)
    res["right_mr"] = round(right["mr"] / count, 5)
    res["right_mrr"] = round(right["mrr"] / count, 5)
    res["mr"] = round((left["mr"] + right["mr"]) / (2 * count), 5)
    res["mrr"] = round((left["mrr"] + right["mrr"]) / (2 * count), 5)
    for k in KS:
        res[f"left_hits@{k}"] = round(left[f"hits@{k}"] / count, 5)
        res[f"right_hits@{k}"] = round(right[f"hits@{k}"] / count, 5)
        res[f"hits@{k}"] = round(
            (left[f"hits@{k}"] + right[f"hits@{k}"]) / (2 * count), 5
        )
    return res


def format_metrics(tag: str, r: Dict) -> str:
    """Kept in the original layout (parsed by run_tau_sweep.py)."""
    return (
        "{} MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
            tag, r["mrr"], r["mr"], r["hits@10"], r["hits@3"], r["hits@1"]
        )
    )


# --------------------------------------------------------------------------- #
# Build data & model
# --------------------------------------------------------------------------- #
def build_data(args, device):
    """Load triples/ORC arrays and construct the directed DGL graph."""
    data = Data(
        args.dataset,
        args.lbl_smooth,
        args.num_workers,
        args.batch_size,
        random_curvature=args.random_curvature,
        prune_topk_orc=args.prune_topk_orc,
    )
    graph = data.g.to(device)
    num_rel = th.max(graph.edata["etype"]).item() + 1
    graph = in_out_norm(graph)
    print(f"device: {device} | num_ent: {graph.num_nodes()} | num_rel: {num_rel}")
    return data, graph, num_rel


def build_model(args, num_rel, num_ent, device):
    """Construct ORCA-GCN with the ConvE decoder."""
    model = ORCAGCN_ConvE(
        num_bases=args.num_bases,
        num_rel=num_rel,
        num_ent=num_ent,
        in_dim=args.init_dim,
        layer_size=args.layer_size,
        comp_fn=args.opn,
        batchnorm=True,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        num_filt=args.num_filt,
        hid_drop=args.hid_drop,
        feat_drop=args.feat_drop,
        ker_sz=args.ker_sz,
        k_w=args.k_w,
        k_h=args.k_h,
        tau=args.tau,
        uniform_geometry=args.uniform_geometry,
    )
    return model.to(device)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.gpu)
        self.logger = WandbLogger(args.use_wandb, args)

        self.data, self.graph, num_rel = build_data(args, self.device)
        self.data_iter = self.data.data_iter
        self.model = build_model(args, num_rel, self.graph.num_nodes(), self.device)

        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.l2
        )
        self.ckpt_path = os.path.join(args.dataset, f"comp_link_{args.dataset}")
        self.results_path = os.path.join(args.dataset, f"results_{args.dataset}.txt")
        self.epoch_train_times = []
        self.valid_eval_times = []
        self.best_test_eval_times = []
        self.final_test_eval_sec = None
        self.total_fit_sec = None
        self.best_epoch = None
        self.last_epoch = 0

    # ---- one training epoch --------------------------------------------- #
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        losses = []
        for batch in tqdm(self.data_iter["train"], desc=f"Epoch {epoch}"):
            triple, label = batch[0].to(self.device), batch[1].to(self.device)
            sub, rel = triple[:, 0], triple[:, 1]

            logits = self.model(self.graph, sub, rel)
            loss = self.loss_fn(logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))  # display only; training unaffected

    # ---- checkpoint + result logging ------------------------------------ #
    def _save_best(self, epoch, val_results, test_results):
        th.save(self.model.state_dict(), self.ckpt_path)
        with open(self.results_path, "a") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write("Valid:\n")
            for key, value in val_results.items():
                f.write(f"{key}: {value:.5f}\n")
            f.write("Test:\n")
            for key, value in test_results.items():
                f.write(f"{key}: {value:.5f}\n")
            f.write("----------------------------\n")

    # ---- full training loop --------------------------------------------- #
    def fit(self):
        """Train with filtered validation and early stopping.

        When ``--profile_runtime`` is enabled, CUDA-synchronized wall-clock
        measurements are collected for every training epoch and evaluation.
        Profiling is opt-in so the default experimental path remains unchanged.
        """
        args = self.args
        best_val_mrr, kill_cnt = 0.0, 0

        if args.profile_runtime:
            sync_device(self.device)
            fit_start = perf_counter()

        for epoch in range(1, args.max_epochs + 1):
            self.last_epoch = epoch

            if args.profile_runtime:
                sync_device(self.device)
                start = perf_counter()
            train_loss = self.train_one_epoch(epoch)
            if args.profile_runtime:
                sync_device(self.device)
                self.epoch_train_times.append(perf_counter() - start)

            self.logger.log({"train/loss": train_loss}, step=epoch)

            if epoch % args.eval_every != 0:
                continue

            if args.profile_runtime:
                sync_device(self.device)
                start = perf_counter()
            val_results = evaluate(
                self.model, self.graph, self.device, self.data_iter, "valid"
            )
            if args.profile_runtime:
                sync_device(self.device)
                self.valid_eval_times.append(perf_counter() - start)

            self.logger.log(
                {f"valid/{k}": val_results[k] for k in ("mrr", "hits@1", "hits@3", "hits@10")},
                step=epoch,
            )

            if val_results["mrr"] >= best_val_mrr:
                best_val_mrr = val_results["mrr"]
                self.best_epoch = epoch
                kill_cnt = 0

                if args.profile_runtime:
                    sync_device(self.device)
                    start = perf_counter()
                test_results = evaluate(
                    self.model, self.graph, self.device, self.data_iter, "test"
                )
                if args.profile_runtime:
                    sync_device(self.device)
                    self.best_test_eval_times.append(perf_counter() - start)

                self.logger.log(
                    {f"test/{k}": test_results[k] for k in ("mrr", "hits@1", "hits@3", "hits@10")},
                    step=epoch,
                )
                print("saving model...")
                self._save_best(epoch, val_results, test_results)
            else:
                kill_cnt += 1
                print(f"no improvement ({kill_cnt}/{args.patience})")
                if kill_cnt > args.patience:
                    print("early stop.")
                    break

            print(f"loss: {train_loss:.6f}")
            print(format_metrics("Valid", val_results))

        if args.profile_runtime:
            sync_device(self.device)
            self.total_fit_sec = perf_counter() - fit_start

    # ---- final test with the best checkpoint ---------------------------- #
    def test_best(self):
        """Evaluate the checkpoint selected by validation MRR."""
        self.model.load_state_dict(th.load(self.ckpt_path, map_location=self.device))
        self.model.eval()

        if self.args.profile_runtime:
            sync_device(self.device)
            start = perf_counter()
        test_results = evaluate(
            self.model, self.graph, self.device, self.data_iter, "test"
        )
        if self.args.profile_runtime:
            sync_device(self.device)
            self.final_test_eval_sec = perf_counter() - start
            self._write_runtime_summary()

        print(format_metrics("Test", test_results))
        self.logger.finish()
        return test_results

    def _write_runtime_summary(self) -> None:
        """Append one completed run to ``results/time_cost.csv``."""
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", "time_cost.csv")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETED",
            "dataset": self.args.dataset,
            "seed": self.args.seed,
            "max_epochs": self.args.max_epochs,
            "eval_every": self.args.eval_every,
            "patience": self.args.patience,
            "uniform_geometry": self.args.uniform_geometry,
            "tau": self.args.tau,
            "random_curvature": self.args.random_curvature,
            "prune_topk_orc": self.args.prune_topk_orc,
            "edge_count_raw": getattr(self.data, "edge_count_raw", ""),
            "edge_count_after_pruning": getattr(self.data, "edge_count_after_pruning", ""),
            "num_edges": self.graph.num_edges(),
            "total_fit_sec": self.total_fit_sec,
            "avg_epoch_train_sec": (
                sum(self.epoch_train_times) / len(self.epoch_train_times)
                if self.epoch_train_times else ""
            ),
            "avg_valid_eval_sec": (
                sum(self.valid_eval_times) / len(self.valid_eval_times)
                if self.valid_eval_times else ""
            ),
            "avg_best_test_eval_sec": (
                sum(self.best_test_eval_times) / len(self.best_test_eval_times)
                if self.best_test_eval_times else ""
            ),
            "final_test_eval_sec": self.final_test_eval_sec,
            "best_epoch": self.best_epoch,
            "last_epoch": self.last_epoch,
            "gpu": self.args.gpu,
            "device": self.device,
        }
        fields = list(row)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"runtime summary appended to {path}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ORCA-GCN training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data / run")
    g.add_argument("--data", dest="dataset", default="FB15k-237", help="Dataset name")
    g.add_argument("--model", dest="model", default="compgcn", help="Model name")
    g.add_argument("--score_func", dest="score_func", default="conve", help="Score function")
    g.add_argument("--opn", dest="opn", default="ccorr", help="Composition operator")
    g.add_argument("--gpu", type=int, default=0, help="GPU id (-1 for CPU)")
    g.add_argument("--seed", dest="seed", type=int, default=41504, help="Random seed")
    g.add_argument("--deterministic", action="store_true",
                   help="Also seed CUDA + cuDNN deterministic (changes results vs. original).")
    g.add_argument("--num_workers", type=int, default=10, help="DataLoader workers")

    g = p.add_argument_group("optimization")
    g.add_argument("--batch", dest="batch_size", type=int, default=512, help="Batch size")
    g.add_argument("--epoch", dest="max_epochs", type=int, default=3500, help="Max epochs")
    g.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    g.add_argument("--l2", type=float, default=0.0, help="L2 weight decay")
    g.add_argument("--lbl_smooth", dest="lbl_smooth", type=float, default=0.1, help="Label smoothing")
    g.add_argument("--eval_every", type=int, default=5, help="Validate every N epochs")
    g.add_argument("--patience", type=int, default=40, help="Early-stop patience (in validations)")

    g = p.add_argument_group("model")
    g.add_argument("--num_bases", dest="num_bases", type=int, default=-1, help="Basis relation vectors")
    g.add_argument("--init_dim", dest="init_dim", type=int, default=100, help="Initial embedding dim")
    g.add_argument("--layer_size", nargs="?", default="[200, 200]", help="Per-layer output sizes")
    g.add_argument("--gcn_drop", dest="dropout", type=float, default=0.2, help="GCN dropout")
    g.add_argument("--layer_dropout", nargs="?", default="[0.3, 0.3]", help="Per-layer dropout")

    g = p.add_argument_group("ConvE decoder")
    g.add_argument("--hid_drop", dest="hid_drop", type=float, default=0.3, help="Hidden dropout")
    g.add_argument("--feat_drop", dest="feat_drop", type=float, default=0.3, help="Feature dropout")
    g.add_argument("--k_w", dest="k_w", type=int, default=10, help="ConvE k_w")
    g.add_argument("--k_h", dest="k_h", type=int, default=20, help="ConvE k_h")
    g.add_argument("--num_filt", dest="num_filt", type=int, default=200, help="ConvE #filters")
    g.add_argument("--ker_sz", dest="ker_sz", type=int, default=7, help="ConvE kernel size")

    g = p.add_argument_group("ORCA-GCN specific / ablations")
    g.add_argument("--tau", dest="tau", type=float, default=0.6,
                   help="Semantic-layer scaling; 1.0 = plain softmax(1-cos); 0 = w/o Semantic-Aware.")
    g.add_argument("--uniform_geometry", action="store_true",
                   help="Ablation: uniform first-layer weights (w/o Geometry-Aware).")
    g.add_argument("--random_curvature", action="store_true",
                   help="Ablation: replace ORC with random noise in [-2, 1].")
    g.add_argument("--prune_topk_orc", dest="prune_topk_orc", type=float, default=0.0,
                   help="Table 3: fraction of highest-ORC edges to remove.")

    g = p.add_argument_group("logging")
    g.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    g.add_argument("--profile_runtime", action="store_true",
                   help="Record synchronized training/evaluation times in results/time_cost.csv.")

    args = p.parse_args()
    args.layer_size = ast.literal_eval(args.layer_size)
    args.layer_dropout = ast.literal_eval(args.layer_dropout)
    return args


def main():
    args = parse_args()
    print(args)
    set_seed(args.seed, deterministic=args.deterministic)

    trainer = Trainer(args)
    trainer.fit()
    trainer.test_best()


if __name__ == "__main__":
    main()