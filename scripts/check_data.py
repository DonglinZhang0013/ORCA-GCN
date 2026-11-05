#!/usr/bin/env python3
"""Validate dataset text files and alignment of the two ORC NumPy arrays."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

def load_map(path: Path) -> dict[str, int]:
    return {line.strip().lower(): i for i, line in enumerate(path.open(encoding="utf-8"))}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    args = p.parse_args()
    d = args.data
    required = ["entities.txt", "relations.txt", "train.txt", "valid.txt", "test.txt",
                "graph_with_olliverRicci.npy", "olliverRicci.npy"]
    missing = [name for name in required if not (d / name).exists()]
    if missing:
        raise FileNotFoundError(f"{d}: missing {missing}")
    ent2id, rel2id = load_map(d / "entities.txt"), load_map(d / "relations.txt")
    train = []
    with (d / "train.txt").open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            h, r, t = (x.lower() for x in raw.rstrip("\n").split("\t"))
            try:
                train.append((ent2id[h], rel2id[r], ent2id[t]))
            except KeyError as exc:
                raise KeyError(f"train line {line_no}: unknown token {exc}") from exc
    edges = np.load(d / "graph_with_olliverRicci.npy")
    orc = np.load(d / "olliverRicci.npy")
    assert edges.ndim == 2 and edges.shape[0] == 3, edges.shape
    assert orc.ndim == 1, orc.shape
    assert edges.shape[1] == len(train) == orc.shape[0], (edges.shape, len(train), orc.shape)
    text_counter = Counter(train)
    array_triples = [(int(edges[0, i]), int(edges[2, i]), int(edges[1, i])) for i in range(edges.shape[1])]
    array_counter = Counter(array_triples)
    if text_counter != array_counter:
        missing = list((text_counter - array_counter).items())[:10]
        extra = list((array_counter - text_counter).items())[:10]
        raise ValueError(f"ORC edge array does not match train.txt; missing={missing}, extra={extra}")
    if edges[0].min() < 0 or edges[1].min() < 0:
        raise ValueError("Negative entity id")
    if edges[0].max() >= len(ent2id) or edges[1].max() >= len(ent2id):
        raise ValueError("Entity id outside entities.txt range")
    if edges[2].min() < 0 or edges[2].max() >= len(rel2id):
        raise ValueError("Relation id outside relations.txt range")
    if not np.isfinite(orc).all():
        raise ValueError("Non-finite ORC value")
    self_mask = edges[0] == edges[1]
    self_non_one = int(np.count_nonzero(~np.isclose(orc[self_mask], 1.0)))
    pair_counts = Counter(zip(edges[0].tolist(), edges[1].tolist()))
    print(f"[OK] {d.name}")
    print(f"  entities: {len(ent2id):,}")
    print(f"  relations: {len(rel2id):,}")
    print(f"  forward training triples / ORC values: {orc.size:,}")
    print(f"  unique ordered entity pairs: {len(pair_counts):,}")
    print(f"  parallel excess triples: {sum(c - 1 for c in pair_counts.values()):,}")
    print(f"  entity self-loop triples: {int(self_mask.sum()):,}")
    print(f"  self-loop ORC values not equal to +1.0: {self_non_one}")
    print(f"  ORC range: [{orc.min():.6f}, {orc.max():.6f}]")

if __name__ == "__main__":
    main()
