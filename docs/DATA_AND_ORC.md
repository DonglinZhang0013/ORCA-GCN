# Data and ORC file formats

## Dataset text files

Each dataset directory contains entities.txt, relations.txt, train.txt, valid.txt,
and test.txt. Entity and relation IDs are zero-based line numbers in their
respective mapping files. Triple files are tab-separated head, relation, tail.
Strings are lower-cased by the loader.

## Precomputed curvature arrays

graph_with_olliverRicci.npy is an int64 array of shape (3, E): row 0 is source
entity ID, row 1 is destination entity ID, and row 2 is forward relation ID.

olliverRicci.npy is a floating-point vector of shape (E,). Column q in the
edge array and element q in the curvature vector refer to the same forward
training triple. Array order may differ from train.txt line order; the
multiset of relation-specific triples is identical.

The repository stores one column per forward relation-specific training triple,
even when several relations share the same ordered entity pair. Pair-level ORC
is mapped back to all such parallel triples.

## Directed graph convention

ORC preprocessing uses original directed training triples. Relation-specific
entity self-loop triples (head == tail) are excluded from transport and assigned
the implementation convention kappa_self = +1.0.

At training time, data_loader.py generates an inverse message edge for every
stored forward edge. The inverse edge inherits its forward edge's ORC value.
The encoder's separately learned self-loop transformation is distinct from
relation-specific self-loop triples in the data.

## Attention conversion

The loader clips ORC to [-2, 1], computes 1 - kappa, duplicates values for
inverse edges, and applies a destination-neighborhood softmax once. Because
softmax is invariant to adding a constant, softmax(1-kappa) is numerically
equivalent to softmax(-kappa) within each destination neighborhood.

## Validation

    python scripts/check_data.py --data FB15k-237
    python scripts/check_data.py --data WN18RR

The checker verifies text/ID consistency, NumPy shapes, equality of the
forward-triple multiset, edge-array/ORC-vector alignment, finite ORC values,
and the self-loop convention.
