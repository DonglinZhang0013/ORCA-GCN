"""Dataset and sparse graph construction for ORCA-GCN.

The loader reads text triples plus two aligned NumPy arrays containing the
forward training edges and their precomputed Ollivier--Ricci curvature values.
It creates inverse message-passing edges online and assigns each inverse edge
the curvature value of its corresponding forward edge.
"""

from collections import defaultdict as ddict

import os
import dgl

import numpy as np
import torch
from ordered_set import OrderedSet
from torch.utils.data import DataLoader, Dataset
from dgl.nn import edge_softmax


class TrainDataset(Dataset):
    """
    Training Dataset class.
    Parameters
    ----------
    triples: The triples used for training the model
    num_ent: Number of entities in the knowledge graph
    lbl_smooth: Label smoothing

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, num_ent, lbl_smooth):
        self.triples = triples
        self.num_ent = num_ent
        self.lbl_smooth = lbl_smooth
        self.entities = np.arange(self.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele["triple"]), np.int32(ele["label"])
        trp_label = self.get_label(label)
        # label smoothing
        if self.lbl_smooth != 0.0:
            trp_label = (1.0 - self.lbl_smooth) * trp_label + (
                1.0 / self.num_ent
            )

        return triple, trp_label

    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.stack(triples, dim=0)
        trp_label = torch.stack(labels, dim=0)
        return triple, trp_label

    # for edges that exist in the graph, the entry is 1.0, otherwise the entry is 0.0
    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


class TestDataset(Dataset):
    """
    Evaluation Dataset class.
    Parameters
    ----------
    triples: The triples used for evaluating the model
    num_ent: Number of entities in the knowledge graph

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, num_ent):
        self.triples = triples
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele["triple"]), np.int32(ele["label"])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.stack(triples, dim=0)
        label = torch.stack(labels, dim=0)
        return triple, label

    # for edges that exist in the graph, the entry is 1.0, otherwise the entry is 0.0
    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


class Data(object):
    def __init__(self, dataset, lbl_smooth, num_workers, batch_size,
                 random_curvature=False, prune_topk_orc=0.0):
        """
        Reading in raw triples and converts it into a standard format.
        Parameters
        ----------
        dataset:           The name of the dataset
        lbl_smooth:        Label smoothing
        num_workers:       Number of workers of dataloaders
        batch_size:        Batch size of dataloaders

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.rel2id:            Relation to unique identifier mapping
        self.id2ent:            Inverse mapping of self.ent2id
        self.id2rel:            Inverse mapping of self.rel2id
        self.num_ent:           Number of entities in the knowledge graph
        self.num_rel:           Number of relations in the knowledge graph

        self.g:                 The dgl graph constucted from the edges in the traing set and all the entities in the knowledge graph
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits
        """
        self.dataset = dataset
        self.lbl_smooth = lbl_smooth
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.random_curvature = random_curvature
        self.prune_topk_orc = prune_topk_orc

        # read in raw data and get mappings
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ["train", "test", "valid"]:
            for line in open("./{}/{}.txt".format(self.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split("\t"))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)
        self.ent2id = {}
        self.rel2id = {}
        for idx, line in enumerate(open('./{}/entities.txt'.format(self.dataset))):
            entity_str = line.strip()
            self.ent2id[entity_str] = idx
        for idx, line in enumerate(open('./{}/relations.txt'.format(self.dataset))):
            relation_str = line.strip()
            self.rel2id[relation_str] = idx

        self.rel2id.update(
            {
                rel + "_reverse": idx + len(self.rel2id)
                for idx, rel in enumerate(rel_set)
            }
        )

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id) // 2

        # read in ids of subjects, relations, and objects for train/test/valid
        self.data = ddict(list)  # stores the triples
        sr2o = ddict(
            set
        )  # The key of sr20 is (subject, relation), and the items are all the successors following (subject, relation)
        src = []
        dst = []
        rels = []
        inver_src = []
        inver_dst = []
        inver_rels = []
        orc = []

        for split in ["train","test", "valid"]:
            for line in open("./{}/{}.txt".format(self.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split("\t"))
                sub_id, rel_id, obj_id = (
                    self.ent2id[sub],
                    self.rel2id[rel],
                    self.ent2id[obj],
                )
                self.data[split].append((sub_id, rel_id, obj_id))

                if split == "train":
                    sr2o[(sub_id, rel_id)].add(obj_id)
                    sr2o[(obj_id, rel_id + self.num_rel)].add(
                        sub_id
                    )  # append the reversed edges
                  
        graph_npy = './{}/graph_with_olliverRicci.npy'.format(self.dataset)
        orc_npy = './{}/olliverRicci.npy'.format(self.dataset)
        if not (os.path.exists(graph_npy) and os.path.exists(orc_npy)):
            raise FileNotFoundError(
                "Missing pre-computed ORC files for dataset '{}':\n"
                "  {}\n  {}\n"
                "These store the training graph as an int array of shape (3, E) "
                "[src_entity_id, dst_entity_id, relation_id] and a matching float "
                "array of ORC values (1 value per edge). Regenerate them with "
                "entity ids (for WN18RR run scripts/build_wn18rr_orc_npy.py; "
                "see README) before running this dataset.".format(self.dataset, graph_npy, orc_npy)
            )
        edge_index_type = np.load(graph_npy)
        orc = np.load(orc_npy)
        if edge_index_type.ndim != 2 or edge_index_type.shape[0] != 3:
            raise ValueError(f"{graph_npy} must have shape (3, E); got {edge_index_type.shape}")
        if orc.ndim != 1 or orc.shape[0] != edge_index_type.shape[1]:
            raise ValueError(
                f"ORC alignment error: {graph_npy} has {edge_index_type.shape[1]} edges "
                f"but {orc_npy} has shape {orc.shape}"
            )
        self.edge_count_raw = int(orc.shape[0])
        if self.random_curvature:
            # Ablation (w/ Random Curvature): replace true ORC with uniform noise
            # sampled from the observed range [-2, 1].
            print("Ablation: replacing true ORC with random noise in [-2, 1]")
            orc = np.random.uniform(-2.0, 1.0, size=orc.shape)
        if self.prune_topk_orc and self.prune_topk_orc > 0.0:
            # Table 3: remove the top-k% highest-ORC edges from the aggregation
            # graph (same selection rule as get_deleted_train_based_on_orc.py).
            E = orc.shape[0]
            k = int(E * self.prune_topk_orc)
            if k > 0:
                gate = np.partition(orc, -k)[-k]  # k-th largest ORC value
                keep = orc < gate
                print("Pruning top {:.0%} highest-ORC edges: {} removed, {} kept".format(
                    self.prune_topk_orc, E - int(keep.sum()), int(keep.sum())))
                edge_index_type = edge_index_type[:, keep]
                orc = orc[keep]
        self.edge_count_after_pruning = int(orc.shape[0])
        for col in range(edge_index_type.shape[1]):
            sub_id = int(edge_index_type[0][col])
            obj_id = int(edge_index_type[1][col])
            rel_id = int(edge_index_type[2][col])
            src.append(sub_id)
            dst.append(obj_id)
            rels.append(rel_id)
            inver_src.append(obj_id)
            inver_dst.append(sub_id)
            inver_rels.append(rel_id + self.num_rel)
        # construct dgl graph
        src = src + inver_src
        dst = dst + inver_dst
        rels = rels + inver_rels
        orc = np.minimum(orc, 1.0)
        orc = np.maximum(orc, -2.0)
        print("orc = 1.0 - orc")
        orc = 1.0 - orc
        orc = torch.Tensor(np.concatenate([orc, orc]))

        self.g = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.num_message_edges = int(self.g.num_edges())
        self.g.edata["etype"] = torch.Tensor(rels).long()
        
        self.g.edata["1-orc"] = edge_softmax(self.g, orc, norm_by="dst")
        # identify in and out edges
        in_edges_mask = [True] * (self.g.num_edges() // 2) + [False] * (
            self.g.num_edges() // 2
        )
        out_edges_mask = [False] * (self.g.num_edges() // 2) + [True] * (
            self.g.num_edges() // 2
        )
        self.g.edata["in_edges_mask"] = torch.Tensor(in_edges_mask)
        self.g.edata["out_edges_mask"] = torch.Tensor(out_edges_mask)

        # Prepare train/valid/test data
        self.data = dict(self.data)
        self.sr2o = {
            k: list(v) for k, v in sr2o.items()
        }  # store only the train data

        for split in ["test", "valid"]:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.num_rel)].add(sub)

        self.sr2o_all = {
            k: list(v) for k, v in sr2o.items()
        }  # store all the data
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples["train"].append(
                {"triple": (sub, rel, -1), "label": self.sr2o[(sub, rel)]}
            )

        for split in ["test", "valid"]:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.num_rel
                self.triples["{}_{}".format(split, "tail")].append(
                    {
                        "triple": (sub, rel, obj),
                        "label": self.sr2o_all[(sub, rel)],
                    }
                )
                self.triples["{}_{}".format(split, "head")].append(
                    {
                        "triple": (obj, rel_inv, sub),
                        "label": self.sr2o_all[(obj, rel_inv)],
                    }
                )

        self.triples = dict(self.triples)

        def get_train_data_loader(split, batch_size, shuffle=True):
            return DataLoader(
                TrainDataset(
                    self.triples[split], self.num_ent, self.lbl_smooth
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.num_workers),
                collate_fn=TrainDataset.collate_fn,
            )

        def get_test_data_loader(split, batch_size, shuffle=True):
            return DataLoader(
                TestDataset(self.triples[split], self.num_ent),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.num_workers),
                collate_fn=TestDataset.collate_fn,
            )

        # train/valid/test dataloaders
        self.data_iter = {
            "train": get_train_data_loader("train", self.batch_size),
            "valid_head": get_test_data_loader("valid_head", self.batch_size * 10),
            "valid_tail": get_test_data_loader("valid_tail", self.batch_size * 10),
            "test_head": get_test_data_loader("test_head", self.batch_size * 10),
            "test_tail": get_test_data_loader("test_tail", self.batch_size * 10),
        }