import argparse
from time import time
import dgl
import numpy as np
import torch as th
import torch.optim as optim
from data_loader import Data
from models import ORCAGCN_ConvE
from utils import in_out_norm
from tqdm import tqdm
import wandb
import os
os.environ["WANDB_MODE"] = "offline"

def predict(model, graph, device, data_iter, split="valid", mode="tail"):
    model.eval()
    with th.no_grad():
        results = {}
        train_iter = iter(data_iter["{}_{}".format(split, mode)])

        for step, batch in enumerate(tqdm(train_iter, desc=f"{split},{mode}")):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            pred = model(graph, sub, rel)
            b_range = th.arange(pred.size()[0], device=device)
            target_pred = pred[b_range, obj]
            pred = th.where(label.bool(), -th.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred

            # compute metrics
            ranks = (
                1
                + th.argsort(
                    th.argsort(pred, dim=1, descending=True),
                    dim=1,
                    descending=False,
                )[b_range, obj]
            )
            ranks = ranks.float()
            results["count"] = th.numel(ranks) + results.get("count", 0.0)
            results["mr"] = th.sum(ranks).item() + results.get("mr", 0.0)
            results["mrr"] = th.sum(1.0 / ranks).item() + results.get(
                "mrr", 0.0
            )
            for k in [1, 3, 10]:
                results["hits@{}".format(k)] = th.numel(
                    ranks[ranks <= (k)]
                ) + results.get("hits@{}".format(k), 0.0)

    return results


# evaluation function, evaluate the head and tail prediction and then combine the results
def evaluate(model, graph, device, data_iter, split="valid"):
    # predict for head and tail
    left_results = predict(model, graph, device, data_iter, split, mode="tail")
    right_results = predict(model, graph, device, data_iter, split, mode="head")
    results = {}
    count = float(left_results["count"])

    # combine the head and tail prediction results
    # Metrics: MRR, MR, and Hit@k
    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round(
        (left_results["mr"] + right_results["mr"]) / (2 * count), 5
    )
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )
    for k in [1, 3, 10]:
        results["left_hits@{}".format(k)] = round(
            left_results["hits@{}".format(k)] / count, 5
        )
        results["right_hits@{}".format(k)] = round(
            right_results["hits@{}".format(k)] / count, 5
        )
        results["hits@{}".format(k)] = round(
            (
                left_results["hits@{}".format(k)]
                + right_results["hits@{}".format(k)]
            )
            / (2 * count),
            5,
        )
    return results

def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"
    print(device)
    # construct graph, split in/out edges and prepare train/validation/test data_loader
    data = Data(
        args.dataset, args.lbl_smooth, args.num_workers, args.batch_size
    )
    data_iter = data.data_iter  # train/validation/test data_loader
    graph = data.g.to(device)
    num_rel = th.max(graph.edata["etype"]).item() + 1
    print(f"num_rel:{num_rel}")
    print(f"num_ent:{graph.num_nodes()}")
    # Compute in/out edge norms and store in edata
    graph = in_out_norm(graph)

    # Step 2: Create model =================================================================== #
    orcagcn_model = ORCAGCN_ConvE(
        num_bases=args.num_bases,
        num_rel=num_rel,
        num_ent=graph.num_nodes(),
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
    )
    orcagcn_model = orcagcn_model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.BCELoss()
    optimizer = optim.Adam(
        orcagcn_model.parameters(), lr=args.lr, weight_decay=args.l2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10)
    # Step 4: training epoches =============================================================== #
    val_best_mrr = 0.0
    # tes_best_mrr = 0.0
    kill_cnt = 0
    for epoch in range(1, args.max_epochs + 1):
        # Training and validation using a full graph
        orcagcn_model.train()
        train_loss = []
        for step, batch in enumerate(tqdm(data_iter["train"], desc=f"Epoch {epoch}")):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            logits = orcagcn_model(graph, sub, rel)

            # compute loss
            tr_loss = loss_fn(logits, label)
            train_loss.append(tr_loss.item())

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()



        train_loss = np.sum(train_loss)
        test_log['train/epoch'] = epoch
        train_log['train/loss'] = train_loss
        # wandb.log(train_log, step=epoch)


        # validate
        if epoch % 5 == 0:
            val_results = evaluate(
            orcagcn_model, graph, device, data_iter, split="valid"
            )
            valid_log['valid/epoch'] = epoch
            valid_log['valid/mrr'] = val_results["mrr"]
            valid_log['valid/hits@1'] = val_results["hits@1"]
            valid_log['valid/hits@3'] = val_results["hits@3"]
            valid_log['valid/hits@10'] = val_results["hits@10"]
            wandb.log(valid_log, step=epoch)
            # scheduler.step(val_results["mrr"])
            if val_results["mrr"] >= val_best_mrr:
                val_best_mrr = val_results["mrr"]
                kill_cnt = 0
                test_results = evaluate(
                    orcagcn_model, graph, device, data_iter, split="test"
                )
                test_log['test/epoch'] = epoch
                test_log['test/mrr'] = test_results["mrr"]
                test_log['test/hits@1'] = test_results["hits@1"]
                test_log['test/hits@3'] = test_results["hits@3"]
                test_log['test/hits@10'] = test_results["hits@10"]
                wandb.log(test_log, step=epoch)
                print("saving model...")
                th.save(
                    orcagcn_model.state_dict(), args.dataset + "/" + "comp_link" + "_" + args.dataset
                )
                with open(f"{args.dataset}/results_{args.dataset}.txt", "a") as f:
                    f.write(f"Epoch: {epoch}\n")
                    f.write("Valid:\n")
                    for key, value in val_results.items():
                        f.write(f"{key}: {value:.5f}\n")
                    f.write("Test:\n")
                    for key, value in test_results.items():
                        f.write(f"{key}: {value:.5f}\n")
                    f.write('----------------------------\n')
            else:
                kill_cnt += 1
                print(kill_cnt)
                if kill_cnt > 40:
                    print("early stop.")
                    break
            print(
            "Valid MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
                val_results["mrr"],
                val_results["mr"],
                val_results["hits@10"],
                val_results["hits@3"],
                val_results["hits@1"],
            )
                )
    # test use the best model
    orcagcn_model.eval()
    orcagcn_model.load_state_dict(th.load(args.dataset + "/" +"comp_link" + "_" + args.dataset))
    test_results = evaluate(
        orcagcn_model, graph, device, data_iter, split="test"
    )
    print(
        "Test MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
            test_results["mrr"],
            test_results["mr"],
            test_results["hits@10"],
            test_results["hits@3"],
            test_results["hits@1"],
        )
    )
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="dataset",
        default="FB15k-237",
        help="Dataset to use, default: FB15k-237",
    )
    parser.add_argument(
        "--model", dest="model", default="compgcn", help="Model Name"
    )
    parser.add_argument(
        "--score_func",
        dest="score_func",
        default="conve",
        help="Score Function for Link prediction",
    )
    parser.add_argument(
        "--opn",
        dest="opn",
        default="ccorr",
        help="Composition Operation to be used in CompGCN",
    )

    parser.add_argument(
        "--batch", dest="batch_size", default=512, type=int, help="Batch size"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default="0",
        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
    )
    parser.add_argument(
        "--epoch",
        dest="max_epochs",
        type=int,
        default=3500,
        help="Number of epochs",
    )
    parser.add_argument(
        "--l2", type=float, default=0.0, help="L2 Regularization for Optimizer"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Starting Learning Rate"
    )
    parser.add_argument(
        "--lbl_smooth",
        dest="lbl_smooth",
        type=float,
        default=0.1,
        help="Label Smoothing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of processes to construct batches",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=41504,
        type=int,
        help="Seed for randomization",
    )

    parser.add_argument(
        "--num_bases",
        dest="num_bases",
        default=-1,
        type=int,
        help="Number of basis relation vectors to use",
    )
    parser.add_argument(
        "--init_dim",
        dest="init_dim",
        default=100,
        type=int,
        help="Initial dimension size for entities and relations",
    )
    parser.add_argument(
        "--layer_size",
        nargs="?",
        default="[200, 200]",
        help="List of output size for each compGCN layer",
    )
    parser.add_argument(
        "--gcn_drop",
        dest="dropout",
        default=0.2,
        type=float,
        help="Dropout to use in GCN Layer",
    )
    parser.add_argument(
        "--layer_dropout",
        nargs="?",
        default="[0.3, 0.3]",
        help="List of dropout value after each compGCN layer",
    )

    # ConvE specific hyperparameters
    parser.add_argument(
        "--hid_drop",
        dest="hid_drop",
        default=0.3,
        type=float,
        help="ConvE: Hidden dropout",
    )
    parser.add_argument(
        "--feat_drop",
        dest="feat_drop",
        default=0.3,
        type=float,
        help="ConvE: Feature Dropout",
    )
    parser.add_argument(
        "--k_w", dest="k_w", default=10, type=int, help="ConvE: k_w"
    )
    parser.add_argument(
        "--k_h", dest="k_h", default=20, type=int, help="ConvE: k_h"
    )
    parser.add_argument(
        "--num_filt",
        dest="num_filt",
        default=200,
        type=int,
        help="ConvE: Number of filters in convolution",
    )
    parser.add_argument(
        "--ker_sz",
        dest="ker_sz",
        default=7,
        type=int,
        help="ConvE: Kernel size to use",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    print(args)

    args.layer_size = eval(args.layer_size)
    args.layer_dropout = eval(args.layer_dropout)

    wandb.init(
        project='orca',
        entity='zhangdonglinzaici-shandong-university',
        name='1-orc 1-sim'
    )
    train_log = {}
    valid_log = {}
    test_log = {}
    main(args)
