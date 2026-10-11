"""
Shared model + training code for BOTH the baseline and the lightweight GNN.
Only the hyper-parameters in the config differ, so the comparison is fair.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from evaluation import compute_metrics

SEED = 42
PROC = Path("data/processed")
MODELS = Path("models")
RESULTS = Path("results")
RUNS = RESULTS / "runs"
SPLITS = {"train": 0, "val": 1, "test": 2}


class EdgeGNN(nn.Module):
    """GCN layers -> IP embeddings -> small head that classifies each flow."""

    def __init__(self, node_in, edge_in, hidden, n_layers, head_hidden, dropout):
        super().__init__()
        dims = [node_in] + [hidden] * n_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1], cached=True) for i in range(n_layers)])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + edge_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x, edge_index, src, dst, flow_x):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, self.training)
        z = torch.cat([x[src], x[dst], flow_x], dim=1)
        return self.head(z).squeeze(-1)          # one logit per flow


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(cfg, seed=SEED, verbose=True):
    """Train + evaluate one model with one seed.
    Always writes results/runs/<name>_seed<seed>_*.csv.
    If seed == SEED (42) it also writes the 'main' model and result files."""
    name = cfg["name"]
    for p in (MODELS, RESULTS, RUNS):
        p.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    def say(*args):
        if verbose:
            print(*args)

    say(f"Model: {name} | seed {seed} | PyTorch threads: {torch.get_num_threads()}")

    # ---------------------------------------------------------- data
    d = torch.load(PROC / "dataset.pt", weights_only=True)
    x, edge_index = d["x"], d["edge_index"]
    f_src, f_dst, f_x = d["flow_src"], d["flow_dst"], d["flow_x"]
    y, split, scn = d["flow_y"], d["flow_split"], d["flow_scenario"]
    idx = {k: (split == v).nonzero(as_tuple=True)[0] for k, v in SPLITS.items()}
    y_float = y.float()
    say({k: len(v) for k, v in idx.items()}, "labeled flows per split")

    def forward(part):
        i = idx[part]
        return model(x, edge_index, f_src[i], f_dst[i], f_x[i])

    # ---------------------------------------------------------- model
    model = EdgeGNN(x.shape[1], f_x.shape[1], cfg["hidden"], cfg["n_layers"],
                    cfg["head_hidden"], cfg["dropout"])
    n_params = count_params(model)
    say(model)
    say(f"Trainable parameters: {n_params:,}\n")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    # ---------------------------------------------------------- training
    history, best_val, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        opt.zero_grad()
        tr_loss = loss_fn(forward("train"), y_float[idx["train"]])
        tr_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(forward("val"), y_float[idx["val"]]).item()
        history.append({"epoch": epoch, "train_loss": tr_loss.item(),
                        "val_loss": va_loss})
        if epoch % 10 == 0 or epoch == 1:
            say(f"epoch {epoch:3d} | train loss {tr_loss.item():.4f} "
                f"| val loss {va_loss:.4f}")

        if va_loss < best_val - 1e-4:               # early stopping
            best_val, best_epoch, bad = va_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                say(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    train_time = time.perf_counter() - t0
    epochs_run = len(history)
    model.load_state_dict(best_state)                # restore the best model

    # ---------------------------------------------------------- inference time
    model.eval()
    with torch.no_grad():
        for _ in range(5):                           # warm-up runs (not timed)
            forward("test")
        times = []
        for _ in range(30):
            t = time.perf_counter()
            forward("test")
            times.append(time.perf_counter() - t)
        prob = {p: torch.sigmoid(forward(p)).numpy() for p in ("val", "test")}
    inference_ms = float(np.median(times) * 1000)

    mi = psutil.Process().memory_info()
    peak_mem_mb = getattr(mi, "peak_wset", mi.rss) / 1e6   # peak RAM of this process

    # ---------------------------------------------------------- evaluation
    y_val, y_te = y[idx["val"]].numpy(), y[idx["test"]].numpy()
    te_scn = scn[idx["test"]].numpy()
    val_m = compute_metrics(y_val, prob["val"])
    say(f"\nValidation: F1 {val_m['f1']}  recall {val_m['recall']}  "
        f"precision {val_m['precision']}")

    eff = {
        "params": n_params,
        "train_time_s": round(train_time, 2),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "time_per_epoch_s": round(train_time / epochs_run, 4),
        "inference_ms": round(inference_ms, 2),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "model_size_kb": round(n_params * 4 / 1024, 2),
    }
    rows = []
    for scope, mask in [("overall", np.ones(len(y_te), dtype=bool)),
                        ("scenario_5", te_scn == 5),
                        ("scenario_11", te_scn == 11)]:
        m = compute_metrics(y_te[mask], prob["test"][mask])
        rows.append({"model": name, "seed": seed, "scope": scope, **m, **eff})
    results = pd.DataFrame(rows)
    hist_df = pd.DataFrame(history)

    # ---------------------------------------------------------- save
    results.to_csv(RUNS / f"{name}_seed{seed}_results.csv", index=False)
    hist_df.to_csv(RUNS / f"{name}_seed{seed}_history.csv", index=False)

    if seed == SEED:                                  # the 'main' saved model
        torch.save({"state_dict": model.state_dict(), "config": cfg},
                   MODELS / f"{name}_gnn.pth")
        results.to_csv(RESULTS / f"{name}_results.csv", index=False)
        hist_df.to_csv(RESULTS / f"{name}_history.csv", index=False)
        np.savez(RESULTS / f"{name}_test_predictions.npz",
                 flow_index=idx["test"].numpy(), y_true=y_te,
                 prob=prob["test"], scenario=te_scn)
        (RESULTS / f"{name}_config.json").write_text(json.dumps(cfg, indent=2))

    say("\n" + "=" * 60)
    say(f"TEST RESULTS - {name} (seed {seed})")
    say("=" * 60)
    show = ["scope", "n_flows", "accuracy", "precision", "recall", "f1",
            "roc_auc", "tn", "fp", "fn", "tp"]
    say(results[show].to_string(index=False))
    say("\nEfficiency:")
    for k, v in eff.items():
        say(f"  {k:<18}: {v}")