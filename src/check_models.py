"""Check that the saved scaler and models reload correctly.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from common import EdgeGNN
from evaluation import compute_metrics

PROC, MODELS, RESULTS = Path("data/processed"), Path("models"), Path("results")
all_ok = True

d = torch.load(PROC / "dataset.pt", weights_only=True)
x, edge_index, flow_x = d["x"], d["edge_index"], d["flow_x"]
te = (d["flow_split"] == 2).nonzero(as_tuple=True)[0]

# 1) The saved scaler must reproduce the scaled features stored in dataset.pt
cfg = json.loads((PROC / "feature_config.json").read_text())
cols = cfg["feature_columns"]
log_cols = [c for c in cols if c.startswith("log_")]
li = [cols.index(c) for c in log_cols]
raw = pd.read_parquet(PROC / "flows_clean.parquet", columns=log_cols)
scaler = joblib.load(MODELS / "scaler.joblib")
rescaled = scaler.transform(raw.to_numpy(dtype=np.float32))
ok = bool(np.allclose(rescaled, flow_x[:, li].numpy(), atol=1e-4))
all_ok &= ok
print(f"[{'PASS' if ok else 'FAIL'}] scaler.joblib reproduces the scaled features "
      f"({len(log_cols)} columns)")

# 2) Each model must reload and reproduce its saved predictions and metrics
for name in ["baseline", "lightweight"]:
    ckpt = torch.load(MODELS / f"{name}_gnn.pth", weights_only=True)
    c = ckpt["config"]
    model = EdgeGNN(x.shape[1], flow_x.shape[1], c["hidden"], c["n_layers"],
                    c["head_hidden"], c["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        prob = torch.sigmoid(model(x, edge_index, d["flow_src"][te],
                                   d["flow_dst"][te], flow_x[te])).numpy()

    saved = np.load(RESULTS / f"{name}_test_predictions.npz")
    same_prob = bool(np.allclose(prob, saved["prob"], atol=1e-5))

    m = compute_metrics(d["flow_y"][te].numpy(), prob)
    row = pd.read_csv(RESULTS / f"{name}_results.csv").query("scope == 'overall'").iloc[0]
    same_f1 = abs(m["f1"] - row["f1"]) < 1e-4

    ok = same_prob and same_f1
    all_ok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {n_params:,} parameters | "
          f"predictions match saved: {same_prob} | "
          f"F1 reloaded {m['f1']} vs saved {row['f1']}")

print("\nAll checks passed." if all_ok else "\nSome checks FAILED - send me this output.")