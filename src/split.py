import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
SEED = 42
N_BLOCKS = 40
FRACTIONS = (0.6, 0.2, 0.2)          # train / val / test
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
PROC = Path("data/processed")
MODELS = Path("models")
RESULTS = Path("results")
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
lines = []
def log(text=""):
    print(text)
    lines.append(str(text))
def make_blocks(t, labeled, n_blocks):
    """Block id per flow; cut points are quantiles of the LABELED flows' times."""
    edges = np.unique(np.quantile(t[labeled], np.linspace(0, 1, n_blocks + 1)))
    return np.searchsorted(edges[1:-1], t, side="right")
def assign_blocks(block_ids, rng):
    """Randomly assign blocks to train(0)/val(1)/test(2) with 60/20/20."""
    block_ids = rng.permutation(np.array(sorted(block_ids), dtype=np.int64))
    n = len(block_ids)
    n_tr = int(round(FRACTIONS[0] * n))
    n_va = int(round(FRACTIONS[1] * n))
    if n >= 3:                        # make sure every split gets at least one block
        n_tr = min(max(n_tr, 1), n - 2)
        n_va = min(max(n_va, 1), n - n_tr - 1)
    out = {}
    for i, b in enumerate(block_ids):
        out[int(b)] = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
    return out
# ------------------------------------------------------------------ load
cfg = json.loads((PROC / "feature_config.json").read_text())
feature_cols = cfg["feature_columns"]
flows = pd.read_parquet(PROC / "flows_clean.parquet",
                        columns=feature_cols + ["scenario", "t_sec", "label"])
graph = torch.load(PROC / "graph_structure.pt", weights_only=True)
assert len(flows) == len(graph["flow_src"]), "Row count mismatch with graph!"
log(f"Flows: {len(flows):,}   Features per flow: {len(feature_cols)}")
scenario = flows["scenario"].to_numpy()
t_all = flows["t_sec"].to_numpy()
y_all = flows["label"].to_numpy().astype(np.int64)
# ------------------------------------------------------------------ split
rng = np.random.default_rng(SEED)
flow_block = np.zeros(len(flows), dtype=np.int64)
flow_split = np.full(len(flows), -1, dtype=np.int64)
for scn in sorted(np.unique(scenario)):
    idx = np.where(scenario == scn)[0]
    t, y = t_all[idx], y_all[idx]
    labeled = y >= 0
    blk = make_blocks(t, labeled, N_BLOCKS)
    flow_block[idx] = blk
    bot_blocks = set(np.unique(blk[y == 1]).tolist())
    all_blocks = set(np.unique(blk[labeled]).tolist())
    mapping = {}
    mapping.update(assign_blocks(bot_blocks, rng))
    mapping.update(assign_blocks(all_blocks - bot_blocks, rng))
    sp = np.array([mapping.get(int(b), -1) for b in blk], dtype=np.int64)
    sp[~labeled] = -1
    flow_split[idx] = sp
    log(f"Scenario {scn}: {len(set(blk.tolist()))} blocks "
        f"({len(bot_blocks)} contain botnet flows)")
# ------------------------------------------------------------------ report
lab = y_all >= 0
rep = pd.DataFrame({
    "scenario": scenario[lab],
    "split": pd.Categorical([SPLIT_NAMES[s] for s in flow_split[lab]],
                            categories=["train", "val", "test"], ordered=True),
    "class": np.where(y_all[lab] == 1, "Botnet", "Normal"),
})
log("\nLabeled flows per scenario / split / class:")
table = pd.crosstab([rep["scenario"], rep["split"]], rep["class"])
log(table.to_string())
log("\nOverall labeled flows per split:")
overall = pd.crosstab(rep["split"], rep["class"])
overall["total"] = overall.sum(axis=1)
overall["botnet_%"] = (overall["Botnet"] / overall["total"] * 100).round(1)
log(overall.to_string())
problems = [(s, sp) for (s, sp), r in table.iterrows()
            if r.get("Botnet", 0) == 0 or r.get("Normal", 0) == 0]
if problems:
    log(f"\nWARNING: these scenario/split cells are missing a class: {problems}")
else:
    log("\nCheck passed: every scenario/split contains both classes.")
log(f"Background flows (split = -1, not used in loss/metrics): "
    f"{int((flow_split == -1).sum()):,}")
# ------------------------------------------------------------------ scaling
log_cols = [c for c in feature_cols if c.startswith("log_")]
li = [feature_cols.index(c) for c in log_cols]
X = flows[feature_cols].to_numpy(dtype=np.float32, copy=True)
scaler = StandardScaler().fit(X[flow_split == 0][:, li])   # TRAIN flows only
X[:, li] = scaler.transform(X[:, li]).astype(np.float32)
log(f"\nScaled {len(log_cols)} log_ columns using "
    f"{int((flow_split == 0).sum()):,} training flows only.")
# ------------------------------------------------------------------ save
graph.update({
    "flow_x": torch.from_numpy(X),
    "flow_y": torch.from_numpy(y_all),
    "flow_split": torch.from_numpy(flow_split),
    "flow_scenario": torch.from_numpy(scenario.astype(np.int64)),
    "flow_block": torch.from_numpy(flow_block),
    "flow_feature_names": feature_cols,
})
torch.save(graph, PROC / "dataset.pt")
joblib.dump(scaler, MODELS / "scaler.joblib")
config = {
    "seed": SEED, "n_blocks": N_BLOCKS, "split_fractions": FRACTIONS,
    "scaled_columns": log_cols, "n_flow_features": len(feature_cols),
    "split_counts": {f"{s}_{sp}": {c: int(v) for c, v in r.items()}
                     for (s, sp), r in table.iterrows()},}
(RESULTS / "experiment_config.json").write_text(json.dumps(config, indent=2))
(RESULTS / "split_report.txt").write_text("\n".join(lines), encoding="utf-8")
log(f"\nSaved: {PROC / 'dataset.pt'} "
    f"({(PROC / 'dataset.pt').stat().st_size / 1e6:.1f} MB)")
log(f"Saved: {MODELS / 'scaler.joblib'}")