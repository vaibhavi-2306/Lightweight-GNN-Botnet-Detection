"""
Reference checks (no GNN): logistic regression on the SAME train/test split.
Run from the project root:   python src/reference_checks.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from evaluation import compute_metrics

d = torch.load("data/processed/dataset.pt", weights_only=True)
split = d["flow_split"].numpy()
y = d["flow_y"].numpy()
scn = d["flow_scenario"].numpy()
tr = np.where(split == 0)[0]
te = np.where(split == 2)[0]

flow_x = d["flow_x"].numpy()
node_x = d["x"].numpy()
src, dst = d["flow_src"].numpy(), d["flow_dst"].numpy()
node_feats = np.hstack([node_x[src], node_x[dst]])        # 8 columns

feature_sets = {
    "LR_flow_features_only": flow_x,
    "LR_node_features_only": node_feats,
    "LR_flow_plus_node": np.hstack([flow_x, node_feats]),
}

rows = []
for name, X in feature_sets.items():
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X[tr], y[tr])
    prob = clf.predict_proba(X[te])[:, 1]
    for scope, mask in [("overall", np.ones(len(te), dtype=bool)),
                        ("scenario_5", scn[te] == 5),
                        ("scenario_11", scn[te] == 11)]:
        m = compute_metrics(y[te][mask], prob[mask])
        rows.append({"model": name, "scope": scope, **m})

out = pd.DataFrame(rows)
cols = ["model", "scope", "accuracy", "precision", "recall", "f1", "roc_auc",
        "fp", "fn"]
print(out[cols].to_string(index=False))
out.to_csv("results/reference_results.csv", index=False)
print("\nSaved: results/reference_results.csv")