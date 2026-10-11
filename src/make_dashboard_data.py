"""
Creates small summary files so the dashboard never needs the raw data."""
import json
from pathlib import Path

import pandas as pd

RAW, RESULTS = Path("data/raw"), Path("results")
SCENARIOS = {5: "scenario_05.binetflow", 11: "scenario_11.binetflow"}
STAT_COLS = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "DstBytes", "Rate"]

summary, types, frames = {}, [], []
for scn, fname in SCENARIOS.items():
    path = RAW / fname
    n_cols = pd.read_csv(path, nrows=0).shape[1]
    df = pd.read_csv(path, usecols=["Label"] + STAT_COLS, low_memory=False)
    label = df["Label"].astype(str)

    group = pd.Series("Background", index=df.index)       # same rule as preprocessing
    group[label.str.contains("Normal")] = "Normal"
    group[label.str.contains("Botnet")] = "Botnet"

    summary[str(scn)] = {"file": fname, "rows": int(len(df)), "columns": int(n_cols),
                         "classes": {k: int(v) for k, v in group.value_counts().items()}}

    bot = label[group == "Botnet"].str.replace(
        r"^flow=From-Botnet-V\d+(?:-\d+)?-", "", regex=True)
    for k, v in bot.value_counts().head(15).items():
        types.append({"scenario": scn, "botnet_flow_type": k, "flows": int(v)})
    frames.append(df[STAT_COLS])

(RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame(types).to_csv(RESULTS / "botnet_label_types.csv", index=False)
pd.concat(frames).describe().T.round(3).to_csv(RESULTS / "dataset_statistics.csv")
print(json.dumps(summary, indent=2))
print("Saved: results/dataset_summary.json, botnet_label_types.csv, dataset_statistics.csv")