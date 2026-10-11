import json
from pathlib import Path
import numpy as np
import pandas as pd
SEED = 42
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
SCENARIOS = {5: "scenario_05.binetflow", 11: "scenario_11.binetflow"}
# Numeric columns we keep (TotPkts/TotBytes/TotAppByte are sums of these, so dropped)
NUMERIC_COLS = ["TcpRtt", "SynAck", "AckDat", "SrcPkts", "DstPkts", "SrcBytes",
                "DstBytes", "SAppBytes", "DAppBytes", "Dur", "Rate", "SrcRate",
                "DstRate"]
USECOLS = (["SrcAddr", "DstAddr", "Proto", "Sport", "Dport", "State",
            "StartTime", "Label"] + NUMERIC_COLS)
# Fixed category lists (chosen from the inspection reports); everything else -> "other"
PROTO_LEVELS = ["tcp", "udp", "icmp"]
STATE_LEVELS = ["CON", "FIN", "RST", "INT", "REQ", "URP", "ECO", "TXD", "UNK"]
def make_label(label_series):
    """1 = Botnet, 0 = Normal, -1 = Background / unlabeled."""
    s = label_series.astype(str)
    y = np.full(len(s), -1, dtype=np.int8)
    y[s.str.contains("Normal", case=False).to_numpy()] = 0
    y[s.str.contains("Botnet", case=False).to_numpy()] = 1
    return y
def port_bucket(series):
    """0 = well-known (<1024), 1 = registered, 2 = dynamic (>=49152),
    3 = no usable port (ICMP hex codes, missing, ARP...)."""
    p = pd.to_numeric(series, errors="coerce")  # hex like '0x0303' becomes NaN
    bucket = np.full(len(p), 3, dtype=np.int8)
    bucket[((p >= 0) & (p < 1024)).to_numpy()] = 0
    bucket[((p >= 1024) & (p < 49152)).to_numpy()] = 1
    bucket[(p >= 49152).to_numpy()] = 2
    return bucket
def build_features(df, scn):
    out = pd.DataFrame(index=df.index)
    out["scenario"] = np.int8(scn)
    out["src"] = df["SrcAddr"].astype(str)
    out["dst"] = df["DstAddr"].astype(str)
    out["t_sec"] = (df["StartTime"] - df["StartTime"].min()).dt.total_seconds()
    out["label"] = make_label(df["Label"])
    for c in NUMERIC_COLS:
        out[f"log_{c}"] = np.log1p(df[c].clip(lower=0)).astype(np.float32)
    proto = df["Proto"].astype(str).str.lower()
    for lvl in PROTO_LEVELS:
        out[f"proto_{lvl}"] = (proto == lvl).astype(np.float32)
    out["proto_other"] = (~proto.isin(PROTO_LEVELS)).astype(np.float32)
    state = df["State"].astype(str).str.upper()
    for lvl in STATE_LEVELS:
        out[f"state_{lvl}"] = (state == lvl).astype(np.float32)
    out["state_other"] = (~state.isin(STATE_LEVELS)).astype(np.float32)
    for name, col in [("sport", "Sport"), ("dport", "Dport")]:
        b = port_bucket(df[col])
        for k in range(4):
            out[f"{name}_b{k}"] = (b == k).astype(np.float32)
    return out
def main():
    lines = []
    def log(text=""):
        print(text)
        lines.append(str(text))
    frames = []
    for scn, fname in SCENARIOS.items():
        log("=" * 60)
        log(f"Scenario {scn}: {fname}")
        log("=" * 60)
        df = pd.read_csv(RAW_DIR / fname, usecols=USECOLS, low_memory=False)
        log(f"Raw rows                  : {len(df):,}")
        n = len(df)
        df = df.drop_duplicates()
        log(f"Duplicates removed        : {n - len(df):,}")
        df["StartTime"] = pd.to_datetime(df["StartTime"],
                                         format="%Y/%m/%d %H:%M:%S.%f",
                                         errors="coerce")
        n = len(df)
        df = df.dropna(subset=["StartTime"])
        log(f"Bad timestamps removed    : {n - len(df):,}")
        df[NUMERIC_COLS] = (df[NUMERIC_COLS]
                            .replace([np.inf, -np.inf], np.nan).fillna(0))
        feats = build_features(df, scn)
        feats = feats.sort_values("t_sec").reset_index(drop=True)
        del df  # free memory
        names = {1: "Botnet", 0: "Normal", -1: "Background"}
        log("\nLabel counts:")
        log(feats["label"].map(names).value_counts().to_string())
        lab = feats[feats["label"] >= 0]
        lab_name = lab["label"].map(names)
        log("\nProtocol vs label (labeled flows only; ICMP = 1):")
        log(pd.crosstab(lab["proto_icmp"].map({1.0: "icmp", 0.0: "non-icmp"}),
                        lab_name).to_string())
        log("\nWhen do labeled flows occur? (capture time split into 5 equal slices)")
        slices = pd.cut(lab["t_sec"], bins=5,
                        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
        log(pd.crosstab(slices, lab_name).to_string())
        log("")
        frames.append(feats)
    all_df = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in all_df.columns
                    if c not in ("scenario", "src", "dst", "t_sec", "label")]
    out_path = OUT_DIR / "flows_clean.parquet"
    all_df.to_parquet(out_path, index=False)
    config = {
        "seed": SEED,
        "scenarios": list(SCENARIOS.keys()),
        "feature_columns": feature_cols,
        "numeric_columns_log1p": NUMERIC_COLS,
        "proto_levels": PROTO_LEVELS,
        "state_levels": STATE_LEVELS,
        "label_meaning": {"1": "Botnet", "0": "Normal", "-1": "Background (unlabeled)"},
    }
    (OUT_DIR / "feature_config.json").write_text(json.dumps(config, indent=2))
    log("=" * 60)
    log(f"Total flows saved   : {len(all_df):,}")
    log(f"Number of features  : {len(feature_cols)}")
    log(f"Saved to            : {out_path}  "
        f"({out_path.stat().st_size / 1e6:.1f} MB)")
    (RESULTS_DIR / "preprocessing_report.txt").write_text("\n".join(lines),
                                                          encoding="utf-8")
if __name__ == "__main__":
    main()