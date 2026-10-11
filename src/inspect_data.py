from pathlib import Path
import numpy as np
import pandas as pd
RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)
def inspect_file(path):
    lines = []
    def log(text=""):
        print(text)
        lines.append(str(text))
    log("=" * 70)
    log(f"FILE: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    log("=" * 70)
    df = pd.read_csv(path, low_memory=False)
    # 1. Shape and memory
    log(f"\nRows    : {len(df):,}")
    log(f"Columns : {df.shape[1]}")
    log(f"RAM used by this DataFrame: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    # 2. Column names and data types
    log("\n--- Columns, data types, missing values ---")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(3),
        "unique": df.nunique(),
    })
    log(info.to_string())
    # 3. Duplicates and infinite values
    log(f"\n--- Duplicate rows (all columns identical): {df.duplicated().sum():,}")
    num = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(num).sum()
    log(f"--- Infinite values in numeric columns: {int(inf_counts.sum())}")
    # 4. Label distribution (raw labels exactly as in the file)
    log("\n--- Raw label values (top 30) ---")
    log(df["Label"].value_counts().head(30).to_string())
    log(f"Number of distinct raw labels: {df['Label'].nunique()}")
    # Coarse grouping, for REPORTING ONLY (final strategy decided later)
    s = df["Label"].astype(str)
    coarse = np.select(
        [s.str.contains("Botnet", case=False),
         s.str.contains("Normal", case=False),
         s.str.contains("Background", case=False)],
        ["Botnet", "Normal", "Background"],
        default="Other",
    )
    log("\n--- Coarse grouping (Botnet / Normal / Background / Other) ---")
    counts = pd.Series(coarse).value_counts()
    pct = (counts / len(df) * 100).round(3)
    log(pd.DataFrame({"flows": counts, "percent": pct}).to_string())
    # 5. Hosts (IP addresses)
    log("\n--- Hosts ---")
    log(f"Unique source IPs      : {df['SrcAddr'].nunique():,}")
    log(f"Unique destination IPs : {df['DstAddr'].nunique():,}")
    all_ips = pd.concat([df["SrcAddr"], df["DstAddr"]]).nunique()
    log(f"Unique IPs overall     : {all_ips:,}")
    is_bot = pd.Series(coarse) == "Botnet"
    if is_bot.any():
        bot = df[is_bot.values]
        log("\nLabels that contain 'Botnet':")
        log(bot["Label"].value_counts().to_string())
        log("\nTop 10 SOURCE IPs in botnet-labelled flows:")
        log(bot["SrcAddr"].value_counts().head(10).to_string())
        log("\nTop 10 DESTINATION IPs in botnet-labelled flows:")
        log(bot["DstAddr"].value_counts().head(10).to_string())
    else:
        log("\nNo flows with 'Botnet' in the label.")
    # 6. Timestamps (errors='coerce' turns malformed ones into NaT so we can count them)
    log("\n--- Timestamps ---")
    fmt = "%Y/%m/%d %H:%M:%S.%f"
    start = pd.to_datetime(df["StartTime"], format=fmt, errors="coerce")
    last = pd.to_datetime(df["LastTime"], format=fmt, errors="coerce")
    log(f"Unparseable StartTime values: {start.isna().sum():,}")
    log(f"Unparseable LastTime values : {last.isna().sum():,}")
    log(f"Capture start: {start.min()}")
    log(f"Capture end  : {start.max()}")
    # 7. Categorical columns
    log("\n--- Categorical columns ---")
    for col in ["Proto", "State", "sTos", "dTos"]:
        log(f"\n{col} (top 10 of {df[col].nunique()} values):")
        log(df[col].value_counts(dropna=False).head(10).to_string())
    log("\nSport / Dport sample of non-numeric values (ports may contain hex or text):")
    for col in ["Sport", "Dport"]:
        non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        log(f"{col}: {int(non_numeric.sum()):,} non-numeric values, "
            f"examples: {df.loc[non_numeric, col].unique()[:5].tolist()}")
    # 8. Basic statistics of numeric columns
    log("\n--- Basic statistics (numeric columns) ---")
    log(num.describe().T.round(3).to_string())
    out_path = RESULTS_DIR / f"inspection_{path.stem}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Report saved to {out_path}]")
if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("scenario_*.binetflow"))
    if not files:
        print("No scenario_*.binetflow files found in data/raw/")
    for f in files:
        inspect_file(f)