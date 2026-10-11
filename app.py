"""
Streamlit dashboard - Lightweight GNN-Based Botnet Detection (CTU-13).
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")          # so we can import the model class from src/common.py

RESULTS, MODELS, PLOTS = Path("results"), Path("models"), Path("plots")
PROC = Path("data/processed")
MODEL_NAMES = {"Baseline GNN": "baseline", "Lightweight GNN": "lightweight"}

st.set_page_config(page_title="Lightweight GNN Botnet Detection", layout="wide")


# ----------------------------------------------------------------- helpers
def need(path):
    """Stop with a clear message if a required file is missing."""
    if not Path(path).exists():
        st.error(f"Missing file: {path}. Run the earlier pipeline steps first.")
        st.stop()
    return Path(path)


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data
def read_json(path):
    return json.loads(Path(path).read_text())


@st.cache_data
def load_nodes():
    return pd.read_parquet(PROC / "nodes.parquet").set_index("node_id")


@st.cache_resource
def load_dataset():
    import torch
    return torch.load(need(PROC / "dataset.pt"), weights_only=True)


def fmt(mean, std):
    """Format 'mean +- std' sensibly for metrics, times and counts."""
    if mean >= 1000:
        return f"{mean:,.0f}"
    if mean <= 1.0001:
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.2f} ± {std:.2f}"


def cm_fig(tn, fp, fn, tp, title):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "Botnet"])
    ax.set_yticks([0, 1], ["Normal", "Botnet"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- pages
def page_home():
    st.title("Lightweight GNN-Based Botnet Detection System")
    st.subheader("Deep Learning for Community Detection - Social Network Analysis project")
    st.markdown("""
**Research question.** Can a lightweight Graph Neural Network detect botnet traffic
with detection performance comparable to a standard GNN, while needing less computation?

**Objective.** Train a baseline GCN-based model and a smaller lightweight model on the
same graph and the same train/validation/test split, then compare detection metrics
(accuracy, precision, recall, F1) and computational cost (parameters, training time,
inference time, memory). The conclusion comes only from the measured results.

**Approach.** Hosts (IP addresses) are nodes and network flows are edges. Each flow is
classified as botnet or normal. This is graph-based edge classification. It connects to the course topic by applying deep learning
to graph-structured data and studying its computational efficiency.

**Dataset.**  Dataset 1 - CTU-13 scenario 5 (Virut) and CTU-13 scenario 11 (Rbot), bidirectional NetFlow files.

**Technologies.** Python, Pandas, NumPy, scikit-learn, PyTorch, PyTorch Geometric,
NetworkX, Matplotlib, Streamlit.
""")


def page_dataset():
    st.header("Dataset: CTU-13 (scenarios 5 and 11)")
    summary = read_json(need(RESULTS / "dataset_summary.json"))
    cfg = read_json(need(RESULTS / "experiment_config.json"))

    total_rows = sum(s["rows"] for s in summary.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total flow records", f"{total_rows:,}")
    c2.metric("Raw columns per file", next(iter(summary.values()))["columns"])
    c3.metric("Features used by the models", cfg["n_flow_features"])

    rows = []
    for scn, s in summary.items():
        rows.append({"Scenario": int(scn), "Records": s["rows"], **s["classes"]})
    dist = pd.DataFrame(rows).set_index("Scenario")
    st.subheader("Class distribution (all flows)")
    st.dataframe(dist)
    st.caption("Background flows are unlabeled in CTU-13. They stay in the graph as "
               "structure but are excluded from training and evaluation.")
    st.subheader("Labeled flows only (used for training and evaluation)")
    st.bar_chart(dist[["Normal", "Botnet"]])

    st.subheader("Botnet flow types (attack distribution)")
    types = read_csv(need(RESULTS / "botnet_label_types.csv"))
    scn = st.selectbox("Scenario", sorted(types["scenario"].unique()))
    sub = types[types["scenario"] == scn].set_index("botnet_flow_type")["flows"]
    st.bar_chart(sub)
    st.caption("Top 15 botnet label types, with the dataset's label prefix removed.")

    st.subheader("Basic statistics (raw numeric columns, both scenarios)")
    st.dataframe(read_csv(need(RESULTS / "dataset_statistics.csv")).set_index("Unnamed: 0")
                 if "Unnamed: 0" in read_csv(RESULTS / "dataset_statistics.csv").columns
                 else read_csv(RESULTS / "dataset_statistics.csv"))

    st.subheader("Train / validation / test split (labeled flows)")
    sp = []
    for key, v in cfg["split_counts"].items():
        scn_id, part = key.split("_")
        sp.append({"Scenario": int(scn_id), "Split": part, **v})
    st.dataframe(pd.DataFrame(sp))
    st.caption(f"Blocked time split: {cfg['n_blocks']} blocks per scenario, "
               f"fractions {cfg['split_fractions']}, seed {cfg['seed']}. "
               "The same split is used for both models.")


def page_graph():
    st.header("Network graph (sampled sub-network)")
    st.write("Only a small sample is drawn: one infected host, the hosts it talked to, "
             "and the flows between them. The full graph has 83,589 nodes.")
    d = load_dataset()
    nodes = load_nodes()
    fs, fd = d["flow_src"].numpy(), d["flow_dst"].numpy()
    fy, fscn = d["flow_y"].numpy(), d["flow_scenario"].numpy()

    c1, c2, c3 = st.columns(3)
    scn = c1.selectbox("Scenario", [5, 11])
    m = fscn == scn
    infected = np.unique(fs[m & (fy == 1)])
    host = c2.selectbox("Infected host", infected.tolist(),
                        format_func=lambda i: nodes.loc[i, "ip"])
    max_nb = c3.slider("Max neighbouring hosts", 10, 80, 40)

    out_m, in_m = m & (fs == host), m & (fd == host)
    nb = np.unique(np.concatenate([fd[out_m], fs[in_m]]))
    nb = nb[nb != host]
    bot_dst = np.unique(fd[out_m & (fy == 1)])
    bot_dst = bot_dst[bot_dst != host][:max_nb]
    others = np.setdiff1d(nb, bot_dst)
    room = max(max_nb - len(bot_dst), 0)
    if len(others) > room:
        others = np.random.default_rng(42).choice(others, room, replace=False)
    keep = np.unique(np.append(np.concatenate([bot_dst, others]), host)).astype(np.int64)

    em = m & np.isin(fs, keep) & np.isin(fd, keep)
    df = pd.DataFrame({"s": fs[em], "d": fd[em], "bot": fy[em] == 1})
    g = df.groupby(["s", "d"], as_index=False).agg(flows=("bot", "size"), bot=("bot", "max"))
    g = g[g["s"] != g["d"]]

    G = nx.DiGraph()
    G.add_nodes_from(keep.tolist())
    for r in g.itertuples():
        G.add_edge(int(r.s), int(r.d), bot=bool(r.bot))
    bot_set = set(bot_dst.tolist())
    pos = nx.spring_layout(G, seed=42)
    colors = ["#d62728" if n == host else "#ff7f0e" if n in bot_set else "#4C72B0"
              for n in G.nodes()]
    sizes = [500 if n == host else 120 for n in G.nodes()]
    ecolors = ["#d62728" if G.edges[e]["bot"] else "#bbbbbb" for e in G.edges()]

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ecolors, alpha=0.6,
                           arrows=True, arrowsize=8, width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes)
    labels = {n: nodes.loc[n, "ip"] for n in G.nodes() if n == host or n in bot_set}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    ax.axis("off")
    st.pyplot(fig)
    st.caption("Red node = infected host. Orange = hosts it contacted in botnet-labelled "
               "flows. Blue = other contacts (random sample). Red edges = at least one "
               "botnet-labelled flow on that host pair.")
    st.write(f"Showing {G.number_of_nodes()} nodes and {G.number_of_edges()} host pairs "
             f"out of {len(nb) + 1} hosts connected to this host.")


def page_detection():
    st.header("Botnet detection with the saved models")
    choice = st.radio("Model", list(MODEL_NAMES), horizontal=True)
    name = MODEL_NAMES[choice]
    res = read_csv(need(RESULTS / f"{name}_results.csv"))
    cfg = read_json(need(RESULTS / f"{name}_config.json"))

    st.subheader("Saved test results (seed 42)")
    st.dataframe(res[["scope", "n_flows", "accuracy", "precision", "recall", "f1",
                      "roc_auc", "tn", "fp", "fn", "tp"]])
    r0 = res.iloc[0]
    c = st.columns(4)
    c[0].metric("Trainable parameters", f"{int(r0.params):,}")
    c[1].metric("Training time (s)", f"{r0.train_time_s:.1f}")
    c[2].metric("Inference time (ms)", f"{r0.inference_ms:.1f}")
    c[3].metric("Peak memory (MB)", f"{r0.peak_mem_mb:.0f}")
    with st.expander("Architecture and hyper-parameters"):
        st.json(cfg)

    st.subheader("Run the saved model on the test flows")
    st.write("Loads the saved weights (no training) and classifies the test flows.")
    if st.button("Run inference"):
        import torch
        from common import EdgeGNN
        from evaluation import compute_metrics

        d = load_dataset()
        nodes = load_nodes()
        ckpt = torch.load(need(MODELS / f"{name}_gnn.pth"), weights_only=True)
        c_ = ckpt["config"]
        model = EdgeGNN(d["x"].shape[1], d["flow_x"].shape[1], c_["hidden"],
                        c_["n_layers"], c_["head_hidden"], c_["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        te = (d["flow_split"] == 2).nonzero(as_tuple=True)[0]
        t0 = time.perf_counter()
        with torch.no_grad():
            prob = torch.sigmoid(model(d["x"], d["edge_index"], d["flow_src"][te],
                                       d["flow_dst"][te], d["flow_x"][te])).numpy()
        ms = (time.perf_counter() - t0) * 1000
        y = d["flow_y"][te].numpy()
        mt = compute_metrics(y, prob)

        a, b, c2_, d2 = st.columns(4)
        a.metric("Accuracy", mt["accuracy"])
        b.metric("Precision", mt["precision"])
        c2_.metric("Recall", mt["recall"])
        d2.metric("F1", mt["f1"])
        st.caption(f"One pass over the test flows took {ms:.1f} ms (includes first-call "
                   "overhead; see Model Comparison for the benchmarked timing).")

        ips = nodes["ip"]
        table = pd.DataFrame({
            "scenario": d["flow_scenario"][te].numpy(),
            "source_ip": ips.loc[d["flow_src"][te].numpy()].values,
            "destination_ip": ips.loc[d["flow_dst"][te].numpy()].values,
            "true_label": np.where(y == 1, "Botnet", "Normal"),
            "botnet_probability": prob.round(4),
            "predicted": np.where(prob >= 0.5, "Botnet", "Normal"),
        })
        st.write("15 flows with the highest botnet probability:")
        st.dataframe(table.sort_values("botnet_probability", ascending=False).head(15))
        wrong = table[table["true_label"] != table["predicted"]]
        st.write(f"Misclassified test flows: {len(wrong)}")
        if len(wrong):
            st.dataframe(wrong.head(20))


def page_comparison():
    st.header("Baseline GNN vs Lightweight GNN")
    comp = read_csv(need(RESULTS / "comparison.csv"))
    n_seeds = read_csv(need(RESULTS / "all_runs.csv"))["seed"].nunique()
    st.write(f"Test-set results, mean ± standard deviation over {n_seeds} seeds, "
             "same graph and same split for both models.")
    table = pd.DataFrame({
        "Metric": comp["metric"],
        "Baseline GNN": [fmt(m, s) for m, s in zip(comp.baseline_mean, comp.baseline_std)],
        "Lightweight GNN": [fmt(m, s) for m, s in
                            zip(comp.lightweight_mean, comp.lightweight_std)],
        "Baseline / Lightweight": [f"{r:.1f}x" if not np.isnan(r) else ""
                                   for r in comp.baseline_over_lightweight],
    })
    st.dataframe(table, hide_index=True)
    st.caption("Peak memory is the peak of the whole Python process (it includes PyTorch "
               "itself), so most of it is shared overhead. Total training time depends on "
               "when early stopping triggers; time per epoch is the cleaner measure.")

    ref_path = RESULTS / "reference_results.csv"
    if ref_path.exists():
        st.subheader("Non-graph reference models (logistic regression)")
        ref = read_csv(ref_path)
        st.dataframe(ref[ref.scope == "overall"][["model", "accuracy", "precision",
                                                  "recall", "f1", "roc_auc"]],
                     hide_index=True)

    for fname, title in [("metrics_comparison.png", "Detection metrics"),
                         ("efficiency_comparison.png", "Computational cost"),
                         ("loss_curves.png", "Training and validation loss (seed 42)")]:
        if (PLOTS / fname).exists():
            st.subheader(title)
            st.image(str(PLOTS / fname))


def page_confusion():
    st.header("Confusion matrices (test set, seed 42)")
    scope = st.selectbox("Scope", ["overall", "scenario_5", "scenario_11"])
    cols = st.columns(2)
    for col, (label, name) in zip(cols, MODEL_NAMES.items()):
        res = read_csv(need(RESULTS / f"{name}_results.csv"))
        r = res[res.scope == scope].iloc[0]
        col.pyplot(cm_fig(int(r.tn), int(r.fp), int(r.fn), int(r.tp), label))
        col.write(f"F1 = {r.f1}, precision = {r.precision}, recall = {r.recall}")


def page_conclusion():
    st.header("Conclusion (generated from the saved results)")
    comp = read_csv(need(RESULTS / "comparison.csv"))
    n_seeds = read_csv(need(RESULTS / "all_runs.csv"))["seed"].nunique()

    def g(metric, col):
        return float(comp.loc[comp.metric == metric, col].iloc[0])

    f1b, f1l = g("F1-score", "baseline_mean"), g("F1-score", "lightweight_mean")
    noise = max(g("F1-score", "baseline_std"), g("F1-score", "lightweight_std"))
    diff = f1l - f1b
    if abs(diff) <= max(noise, 0.001):
        verdict = ("This difference is within seed-to-seed variation (or below 0.001), "
                   "so detection performance is comparable.")
    elif diff < 0:
        verdict = ("The lightweight model scored lower by more than the seed variation: "
                   "a performance-efficiency trade-off.")
    else:
        verdict = "The lightweight model scored higher by more than the seed variation."

    def ratio(metric):
        return g(metric, "baseline_over_lightweight")

    st.markdown(f"""
**Detection.** Over {n_seeds} seeds, mean test F1 was {f1b:.4f} for the baseline and
{f1l:.4f} for the lightweight model (difference {diff:+.4f}). {verdict}
Scenario 5 F1: {g('F1 (scenario 5)', 'baseline_mean'):.4f} (baseline) vs
{g('F1 (scenario 5)', 'lightweight_mean'):.4f} (lightweight).

**Efficiency.** The lightweight model has {ratio('Trainable parameters'):.1f}x fewer
trainable parameters ({g('Trainable parameters', 'lightweight_mean'):,.0f} vs
{g('Trainable parameters', 'baseline_mean'):,.0f}), {ratio('Inference time (ms)'):.1f}x lower
inference time, {ratio('Time per epoch (s)'):.1f}x lower time per epoch and
{ratio('Training time (s)'):.1f}x lower total training time. Process peak memory differed
by {ratio('Peak memory (MB)'):.1f}x. On average the baseline trained for
{g('Epochs run', 'baseline_mean'):.0f} epochs and the lightweight model for
{g('Epochs run', 'lightweight_mean'):.0f}, because early stopping triggered at different points.
""")

    ref_path = RESULTS / "reference_results.csv"
    if ref_path.exists():
        ref = read_csv(ref_path)
        ref = ref[ref.scope == "overall"].set_index("model")
        if "LR_node_features_only" in ref.index and "LR_flow_features_only" in ref.index:
            node_f1 = float(ref.loc["LR_node_features_only", "f1"])
            flow_f1 = float(ref.loc["LR_flow_features_only", "f1"])
            same = "matched" if abs(node_f1 - f1b) <= 0.001 else "differed from"
            st.markdown(f"""
**Reference check.** A logistic regression using only the endpoint activity counts (no
graph message passing) reached F1 = {node_f1:.4f}, which {same} the baseline GNN
({f1b:.4f}); using only the 35 flow features it reached F1 = {flow_f1:.4f}. This indicates
how much of the detection signal in this dataset comes from host activity levels.
""")

    st.subheader("Limitations")
    st.markdown("""
- Only two CTU-13 scenarios (Virut and Rbot) are used, with one test split.
- Scenario 11's botnet flows are almost all one ICMP burst, so it dominates the pooled test set.
- Few hosts are infected and infected hosts are distinctive in how much they send, so detection may rely on host activity rather than learned botnet behaviour.
- Timing was measured on one CPU laptop and varies between runs. Memory is process-level.
""")


# -------------------------------------------------------------- navigation
PAGES = {
    "Home": page_home,
    "Dataset": page_dataset,
    "Network Graph": page_graph,
    "Botnet Detection": page_detection,
    "Model Comparison": page_comparison,
    "Confusion Matrix": page_confusion,
    "Conclusion": page_conclusion,
}
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(PAGES))
PAGES[page]()