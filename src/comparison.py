""" Baseline vs lightweight comparison over several seeds.
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = [42, 43, 44, 45, 46]               # use [42, 43, 44] for a faster run
MODELS = {"baseline": "src/baseline_gnn.py",
          "lightweight": "src/lightweight_gnn.py"}
RESULTS, RUNS, PLOTS = Path("results"), Path("results/runs"), Path("plots")
for p in (RESULTS, RUNS, PLOTS):
    p.mkdir(parents=True, exist_ok=True)
COLORS = ["#4C72B0", "#55A868"]            # baseline, lightweight
LABELS = ["Baseline GNN", "Lightweight GNN"]

# ------------------------------------------------------------ run experiments
for seed in SEEDS:
    for name, script in MODELS.items():
        out = RUNS / f"{name}_seed{seed}_results.csv"
        if out.exists():
            print(f"[skip] {name} seed {seed} already done", flush=True)
            continue
        print(f"[run ] {name} seed {seed} ...", flush=True)
        r = subprocess.run([sys.executable, script, "--seed", str(seed), "--quiet"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2500:])
            raise SystemExit(f"Run failed: {name} seed {seed}")

# ------------------------------------------------------------ collect results
runs = {}
for name in MODELS:
    frames = [pd.read_csv(RUNS / f"{name}_seed{s}_results.csv") for s in SEEDS]
    runs[name] = pd.concat(frames, ignore_index=True)
pd.concat(runs.values(), ignore_index=True).to_csv(RESULTS / "all_runs.csv",
                                                    index=False)


def series_for(df):
    o = df[df.scope == "overall"].set_index("seed")
    s5 = df[df.scope == "scenario_5"].set_index("seed")
    s11 = df[df.scope == "scenario_11"].set_index("seed")
    return {
        "Accuracy": o.accuracy, "Precision": o.precision, "Recall": o.recall,
        "F1-score": o.f1, "ROC-AUC": o.roc_auc,
        "F1 (scenario 5)": s5.f1, "F1 (scenario 11)": s11.f1,
        "Training time (s)": o.train_time_s, "Epochs run": o.epochs_run,
        "Time per epoch (s)": o.time_per_epoch_s,
        "Inference time (ms)": o.inference_ms,
        "Peak memory (MB)": o.peak_mem_mb,
        "Trainable parameters": o.params, "Model size (KB)": o.model_size_kb,
    }


ser = {name: series_for(df) for name, df in runs.items()}
COST = ["Training time (s)", "Time per epoch (s)", "Inference time (ms)",
        "Peak memory (MB)", "Trainable parameters", "Model size (KB)"]

rows = []
for m in ser["baseline"]:
    b, l = ser["baseline"][m], ser["lightweight"][m]
    rows.append({
        "metric": m,
        "baseline_mean": b.mean(), "baseline_std": b.std(),
        "lightweight_mean": l.mean(), "lightweight_std": l.std(),
        "baseline_over_lightweight": (b.mean() / l.mean()) if m in COST else np.nan,
    })
comp = pd.DataFrame(rows)
comp.to_csv(RESULTS / "comparison.csv", index=False)

print(f"\nResults over {len(SEEDS)} seeds (mean +- std), test set, same split:\n")
print(f"{'Metric':<22}{'Baseline GNN':>26}{'Lightweight GNN':>26}{'Base/Light':>12}")
for _, r in comp.iterrows():
    ratio = "" if np.isnan(r.baseline_over_lightweight) \
        else f"{r.baseline_over_lightweight:.1f}x"
    print(f"{r.metric:<22}"
          f"{r.baseline_mean:>16.4g} +- {r.baseline_std:<6.3g}"
          f"{r.lightweight_mean:>16.4g} +- {r.lightweight_std:<6.3g}{ratio:>10}")

ref_path = RESULTS / "reference_results.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path)
    ref = ref[ref.scope == "overall"][["model", "accuracy", "precision",
                                       "recall", "f1", "roc_auc"]]
    print("\nNon-graph reference (logistic regression, seed-free), overall test:")
    print(ref.to_string(index=False))


# ------------------------------------------------------------ plots
def bar_panel(ax, metric, fmt="{:.4f}", ylim=None):
    r = comp[comp.metric == metric].iloc[0]
    vals = [r.baseline_mean, r.lightweight_mean]
    errs = [r.baseline_std, r.lightweight_std]
    ax.bar(LABELS, vals, yerr=np.nan_to_num(errs), color=COLORS, capsize=5)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(metric)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelsize=8)


quality = ["Accuracy", "Precision", "Recall", "F1-score"]
lo = min((comp[comp.metric == m][["baseline_mean", "lightweight_mean"]].min(axis=1)
          - comp[comp.metric == m][["baseline_std", "lightweight_std"]].max(axis=1)
          ).iloc[0] for m in quality)
ylo = max(0.0, np.floor((lo - 0.01) * 100) / 100)
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, m in zip(axes.ravel(), quality):
    bar_panel(ax, m, ylim=(ylo, 1.015))
fig.suptitle(f"Detection metrics, mean +- std over {len(SEEDS)} seeds "
             f"(y-axis starts at {ylo:.2f})")
fig.tight_layout()
fig.savefig(PLOTS / "metrics_comparison.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (m, fmt) in zip(axes.ravel(), [
        ("Training time (s)", "{:.1f}"), ("Time per epoch (s)", "{:.3f}"),
        ("Inference time (ms)", "{:.1f}"), ("Trainable parameters", "{:,.0f}"),
        ("Model size (KB)", "{:.1f}"), ("Peak memory (MB)", "{:.0f}")]):
    bar_panel(ax, m, fmt)
fig.suptitle(f"Computational cost, mean +- std over {len(SEEDS)} seeds")
fig.tight_layout()
fig.savefig(PLOTS / "efficiency_comparison.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name, title in zip(axes, MODELS, LABELS):
    row = runs[name].query("seed == 42 and scope == 'overall'").iloc[0]
    cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "Botnet"])
    ax.set_yticks([0, 1], ["Normal", "Botnet"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} (seed 42, test set)")
fig.tight_layout()
fig.savefig(PLOTS / "confusion_matrices.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, name, title in zip(axes, MODELS, LABELS):
    h = pd.read_csv(RUNS / f"{name}_seed42_history.csv")
    ax.plot(h.epoch, h.train_loss, label="Training loss")
    ax.plot(h.epoch, h.val_loss, label="Validation loss")
    ax.set_title(f"{title} (seed 42)")
    ax.set_xlabel("Epoch")
    ax.legend()
axes[0].set_ylabel("Binary cross-entropy loss")
fig.tight_layout()
fig.savefig(PLOTS / "loss_curves.png", dpi=150)
plt.close(fig)

print("\nSaved: results/all_runs.csv, results/comparison.csv")
print("Saved plots: metrics_comparison.png, efficiency_comparison.png, "
      "confusion_matrices.png, loss_curves.png (in plots/)")