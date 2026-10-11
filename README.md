# Lightweight GNN-Based Botnet Detection System

Social Network Analysis project (topic: *Deep Learning for Community Detection*)

A baseline Graph Neural Network (GCN) and a lightweight GNN are trained on the same
network-flow graph built from the CTU-13 botnet dataset, with the same train/validation/test
split. Detection quality and computational cost are compared. A Streamlit dashboard shows the
saved results.

All numbers in this README were measured by running the code in this repository.

## 1. Problem

Detect botnet-related traffic in network flow data using a graph-based deep learning model.

## 2. Motivation

GNNs can use communication structure (who talks to whom) in addition to per-flow features.
Standard GNNs can be costly on small devices or large networks, so this project investigates
whether a smaller GNN keeps similar detection quality at lower computational cost.

## 3. Research question

Can a lightweight GNN detect botnet flows with detection performance comparable to a
standard (baseline) GNN while needing fewer parameters and less compute? The project does not
assume the lightweight model wins. It measures the trade-off.

## 4. Relation to the course topic

The project applies deep learning on graph-structured data (hosts as nodes, flows as edges) and
studies its computational efficiency. It is graph-based **edge classification**, not a community
detection algorithm, and it does not claim to solve the challenges discussed in the review paper
"A systematic review of deep learning methods for community detection in social networks", which
inspired the efficiency/scalability angle.

## 5. Dataset

CTU-13 (Stratosphere Laboratory, Czech Technical University), bidirectional NetFlow files, two
scenarios:

| Scenario | Malware | Flows | Botnet | Normal | Background (unlabeled) |
|---|---|---|---|---|---|
| 5 | Virut | 129,832 | 901 | 4,679 | 124,252 |
| 11 | Rbot | 107,251 | 8,164 | 2,718 | 96,369 |

Each file has 33 columns (IP addresses, ports, protocol, TCP state, packet/byte counts, rates,
timing, label). The raw data is not included in this repository.

**Why CTU-13.** It contains real botnet captures with labeled botnet flows and IP addresses,
which allow a natural host/flow graph, and the scenarios are small enough for an 8 GB RAM CPU
laptop. Other candidates considered: BoT-IoT, TON_IoT, CSE-CIC-IDS2018, CICIDS2017.

**Label strategy.** Binary: Botnet = 1, Normal = 0. Background flows are unlabeled in CTU-13, so
they are kept in the graph as structure but excluded from the loss and from all metrics.

## 6. Preprocessing

- No duplicate rows, infinite values or unparseable timestamps were found.
- 13 numeric columns (packet/byte counts, rates, durations, TCP timing) are log1p-transformed
  and standardized using training flows only.
- Protocol, TCP state and port range are one-hot encoded (35 flow features in total).
- IP addresses, TTL, hop counts, window sizes and ToS are not used as model inputs.
- ICMP "ports" are hexadecimal type/code values and are treated as "no usable port".

## 7. Graph construction

| Element | Definition |
|---|---|
| Nodes | one per (scenario, IP address); the same IP in two scenarios is two nodes |
| Edges | each flow is a directed edge source IP to destination IP (237,083 flows) |
| Message-passing graph | unique IP pairs, made undirected, without duplicates (93,434 directed pairs, 180,920 edges) |
| Node features (4) | log-scaled number of flows sent, flows received, distinct peers contacted, distinct peers received from |
| Edge features | the 35 flow features |
| Task | classify each labeled flow edge as botnet or normal |

The graph has 83,589 nodes, an average degree of 2.16 and 147 connected components. It is stored as
a sparse edge index. Node features are computed from all flows of a capture but never from labels
(a transductive setup).

Why edge classification: only one host (scenario 5) or three hosts (scenario 11) are infected, so
node classification would have almost no positive nodes to split.

## 8. Train / validation / test split

Blocked time split, identical for both models:

- Each scenario's timeline is cut into 40 contiguous blocks with equal numbers of labeled flows
  (scenario 11's botnet flows are concentrated in a short burst, so equal-duration blocks would
  place them in one split).
- Blocks containing botnet flows and blocks without them are each shuffled (seed 42) and assigned
  60/20/20 to train/validation/test.
- Result: 9,877 training, 3,292 validation and 3,293 test flows. The test set has 1,762 botnet and
  1,531 normal flows. Every scenario/split contains both classes.

## 9. Models

Both models share the same code (`src/common.py`). Only the size settings differ.

| | Baseline GNN | Lightweight GNN |
|---|---|---|
| GCN layers | 2 | 1 |
| Hidden size | 64 | 16 |
| Classifier head hidden size | 64 | 16 |
| Trainable parameters | 15,041 | 1,185 |

Architecture: GCN layers produce IP embeddings. For each flow, the source embedding, destination
embedding and the 35 flow features are concatenated and passed through a small MLP that outputs
the botnet probability.

Shared training settings: Adam (learning rate 0.01, weight decay 5e-4), binary cross-entropy,
dropout 0.2, full-batch training, up to 200 epochs, early stopping on validation loss with
patience 20, restoring the best weights. Seeds 42 to 46 were used.

## 10. Evaluation

Accuracy, precision, recall, F1 (botnet is the positive class), ROC-AUC and confusion matrices on
the test set, overall and per scenario. Computational cost: trainable parameters, model size,
training time, time per epoch, inference time (median of 30 full test-set passes after warm-up)
and peak process memory.

## 11. Results

Test set, mean ± standard deviation over 5 seeds (same graph and split):

| Metric | Baseline GNN | Lightweight GNN | Baseline / Lightweight |
|---|---|---|---|
| Accuracy | 0.9988 ± 0.0000 | 0.9991 ± 0.0006 | |
| Precision | 0.9977 ± 0.0000 | 0.9984 ± 0.0010 | |
| Recall | 1.0000 ± 0.0000 | 0.9999 ± 0.0003 | |
| F1-score | 0.9989 ± 0.0000 | 0.9992 ± 0.0006 | |
| ROC-AUC | 0.9984 ± 0.0014 | 0.9995 ± 0.0012 | |
| F1, scenario 5 | 0.9714 ± 0.0000 | 0.9785 ± 0.0143 | |
| F1, scenario 11 | 1.0000 | 1.0000 | |
| Trainable parameters | 15,041 | 1,185 | 12.7x |
| Model size | 58.75 KB | 4.63 KB | 12.7x |
| Inference time | 139.84 ± 3.11 ms | 21.19 ± 0.54 ms | 6.6x |
| Time per epoch | 1.66 ± 0.11 s | 0.23 ± 0.03 s | 7.2x |
| Training time | 96.71 ± 59.56 s | 22.16 ± 3.72 s | 4.4x |
| Epochs run | 58.8 ± 36.8 | 96.8 ± 13.7 | |
| Peak process memory | 802.94 ± 0.29 MB | 542.34 ± 1.09 MB | 1.5x |

Non-graph reference (logistic regression, same split, overall test):

| Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 35 flow features only | 0.9875 | 0.9864 | 0.9904 | 0.9884 | 0.9988 |
| 8 endpoint node features only | 0.9988 | 0.9977 | 1.0000 | 0.9989 | 0.9974 |
| flow + node features | 0.9988 | 0.9977 | 1.0000 | 0.9989 | 0.9974 |

Plots are in `plots/`: metric comparison, computational cost, confusion matrices (seed 42) and loss
curves (seed 42).

## 12. Discussion

- **Detection.** The mean F1 difference (+0.0003 for the lightweight model) is smaller than the
  seed-to-seed variation, so detection performance is comparable. It is not evidence that the
  lightweight model is more accurate.
- **Efficiency.** The lightweight model has 12.7x fewer parameters, 6.6x lower inference time and
  7.2x lower time per epoch. Total training time is only 4.4x lower because the lightweight model
  needed more epochs on average before early stopping (96.8 vs 58.8). The baseline's training time
  varies a lot between seeds for the same reason. Peak memory is measured for the whole Python
  process, so most of it is shared overhead and it is a weak comparison.
- **Reference check.** A logistic regression that sees only the endpoint activity counts matches
  the baseline GNN's F1 (0.9989). On this dataset, detection is largely explained by how active
  each host is, so the comparison mainly shows that the lightweight GNN reaches the same detection
  level at lower cost. It does not show that graph message passing is needed.
- **Errors.** In the seed-42 baseline run, the 4 misclassified test flows are all Normal flows from
  host 147.32.80.9 to 192.168.88.159 in scenario 5, predicted as botnet.

## 13. Limitations

- Only two CTU-13 scenarios (Virut and Rbot) and one test split are used.
- Scenario 11's botnet flows are almost all one ICMP burst (8,143 of 8,164), so it supplies 96% of
  the botnet flows in the test set and dominates pooled metrics. Scenario 5's test set has only 68
  botnet flows, so its metrics are noisy.
- Few hosts are infected, and infected hosts send far more flows than typical hosts, so models may
  rely on host activity levels instead of learned botnet behaviour.
- Timing was measured on one CPU laptop (Windows, about 7.4 GB RAM) and varies between runs.
- Cross-scenario generalization (training on one scenario and testing on another) was not tested.

## 14. Future work

- Cross-scenario evaluation (train on one scenario, test on another).
- More scenarios, and a strict chronological split.
- A sweep over lightweight model sizes to map the efficiency/performance curve.
- Ablations that remove the node features, and other lightweight designs.

## 15. Project structure

```
Lightweight-GNN-Botnet-Detection/
├── app.py                      Streamlit dashboard (loads saved files, never retrains)
├── check_env.py                environment check
├── requirements.txt
├── src/
│   ├── inspect_data.py         dataset inspection report
│   ├── preprocessing.py        cleaning, labels, flow features
│   ├── graph_construction.py   nodes, edges, node features
│   ├── split.py                blocked time split and scaling
│   ├── common.py               model and training code shared by both GNNs
│   ├── evaluation.py           metrics
│   ├── baseline_gnn.py
│   ├── lightweight_gnn.py
│   ├── reference_checks.py     logistic regression reference models
│   ├── comparison.py           multi-seed comparison, tables and plots
│   ├── check_models.py         checks that saved models reload correctly
│   └── make_dashboard_data.py  dataset summary files for the dashboard
├── models/                     saved models and scaler
├── results/                    result tables and reports
├── plots/
└── data/                       raw/ and processed/ (not committed)
```

## 16. How to run

Requires Python 3.12 (developed with 3.12.0) and works on CPU.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python check_env.py
```

Download the CTU-13 bidirectional flow files (the `.binetflow.2format` files) and save them as:

```
data/raw/scenario_05.binetflow   from CTU-Malware-Capture-Botnet-46 (capture20110815-2.binetflow.2format)
data/raw/scenario_11.binetflow   from CTU-Malware-Capture-Botnet-52 (capture20110818-2.binetflow.2format)
```

Both folders are under https://mcfp.felk.cvut.cz/publicDatasets/. Then run, from the project root:

```powershell
python src/inspect_data.py          # optional: dataset report
python src/preprocessing.py
python src/graph_construction.py
python src/split.py
python src/reference_checks.py
python src/comparison.py            # trains both models with 5 seeds (about 15 minutes on CPU)
python src/check_models.py
python src/make_dashboard_data.py
streamlit run app.py
```

The `models/`, `results/` and `plots/` folders in this repository already contain the saved outputs.
The dashboard pages "Network Graph" and "Botnet Detection" additionally need
`data/processed/` (created by the steps above).

## 17. Dataset reference

CTU-13: Stratosphere Laboratory, Czech Technical University in Prague
(https://www.stratosphereips.org/datasets-ctu13). Please cite the CTU-13 authors if you reuse it.
