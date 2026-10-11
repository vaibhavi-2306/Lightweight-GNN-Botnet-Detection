from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
PROC = Path("data/processed")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
lines = []
def log(text=""):
    print(text)
    lines.append(str(text))
# ---------------------------------------------------------------- load flows
flows = pd.read_parquet(PROC / "flows_clean.parquet",
                        columns=["scenario", "src", "dst", "label"])
n_flows = len(flows)
log(f"Flows loaded: {n_flows:,}")
# ------------------------------------------------------------------- nodes
# Key = "scenario|ip" so the same IP in two scenarios becomes two nodes.
key_src = flows["scenario"].astype(str) + "|" + flows["src"]
key_dst = flows["scenario"].astype(str) + "|" + flows["dst"]
codes, uniques = pd.factorize(pd.concat([key_src, key_dst], ignore_index=True))
src_idx = codes[:n_flows].astype(np.int64)   # node id of each flow's source
dst_idx = codes[n_flows:].astype(np.int64)   # node id of each flow's destination
N = len(uniques)
node_scn = np.array([int(k.split("|")[0]) for k in uniques], dtype=np.int64)
node_ip = [k.split("|", 1)[1] for k in uniques]
del key_src, key_dst, codes
# ------------------------------------------------------------------- edges
# Unique directed IP pairs (many flows can share the same pair)
pair = src_idx * N + dst_idx
upair = np.unique(pair)
u_src, u_dst = upair // N, upair % N
keep = u_src != u_dst                       # drop self-loops (GCN adds its own)
u_src, u_dst = u_src[keep], u_dst[keep]
# Symmetrise for message passing and remove duplicates
sym = np.unique(np.concatenate([u_src * N + u_dst, u_dst * N + u_src]))
edge_index = np.stack([sym // N, sym % N])  # shape (2, M)
# --------------------------------------------------------- node features
out_flows = np.bincount(src_idx, minlength=N)
in_flows = np.bincount(dst_idx, minlength=N)
out_peers = np.bincount(u_src, minlength=N)   # distinct IPs this host contacts
in_peers = np.bincount(u_dst, minlength=N)    # distinct IPs contacting this host
feat_names = ["log_out_flows", "log_in_flows", "log_out_peers", "log_in_peers"]
x_raw = np.log1p(np.stack([out_flows, in_flows, out_peers, in_peers], axis=1)
                 ).astype(np.float32)
x_mean = x_raw.mean(axis=0)
x_std = x_raw.std(axis=0)
x_std[x_std == 0] = 1.0
x = (x_raw - x_mean) / x_std
# ------------------------------------------------------------------ report
log("\n" + "=" * 60)
log("GRAPH SUMMARY")
log("=" * 60)
for scn in sorted(np.unique(node_scn)):
    m = node_scn == scn
    log(f"Scenario {scn}: {int(m.sum()):,} nodes, "
        f"{int((flows['scenario'] == scn).sum()):,} flows")
log(f"Total nodes                    : {N:,}")
log(f"Total flows (directed edges)   : {n_flows:,}")
log(f"Unique directed IP pairs       : {len(u_src):,}")
log(f"Message-passing edges (2-way)  : {edge_index.shape[1]:,}")
log(f"Average degree (msg-passing)   : {edge_index.shape[1] / N:.2f}")
log(f"Density                        : {edge_index.shape[1] / (N * (N - 1)):.2e}")
adj = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                 shape=(N, N))
n_comp, comp = connected_components(adj, directed=False)
sizes = np.bincount(comp)
log(f"\nConnected components           : {n_comp:,}")
log(f"Largest component              : {sizes.max():,} nodes "
    f"({sizes.max() / N * 100:.1f}% of all nodes)")
log(f"Isolated nodes (no neighbours) : {int((sizes == 1).sum()):,}")
# Is the infected host trivially recognisable by its degree?
log("\n" + "-" * 60)
log("Infected hosts (sources of Botnet-labelled flows) - degree check")
log("-" * 60)
bot_nodes = np.unique(src_idx[flows["label"].to_numpy() == 1])
for nid in bot_nodes:
    same = node_scn == node_scn[nid]
    pct = (out_flows[same] < out_flows[nid]).mean() * 100
    log(f"scenario {node_scn[nid]:>2} | {node_ip[nid]:<16} | "
        f"flows sent {out_flows[nid]:>6,} | distinct peers {out_peers[nid]:>5,} "
        f"| sends more flows than {pct:.2f}% of hosts in its scenario")
log("\nNode feature summary (before standardising):")
log(pd.DataFrame(np.expm1(x_raw), columns=["out_flows", "in_flows",
                                           "out_peers", "in_peers"]
                 ).describe().round(2).to_string())
# -------------------------------------------------------------------- save
torch.save({
    "x": torch.from_numpy(x),
    "edge_index": torch.from_numpy(edge_index),
    "flow_src": torch.from_numpy(src_idx),
    "flow_dst": torch.from_numpy(dst_idx),
    "node_scenario": torch.from_numpy(node_scn),
    "x_mean": torch.from_numpy(x_mean),
    "x_std": torch.from_numpy(x_std),
    "node_feature_names": feat_names,
}, PROC / "graph_structure.pt")
pd.DataFrame({"node_id": np.arange(N), "scenario": node_scn, "ip": node_ip,
              "out_flows": out_flows, "in_flows": in_flows,
              "out_peers": out_peers, "in_peers": in_peers}
             ).to_parquet(PROC / "nodes.parquet", index=False)
log(f"\nSaved: {PROC / 'graph_structure.pt'}  "
    f"({(PROC / 'graph_structure.pt').stat().st_size / 1e6:.1f} MB)")
log(f"Saved: {PROC / 'nodes.parquet'}")
(RESULTS / "graph_report.txt").write_text("\n".join(lines), encoding="utf-8")