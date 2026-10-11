"""Quick check that the environment is installed correctly."""
import sys

import numpy as np
import pandas as pd
import sklearn
import networkx as nx
import matplotlib
import streamlit
import psutil
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

print("Python            :", sys.version.split()[0])
print("NumPy             :", np.__version__)
print("Pandas            :", pd.__version__)
print("scikit-learn      :", sklearn.__version__)
print("NetworkX          :", nx.__version__)
print("Matplotlib        :", matplotlib.__version__)
print("Streamlit         :", streamlit.__version__)
print("PyTorch           :", torch.__version__)
print("PyTorch Geometric :", torch_geometric.__version__)
print("CUDA available    :", torch.cuda.is_available())
print("Total RAM (GB)    : %.1f" % (psutil.virtual_memory().total / 1e9))
print("Free RAM (GB)     : %.1f" % (psutil.virtual_memory().available / 1e9))

# Tiny GCN test: 4 nodes, 3 edges, 2 features each
x = torch.randn(4, 2)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

conv = GCNConv(2, 8)
out = conv(data.x, data.edge_index)
print("GCN test output shape:", tuple(out.shape), "(expected (4, 8))")
print("\nEnvironment OK")