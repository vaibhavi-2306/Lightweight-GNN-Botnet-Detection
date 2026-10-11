"""Baseline GNN: 2 GCN layers, 64 hidden units."""
import argparse

from common import run_experiment, SEED

BASELINE_CONFIG = {
    "name": "baseline",
    "hidden": 64,
    "n_layers": 2,
    "head_hidden": 64,
    "dropout": 0.2,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "max_epochs": 200,
    "patience": 20,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_experiment(BASELINE_CONFIG, seed=args.seed, verbose=not args.quiet)