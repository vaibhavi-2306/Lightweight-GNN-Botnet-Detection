"""Lightweight GNN: 1 GCN layer, 16 hidden units."""
import argparse

from common import run_experiment, SEED

LIGHTWEIGHT_CONFIG = {
    "name": "lightweight",
    "hidden": 16,
    "n_layers": 1,
    "head_hidden": 16,
    "dropout": 0.2,          # same as baseline
    "lr": 0.01,              # same as baseline
    "weight_decay": 5e-4,    # same as baseline
    "max_epochs": 200,       # same as baseline
    "patience": 20,          # same as baseline
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_experiment(LIGHTWEIGHT_CONFIG, seed=args.seed, verbose=not args.quiet)