import torch
from torch_geometric.data import Data
import numpy as np


def create_fraud_graph(
    num_nodes=300,
    fraud_ratio=0.15,
    heterophily=0.9,
    feature_noise=1.5,
):
    """
    Harder synthetic fraud graph:
    - High heterophily
    - Noisy overlapping features
    """

    # -----------------------------
    # Labels
    # -----------------------------
    num_fraud = int(num_nodes * fraud_ratio)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[:num_fraud] = 1
    y = y[torch.randperm(num_nodes)]

    # -----------------------------
    # Node features (INTENTIONALLY WEAK)
    # -----------------------------
    # Fraud and non-fraud heavily overlap
    x = torch.randn(num_nodes, 8) * feature_noise
    x[y == 1] += 0.3   # very small signal
    x[y == 0] -= 0.3

    # -----------------------------
    # Edges (STRONG HETEROPHILY)
    # -----------------------------
    edge_list = []

    for i in range(num_nodes):
        for _ in range(4):  # sparse graph
            if torch.rand(1).item() < heterophily:
                # connect to opposite class
                candidates = torch.where(y != y[i])[0]
            else:
                candidates = torch.where(y == y[i])[0]

            j = candidates[torch.randint(len(candidates), (1,))].item()
            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(edge_list).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)
