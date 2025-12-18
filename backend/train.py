import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Dict, List

from data.synthetic_fraud import create_fraud_graph
from models.graphsage import GraphSAGE
from utils.metrics import homophily_ratio
from spectral.laplacian import (
    compute_normalized_laplacian,
    compute_spectrum,
)
from spectral.filtering import spectral_filter_features


# --------------------------------------------------
# Helper: Train + Evaluate (returns accuracy)
# --------------------------------------------------
def eval_accuracy(x, edge_index, y):
    model = GraphSAGE(
        in_dim=x.size(1),
        hidden_dim=8,
        out_dim=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for _ in range(30):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    pred = out.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    return acc


# --------------------------------------------------
# Main Training Pipeline
# --------------------------------------------------
def train():
    device = torch.device("cpu")

    # -----------------------------
    # Load data
    # -----------------------------
    data = create_fraud_graph()
    data = data.to(device)

    assert data.x is not None
    assert data.edge_index is not None
    assert data.y is not None

    x = torch.as_tensor(data.x)
    edge_index = torch.as_tensor(data.edge_index)
    y = torch.as_tensor(data.y, dtype=torch.long)

    # -----------------------------
    # Heterophily analysis
    # -----------------------------
    heterophily = homophily_ratio(edge_index, y)
    print(f"Heterophily score: {heterophily:.4f}")

    # -----------------------------
    # Spectral analysis
    # -----------------------------
    num_nodes = x.size(0)

    L = compute_normalized_laplacian(
        edge_index.cpu(),
        num_nodes=num_nodes,
    )

    eigvals, eigvecs = compute_spectrum(L, k=20)

    print("\nGraph Laplacian Spectrum (first 10 eigenvalues):")
    for i, val in enumerate(eigvals[:10]):
        print(f"λ{i}: {val:.4f}")
    print("--------------------------------------")

    # -----------------------------
    # Accuracy vs spectral cutoff
    # -----------------------------
    cutoffs = list(range(2, 21, 2))
    low_accs = []
    high_accs = []

    for c in cutoffs:
        x_low = spectral_filter_features(
            x, eigvals, eigvecs, mode="low", cutoff=c
        )
        x_high = spectral_filter_features(
            x, eigvals, eigvecs, mode="high", cutoff=c
        )

        low_accs.append(eval_accuracy(x_low, edge_index, y))
        high_accs.append(eval_accuracy(x_high, edge_index, y))

    baseline_acc = eval_accuracy(x, edge_index, y)

    for c, la, ha in zip(cutoffs, low_accs, high_accs):
        print(f"Cutoff {c:02d} | Low-pass {la:.4f} | High-pass {ha:.4f}")

    print(f"Baseline Accuracy: {baseline_acc:.4f}")


    # -----------------------------
    # Plot results
    # -----------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(
        cutoffs,
        low_accs,
        marker="o",
        label="Low-pass filtering",
    )
    plt.plot(
        cutoffs,
        high_accs,
        marker="s",
        label="High-pass filtering",
    )

    plt.axhline(
        y=baseline_acc,
        color="gray",
        linestyle="--",
        label="No filtering baseline",
    )

    plt.xlabel("Spectral cutoff (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Spectral Cutoff under Heterophily")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_experiment() -> Dict:
    device = torch.device("cpu")

    data = create_fraud_graph()
    data = data.to(device)

    assert data.x is not None
    assert data.edge_index is not None
    assert data.y is not None

    x = torch.as_tensor(data.x)
    edge_index = torch.as_tensor(data.edge_index)
    y = torch.as_tensor(data.y, dtype=torch.long)

    heterophily = homophily_ratio(edge_index, y)

    num_nodes = x.size(0)
    L = compute_normalized_laplacian(edge_index.cpu(), num_nodes)
    eigvals, eigvecs = compute_spectrum(L, k=20)

    cutoffs = list(range(2, 21, 2))
    low_accs: List[float] = []
    high_accs: List[float] = []

    for c in cutoffs:
        x_low = spectral_filter_features(x, eigvals, eigvecs, "low", c)
        x_high = spectral_filter_features(x, eigvals, eigvecs, "high", c)

        low_accs.append(eval_accuracy(x_low, edge_index, y))
        high_accs.append(eval_accuracy(x_high, edge_index, y))

    baseline_acc = eval_accuracy(x, edge_index, y)

    return {
        "heterophily": round(float(heterophily), 4),
        "cutoffs": cutoffs,
        "low_pass_accuracy": low_accs,
        "high_pass_accuracy": high_accs,
        "baseline_accuracy": round(float(baseline_acc), 4),
    }


if __name__ == "__main__":
    result = run_experiment()
    print(result)

