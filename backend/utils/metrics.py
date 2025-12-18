import torch


def homophily_ratio(edge_index, labels):
    """
    Computes homophily / heterophily score
    """
    src, dst = edge_index
    same = labels[src] == labels[dst]
    return same.float().mean().item()
