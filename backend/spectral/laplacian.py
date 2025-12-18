import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def compute_normalized_laplacian(edge_index, num_nodes):
    """
    L = I - D^{-1/2} A D^{-1/2}
    """
    row, col = edge_index.numpy()
    data = np.ones(len(row))

    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = csr_matrix(
        (deg_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))),
        shape=(num_nodes, num_nodes),
    )

    I = csr_matrix(np.eye(num_nodes))
    L = I - D_inv_sqrt @ A @ D_inv_sqrt

    return L


def compute_spectrum(L, k=20):
    """
    Compute smallest k eigenvalues of Laplacian
    """
    eigvals, eigvecs = eigsh(L, k=k, which="SM")
    return eigvals, eigvecs
