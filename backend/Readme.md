# Graph Fraud Detection Backend

This backend implements a research-inspired pipeline for **graph-based fraud detection under heterophily**, following ideas from AAAI 2024 literature.

## Features
- Synthetic heterophilic fraud graph generation
- GraphSAGE-based node classification
- Spectral graph analysis using Laplacian eigenvalues
- Low-pass vs high-pass spectral feature filtering
- FastAPI endpoint for running experiments

## Tech Stack
- Python
- PyTorch
- PyTorch Geometric
- FastAPI
- NumPy

## How to Run

Activate virtual environment:

```bash
.\venv\Scripts\activate
```

Start API server:

```bash
python -m uvicorn api:app --reload
```

Open:

```bash
http://127.0.0.1:8000/docs
```
- Experiment Endpoint: /run

## Output

The /run endpoint returns:

- Heterophily score

- Baseline accuracy

- Accuracy vs spectral cutoff for low-pass and high-pass filtering