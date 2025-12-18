from fastapi import FastAPI
from train import run_experiment
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI(
    title="Graph Fraud Detection API",
    description="Spectral analysis under heterophily (AAAI-style)",
    version="1.0",
)

# ✅ ADD THIS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "Backend is running"}


@app.get("/run")
def run_fraud_experiment():
    """
    Run spectral fraud detection experiment
    """
    result = run_experiment()
    return result
