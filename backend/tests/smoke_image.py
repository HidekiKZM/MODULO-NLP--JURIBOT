"""Run inside the built image with network disabled and model cache empty."""
import sys

sys.path.insert(0, "/app")

import backend.app_main
import backend.ingest.prepare_index
import sentence_transformers
import torch
from fastapi.testclient import TestClient

assert torch.version.cuda is None, "The default image must use CPU PyTorch"
assert "/chat" in backend.app_main.app.openapi()["paths"]
with TestClient(backend.app_main.app) as client:
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["embeddings"] == "unavailable"
    assert response.json()["checks"]["qdrant"] == "unavailable"
    assert response.json()["checks"]["gemini"] == "disabled"
    assert client.get("/ui/").status_code == 200
    print("Offline startup verified:", response.json())
