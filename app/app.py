import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Import download helpers
from app.utils.download_data import download_artifacts, download_processed_dataset


# -------------------------------------------------
# Project paths
# -------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = PROJECT_DIR / "data/processed/documents_with_clusters.csv"

ARTIFACT_DIR = PROJECT_DIR / "data/artifacts"
FAISS_INDEX_PATH = ARTIFACT_DIR / "faiss_index.bin"
DOC_EMBEDDINGS_PATH = ARTIFACT_DIR / "document_embeddings.npy"
CLUSTER_CENTERS_PATH = ARTIFACT_DIR / "cluster_centers.npy"


# -------------------------------------------------
# Model configuration
# -------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85


# -------------------------------------------------
# FastAPI App
# -------------------------------------------------

app = FastAPI(
    title="Trademarkia Semantic Cache API",
    description="Semantic search with cluster-aware cache",
    version="1.0"
)


# -------------------------------------------------
# Request schema
# -------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


# -------------------------------------------------
# Global state
# -------------------------------------------------

df = None
index = None
cluster_centers = None
embedding_model = None

semantic_cache = {}

cache_stats = {
    "total_entries": 0,
    "hit_count": 0,
    "miss_count": 0
}


# -------------------------------------------------
# Startup event
# -------------------------------------------------

@app.on_event("startup")
def startup_event():

    global df, index, cluster_centers, embedding_model

    print("Starting Semantic Cache API...")

    # Download artifacts from Google Drive
    download_artifacts()
    download_processed_dataset()

    # Check required files
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError("documents_with_clusters.csv missing")

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError("faiss_index.bin missing")

    if not CLUSTER_CENTERS_PATH.exists():
        raise FileNotFoundError("cluster_centers.npy missing")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(DOCUMENTS_PATH)

    # Load FAISS index
    print("Loading FAISS index...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    # Load cluster centers
    print("Loading cluster centers...")
    cluster_centers = np.load(CLUSTER_CENTERS_PATH).astype(np.float32)

    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        MODEL_NAME,
        cache_folder="/tmp"
    )

    print("API startup complete")


# -------------------------------------------------
# Root endpoint
# -------------------------------------------------

@app.get("/")
def home():
    return {"message": "Trademarkia semantic search API running"}


# -------------------------------------------------
# Health check (useful for Railway)
# -------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------
# Query endpoint
# -------------------------------------------------

@app.post("/query")
def query_endpoint(request: QueryRequest):

    global semantic_cache

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # Generate query embedding
    q_emb = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )[0].astype(np.float32)

    q_emb = q_emb / np.linalg.norm(q_emb)

    # Predict cluster
    similarities = np.dot(cluster_centers, q_emb)
    query_cluster = int(np.argmax(similarities))

    # -------------------------------------------------
    # Semantic cache lookup
    # -------------------------------------------------

    best_match = None
    best_score = 0

    for cached_q, entry in semantic_cache.items():

        if entry["cluster"] == query_cluster:

            score = np.dot(q_emb, entry["embedding"])

            if score > best_score:
                best_score = score
                best_match = cached_q

    # -------------------------------------------------
    # Cache hit
    # -------------------------------------------------

    if best_match and best_score >= SIMILARITY_THRESHOLD:

        cache_stats["hit_count"] += 1

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": best_match,
            "similarity_score": float(best_score),
            "result": semantic_cache[best_match]["result"],
            "dominant_cluster": query_cluster
        }

    # -------------------------------------------------
    # Cache miss → FAISS search
    # -------------------------------------------------

    cache_stats["miss_count"] += 1

    distances, indices = index.search(np.array([q_emb]), 1)

    doc_idx = int(indices[0][0])

    if doc_idx < 0 or doc_idx >= len(df):
        raise HTTPException(status_code=500, detail="FAISS returned invalid index")

    result_text = str(df.iloc[doc_idx]["clean_text"])

    # Store in cache
    semantic_cache[query] = {
        "embedding": q_emb,
        "result": result_text,
        "cluster": query_cluster
    }

    cache_stats["total_entries"] = len(semantic_cache)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(distances[0][0]),
        "result": result_text,
        "dominant_cluster": query_cluster
    }


# -------------------------------------------------
# Cache stats endpoint
# -------------------------------------------------

@app.get("/cache/stats")
def get_stats():

    total = cache_stats["hit_count"] + cache_stats["miss_count"]

    hit_rate = cache_stats["hit_count"] / total if total > 0 else 0

    return {
        "total_entries": cache_stats["total_entries"],
        "hit_count": cache_stats["hit_count"],
        "miss_count": cache_stats["miss_count"],
        "hit_rate": round(hit_rate, 3)
    }


# -------------------------------------------------
# Clear cache endpoint
# -------------------------------------------------

@app.delete("/cache")
def clear_cache_endpoint():

    semantic_cache.clear()

    cache_stats["total_entries"] = 0
    cache_stats["hit_count"] = 0
    cache_stats["miss_count"] = 0

    return {"message": "Semantic cache cleared successfully"}
