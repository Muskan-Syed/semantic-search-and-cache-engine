
# Trademarkia Semantic Search with Cluster-Aware Semantic Cache

## Project Overview

This project implements a **production-style semantic search system** using modern Natural Language Processing and vector search techniques. 
The system retrieves relevant documents using **sentence embeddings**, **FAISS vector indexing**, **fuzzy clustering**, and a **semantic caching mechanism** to improve query performance.

The API is built using **FastAPI** and exposed publicly via **ngrok** for testing.

---

# System Architecture

User Query → Sentence Embedding → Cluster Prediction (Fuzzy C-Means) → Semantic Cache Lookup → FAISS Vector Search (if cache miss) → Return Result + Store in Cache.

---

# Repository Structure

```text
trademarkia-semantic-search/
├── app/
│   └── app.py                      # FastAPI application core logic
├── data/
│   ├── raw/
│   │   └── twenty+newsgroups.zip    # Original dataset archive
│   ├── processed/
│   │   ├── raw_documents.csv       # Initial loaded dataset
│   │   ├── cleaned_documents.csv   # Text-processed dataset
│   │   ├── filtered_documents.csv  # Final refined dataset
│   │   └── documents_with_clusters.csv # Clustered results
│   └── artifacts/
│       ├── document_embeddings.npy # Generated vector embeddings
│       ├── faiss_index.bin         # Efficient search index
│       ├── cluster_centers.npy     # Fuzzy C-Means centroids
│       └── cluster_membership.npy  # Membership scores
├── notebooks/
│   └── pipeline.ipynb              # Development and evaluation notebook
├── Dockerfile                      # Containerization instructions
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Project dependencies
├── .gitignore                      # Files excluded from Git tracking
├── LICENSE                         # Project license/copyright notice
└── README.md                       # Project documentation
```

---

# Key Features

### Semantic Document Retrieval
Uses **SentenceTransformer embeddings** to capture semantic meaning rather than simple keyword matching.

### FAISS Vector Search
Efficient approximate nearest neighbor search for fast document retrieval across 20,000 documents.

### Fuzzy Clustering
Groups documents using **Fuzzy C-Means**, allowing for more nuanced semantic categorization.

### Cluster-Aware Semantic Cache
New queries first check the cache within their predicted clusters. This significantly reduces response time for repeated or semantically similar queries.

---

# Running the Project

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Start the API Server
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

---

# Docker Deployment

## Build Docker Image
```bash
docker build -t trademarkia-api .
```

## Run Docker Container
```bash
docker run -p 8000:8000 trademarkia-api
```

---

# API Documentation

FastAPI automatically generates documentation at:  
`http://localhost:8000/docs`

---

# Author

**Syed Muskan**  
B.Tech Computer Science (AI & ML) - PreFinal Year Student  
VIT-AP UNIVERSITY
