# Trademarkia Semantic Search with Cluster-Aware Semantic Cache

## Project Overview

This project implements a **semantic search system** built using modern **Natural Language Processing (NLP)** and **vector similarity search techniques**.

The system retrieves relevant documents using **Sentence Embeddings**, **FAISS Vector Indexing**, **Fuzzy Clustering**, and a **custom semantic caching layer**.

The search system is exposed through a **FastAPI service**, allowing users to query the corpus using **natural language queries**.

During development, the API was temporarily exposed using **ngrok public tunnels** to allow external access and testing.

This project follows the architecture specified in the **Trademarkia AI/ML Engineer Task** and demonstrates how **clustering, vector search, and semantic caching** can work together in a scalable **semantic retrieval system**.

---

# System Architecture

**User Query**  
→ **Query Embedding Generation**  
→ **Cluster Prediction (Fuzzy C-Means)**  
→ **Semantic Cache Lookup**  
→ **FAISS Vector Search (on cache miss)**  
→ **Retrieve Top Documents**  
→ **Store Result in Semantic Cache**  
→ **Return Response via FastAPI**

---

# Repository Structure

trademarkia-semantic-search/

app/
│
├── __init__.py  
├── app.py  → **Main FastAPI service implementation**

└── utils/  
  ├── __init__.py  
  └── download_data.py → **Utility script to download large model artifacts from Google Drive**

data/

├── raw/  
│   └── twenty+newsgroups.zip → **Original dataset**

├── processed/  
│   ├── raw_documents.csv  
│   ├── cleaned_documents.csv  
│   ├── filtered_documents.csv  
│   └── documents_with_clusters.csv

└── artifacts/  
  ├── document_embeddings.npy → **Dense vector representations of documents**  
  ├── faiss_index.bin → **Vector search index**  
  ├── cluster_centers.npy → **Fuzzy clustering centroids**  
  └── cluster_membership.npy → **Cluster membership probabilities**

notebooks/

└── pipeline.ipynb → **End-to-end development and experimentation notebook**

Other files

Dockerfile → **Container build instructions**  
docker-compose.yml → **Container orchestration**  
requirements.txt → **Python dependencies**  
.gitignore  
LICENSE  
README.md

---

# Complete System Pipeline

This section explains the **end-to-end machine learning pipeline** implemented in this project.

---

# 1. Dataset Collection

The dataset used in this project is the **20 Newsgroups Dataset**, obtained from the **UCI Machine Learning Repository**.

The dataset contains approximately **20,000 news articles** across **20 discussion categories**.

Example categories include:

- **Politics**
- **Religion**
- **Sports**
- **Computer Hardware**
- **Space**
- **Firearms**

The dataset is distributed as a **compressed archive of raw text documents**, which must be parsed and structured before further processing.

---

# 2. Data Preprocessing

The raw dataset contains several types of **noise and irrelevant metadata**, including:

- **Email headers**
- **Formatting artifacts**
- **Punctuation**
- **Stopwords**

To improve the quality of the downstream semantic representations, a **text preprocessing pipeline** was implemented.

### Preprocessing Steps

The following transformations were applied:

• **Lowercasing**  
• **Punctuation removal**  
• **Stopword removal**  
• **Metadata stripping**  
• **Whitespace normalization**

### Generated Intermediate Files

The preprocessing pipeline produces several intermediate datasets:

**raw_documents.csv** → structured version of the original dataset  
**cleaned_documents.csv** → cleaned textual content  
**filtered_documents.csv** → final dataset after noise filtering

Documents that were **too short, noisy, or semantically weak** were removed to ensure **high-quality embeddings**.

---

# 3. Document Embedding Generation

To enable **semantic similarity search**, each document is converted into a **dense vector embedding**.

### Embedding Model

**Sentence Transformers**

Model used:

**all-MiniLM-L6-v2**

Library:

**sentence-transformers**

Embedding dimension:

**384-dimensional vector representation**

This model converts each document into a **semantic vector representation** capturing contextual meaning rather than simple keyword presence.

### Generated Artifact

document_embeddings.npy

This file contains the **vector embeddings for all documents in the dataset**.

---

# 4. Vector Database using FAISS

To efficiently search across thousands of embeddings, a **vector similarity index** is built using **FAISS (Facebook AI Similarity Search)**.

### Technology Used

**FAISS Vector Database**

### Index Type

**Flat L2 Index**

FAISS enables **fast nearest-neighbour search** across high-dimensional embeddings.

When a query embedding is generated, FAISS retrieves the **most semantically similar documents**.

### Generated Artifact

faiss_index.bin

This binary file contains the **precomputed vector search index**.

---

# 5. Fuzzy Clustering of Documents

Real-world documents often belong to **multiple semantic topics simultaneously**.

For example:

A document discussing **gun legislation** may relate to both **politics** and **firearms**.

Therefore, traditional **hard clustering algorithms** such as K-Means are not suitable.

Instead, this project uses **Fuzzy C-Means Clustering**.

### Algorithm Used

**Fuzzy C-Means**

Library used:

**scikit-fuzzy**

### Key Concept

Instead of assigning each document to a single cluster, **Fuzzy C-Means assigns membership probabilities**.

Example:

Document A →  
Cluster 1 → 0.2  
Cluster 3 → 0.7  
Cluster 7 → 0.1

This allows documents to belong **partially to multiple clusters**, which better reflects real semantic relationships.

### Generated Artifacts

cluster_centers.npy  
cluster_membership.npy

These cluster assignments are later used to **optimize semantic cache lookup**.

---

# 6. Semantic Cache Design

Traditional caches fail when **queries are phrased differently but have the same meaning**.

Example:

"How do graphics cards work?"

vs

"Explain GPU architecture"

A keyword-based cache would treat these as **different queries**.

To solve this, the project implements a **semantic cache from first principles**.

### Core Idea

Queries are compared using **embedding similarity** instead of exact string matching.

### Cache Workflow

When a query arrives:

1. The query is converted into a **vector embedding**
2. The system checks similarity against **previous cached query embeddings**
3. If similarity exceeds a **predefined threshold**, it becomes a **cache hit**
4. Otherwise the system performs a **FAISS vector search**

The result is then **stored in the semantic cache**.

### Additional Optimization

The **dominant cluster** of the query is used to **limit cache search scope**, improving lookup efficiency as the cache grows.

### Cache Metrics

The system tracks the following metrics:

- **Total entries**
- **Cache hits**
- **Cache misses**
- **Hit rate**

---

# 7. FastAPI Service

The search system is exposed as a **REST API** using **FastAPI**.

### Framework

**FastAPI**

### ASGI Server

**Uvicorn**

### Implemented Endpoints

POST /query

Accepts a **natural language query** and returns search results.

Example Response:

{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}

---

GET /cache/stats

Returns **semantic cache statistics**.

Example:

{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}

---

DELETE /cache

Clears all **cache entries and statistics**.

---

# Handling Large Model Artifacts (FAISS + Embeddings)

The files

faiss_index.bin  
document_embeddings.npy

are **large binary artifacts** generated during the pipeline.

Because of **GitHub repository size limits**, these files cannot be uploaded directly.

### Solution Implemented

The artifacts were uploaded to **Google Drive**, and **public download links** were generated.

A helper function was created inside:

app/utils/download_data.py

This module **automatically downloads the required artifacts** when the API starts.

### Workflow

When **FastAPI starts**:

1. The system checks if required artifacts exist locally
2. If not, the **download_data() function** downloads them from **Google Drive**
3. Files are saved inside the **artifacts directory**
4. FAISS index and embeddings are loaded normally

### Advantages

This approach ensures:

• **GitHub repository remains lightweight**  
• **Artifacts are automatically fetched during deployment**  
• **Future deployment environments remain reproducible**

This technique is commonly used in **machine learning deployment pipelines**.

---

# Temporary Public API via ngrok

During development, the FastAPI service running inside **Google Colab** was exposed to the internet using **ngrok tunnels**.  
This allowed the API to be accessed publicly for testing without deploying it to a cloud server.

The tunnels were created in the notebook at:

**Step 47.2 – Creating the Public API Tunnel**  
**Step 47.3 – Accessing API Documentation Endpoints**

---

## Step 47.2 – Public API Endpoint

When the FastAPI server running on **localhost:8000** was exposed using ngrok, the following public tunnel was generated:

Public API URL

https://arleen-unreprehensible-nontropically.ngrok-free.dev  
→ forwards to → http://localhost:8000

This URL allowed external users to send API requests to the FastAPI service running inside the Colab environment.

---

## Step 47.3 – API Documentation Endpoints

FastAPI automatically provides interactive documentation interfaces.  
Once the ngrok tunnel was created, the following documentation endpoints became publicly accessible.

### Swagger API Documentation

https://arleen-unreprehensible-nontropically.ngrok-free.dev/docs  
→ forwards to → http://localhost:8000/docs

This interface allows users to **interactively test the API endpoints directly from the browser**.

---

### OpenAPI Schema

https://arleen-unreprehensible-nontropically.ngrok-free.dev/openapi.json  
→ forwards to → http://localhost:8000/openapi.json

This endpoint exposes the **machine-readable OpenAPI specification** of the API.

---

### Important Note

These URLs are **temporary**.

They remain active **only while the Colab runtime and ngrok process are running**.

Once the session stops, the **public URL becomes invalid**.

For real-world deployment, the service would typically be hosted using:

- **Docker containers**
- **Cloud servers**
- **Managed platforms such as AWS, GCP, or Azure**

---

# Running the Project

Install dependencies:

pip install -r requirements.txt

Start the API server:

uvicorn app.app:app --host 0.0.0.0 --port 8000

API documentation will be available at:

http://localhost:8000/docs

---

# Docker Deployment (Optional)

Build Docker image:

docker build -t trademarkia-api .

Run container:

docker run -p 8000:8000 trademarkia-api

---

# Author

**Syed Muskan**  
23BCE7305
B.Tech Computer Science (AI & ML)  
Pre-Final Year Student  
VIT-AP University
