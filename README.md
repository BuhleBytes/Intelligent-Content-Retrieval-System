# Intelligent Content Retrieval System

**Author:** Buhle Mlandu  
**Date:** January 2026

---

## 🌐 Quick Start - Live Demo

- **Frontend**:https://contentretrievalfrontend.vercel.app/
- **Backend API**: https://web-production-f2b40.up.railway.app

Try a query like "What is machine learning?" to see semantic search in action.

---

## 📖 Overview

A semantic search system that scrapes 4 websites, processes text into 251 chunks, generates 768D embeddings, stores them in ChromaDB, and enables natural language queries.

**Assignment Requirements Met:**

- 4 websites scraped (News, Educational, Technical, Research)
- 251 chunks (800-1200 chars, 150+ overlap)
- 768D embeddings (all-mpnet-base-v2, normalized)
- ChromaDB vector database (cosine similarity, HNSW indexing)
- 5+ diverse test queries
- **BONUS**: Web interface + LLM enhancement + APIs + Hybrid search + Re-ranking

---

## 🚀 Installation

### 1. Navigate to Project Directory

```bash
cd Mlandu_ContentRetrieval
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook (Main Deliverable)

```bash
jupyter notebook content_retrieval_system.ipynb
```

**Then:** Kernel → Restart & Run All

**Expected output:**

- Part 1: 4 websites scraped → `data/raw/`
- Part 2: 251 chunks created → `data/processed/`
- Part 3: 251 embeddings generated → `data/embeddings/`
- Part 4: ChromaDB database created → `data/chromadb/`
- Part 5: 5 test queries with results

---

## Alternative: Run Individual Scripts

**Important:** Run all scripts from the project root directory

```bash
# Make sure you're in the project root
cd Mlandu_ContentRetrieval

# Then run scripts:
python part01Scraping.py                # Scrape 4 websites
python part02TextProcessing.py          # Create 251 chunks
python part03EmbeddingsGeneration.py    # Generate embeddings
python part04VectorDB.py                # Create vector database
python part05SearchInterface.py         # Interactive search CLI
```

**Note:** Scripts use relative paths (`data/raw/`, `data/processed/`, etc.) and must be run from project root.

---

## 📊 Manual Relevance Evaluation (Testing)

**Purpose:** Compare semantic vs hybrid search performance through manual relevance ratings.

This script runs 15 test queries using both search modes and collects manual ratings (0-2 scale) for each result. It then generates comparative metrics and visualizations.

### Run Evaluation

```bash
# From project root directory
python -m testing.manualRelevanceEvaluation
```

**Process:**

- Runs 14 queries × 2 modes (semantic + hybrid) × 5 results = 140 total ratings
- You rate each result: 0 (Not Relevant), 1 (Somewhat), 2 (Highly Relevant)
- Generates comparative metrics and graphs

**Output Location:** `evaluation_results/`

- `comparison_rating_distribution.png` - Rating distribution by mode
- `comparison_rating_by_rank.png` - Quality by rank position
- `comparison_overall_metrics.png` - Overall performance comparison
- `comparative_results.csv` - Raw evaluation data
- `comparative_metrics.json` - Calculated metrics (Precision@5, MRR, etc.)

---

## 🌐 API Endpoints (BONUS)

**Live API:** https://web-production-f2b40.up.railway.app

### Core Endpoints

#### `GET /`

API information and available endpoints

#### `GET /health`

System health check

```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_loaded": true,
  "document_count": 251
}
```

#### `GET /stats`

Database statistics and content distribution

```json
{
  "total_documents": 251,
  "model": "all-mpnet-base-v2",
  "dimensions": 768,
  "categories": { ... }
}
```

#### `POST /search`

Semantic search with optional AI enhancement

```json
{
  "query": "What is machine learning?",
  "n_results": 5,
  "filter_category": "Educational", // Optional
  "enable_llm": true // Optional: AI enhancement
}
```

**Response:**

```json
{
  "query": "What is machine learning?",
  "search_mode": "semantic",
  "results": [
    {
      "text": "...",
      "similarity": 0.8234,
      "enhanced_text": "...",        // If enable_llm: true
      "relevance": "HIGH",            // If enable_llm: true
      "metadata": { ... }
    }
  ],
  "enhancement_info": {              // If enable_llm: true
    "auto_enhanced": 5,
    "pending_enhancement": 0,
    "total_tokens": 1234,
    "cost_estimate": 0.0111
  }
}
```

#### `POST /hybrid`

Hybrid search (semantic + keyword matching)

```json
{
  "query": "gorilla conservation",
  "keywords": ["gorilla", "conservation", "efforts"],
  "n_results": 5,
  "semantic_weight": 0.7,
  "keyword_weight": 0.3,
  "enable_llm": false
}
```

**Response:**

```json
{
  "search_mode": "hybrid",
  "results": [
    {
      "text": "...",
      "hybrid_score": 0.7845,
      "semantic_score": 0.8123,
      "keyword_score": 0.6667,
      "metadata": { ... }
    }
  ]
}
```

#### `POST /enhance`

On-demand enhancement for specific chunks (beyond first 5)

```json
{
  "query": "What is machine learning?",
  "results": [ ... ],              // Full results array
  "indices": [5, 6, 7]             // Chunks to enhance
}
```

#### `GET /cache/stats`

Cache performance statistics

#### `POST /cache/clear`

Clear cached results

### Try It Out

**Using curl:**

```bash
curl -X POST https://web-production-f2b40.up.railway.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "n_results": 3}'
```

**Using Python:**

```python
import requests

response = requests.post(
    'https://web-production-f2b40.up.railway.app/search',
    json={'query': 'What is machine learning?', 'n_results': 5}
)
print(response.json())
```

---

## 🌐 Run Web Interface Locally (BONUS)

**Note:** Web interface is already deployed. Local setup is optional for development/testing.

### Backend (Flask)

```bash
# Navigate to backend folder
cd backend/api/

# Backend already has .env file with API key configured

# Run server
python app.py
```

**Runs at:** http://localhost:5000

### Frontend (React)

**Note:** `npm install` may take a few minutes to download and install all node modules.

```bash
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install

# Run development server
npm start
```

**Runs at:** http://localhost:3000

---

## 📁 Project Structure

```
Mlandu_ContentRetrieval/
├── content_retrieval_system.ipynb    # Main notebook (ALL PARTS)
├── technical_report.pdf              # 3-5 page report
├── requirements.txt                  # Core dependencies
├── README.md                         # This file
├── part01Scraping.py                 # Web scraping
├── part02TextProcessing.py           # Text processing
├── part03EmbeddingsGeneration.py     # Embeddings
├── part04VectorDB.py                 # Vector database
├── part05SearchInterface.py          # Search interface
├── screenshots/                      # Visual evidence
│   ├── scraping_process.png
│   ├── data_processing.png
│   ├── vector_database.png
│   ├── search_query_1.png
│   ├── search_query_2.png
│   └── search_query_3.png
├── testing/                          # Evaluation scripts
│   └── manualRelevanceEvaluation.py  # Manual relevance testing
├── evaluation_results/               # Evaluation outputs
│   ├── comparison_rating_distribution.png
│   ├── comparison_rating_by_rank.png
│   ├── comparison_overall_metrics.png
│   ├── comparative_results.csv
│   └── comparative_metrics.json
├── data/                             # Generated data folders
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── chromadb/
├── backend/                          # Flask backend (BONUS)
│   ├── app.py
│   ├── llm_enhancer.py
│   ├── .env                          # API key (not in submission)
│   └── requirements.txt              # Backend dependencies
└── frontend/                         # React frontend (BONUS)
    ├── package.json
    ├── src/
    └── public/
```

---

## Technologies

| Component   | Technology                                |
| ----------- | ----------------------------------------- |
| Scraping    | BeautifulSoup, Requests                   |
| Processing  | ftfy, clean-text                          |
| Embeddings  | sentence-transformers (all-mpnet-base-v2) |
| Database    | ChromaDB (HNSW indexing)                  |
| Search      | Semantic + Hybrid modes                   |
| Web (Bonus) | Flask, React                              |
| LLM (Bonus) | Claude API                                |

---

## 📊 Dataset

- **4 websites** scraped (50,000+ characters)
- **251 chunks** (avg 1,000 chars, 150+ overlap)
- **768D embeddings** (normalized for cosine similarity)
- **ChromaDB** with HNSW indexing

---

## 🧪 Test Queries

1. "What is machine learning?" (Definition)
2. "How do neural networks learn?" (How-to)
3. "Supervised vs unsupervised learning" (Comparison)
4. "AI applications in healthcare" (Domain-specific)
5. "How gradient descent optimizes networks" (Procedural)

---

## 🐛 Troubleshooting

**File not found errors:**
Make sure you're in the project root directory (`Mlandu_ContentRetrieval/`)

**ChromaDB not found:**

```bash
python part04VectorDB.py
```

**Import errors:**

```bash
pip install -r requirements.txt --force-reinstall
```

**Out of memory:**
Reduce `batch_size` in `part03EmbeddingsGeneration.py` (32 → 16)

**Module not found when running testing scripts:**
Use the `-m` flag:

```bash
python -m testing.manualRelevanceEvaluation
```

---

## 👤 Author

**Buhle Mlandu**

---

## Additional Resources

- Technical Report: `technical_report.pdf`
- Screenshots: `screenshots/` folder
- Evaluation Results: `evaluation_results/` folder
