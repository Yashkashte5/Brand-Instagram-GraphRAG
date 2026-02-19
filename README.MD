# Brand Instagram GraphRAG

## Preview

<video src="demo/brandgraphrag.mp4" controls width="100%"></video>

## Overview

Brand Instagram GraphRAG is a analytics system that ingests Instagram brand data, constructs a knowledge graph from posts, and enables semantic exploration, comparison, and strategic insight generation using LLM‑assisted Graph Retrieval Augmented Generation (GraphRAG).

The project focuses on understanding brand positioning, content strategy, audience targeting, and engagement patterns through structured graph analysis rather than simple keyword search.

---

## Core Features

### 1. Instagram Data Ingestion

* Scrapes Instagram account posts via Apify actor
* Normalizes captions, hashtags, mentions, timestamps, engagement metrics
* Extracts structured entities, relationships, sentiment, and content type using LLM extraction

### 2. Knowledge Graph Construction

* Builds a directed graph representing:

  * Brand accounts
  * Posts
  * Entities (topics, products, campaigns, people, etc.)
  * Hashtags, mentions, audience signals
  * Media types, sentiment, content classification
* Community detection for thematic clustering

### 3. GraphRAG

GraphRAG (Graph Retrieval Augmented Generation) combines knowledge graph traversal with semantic vector retrieval to provide context-aware insights from Instagram brand data. Instead of relying only on embeddings, it leverages structured relationships between posts, entities, hashtags, and audiences to improve query relevance, comparative analysis, and strategic interpretation.

### 4. Interactive Visualization

* Streamlit web app
* Graph visualization with dynamic node sizing, clustering, and highlighting
* Strategic Q&A powered by LLM over graph context

---

## Project Structure

```
.
├── app.py              # Streamlit UI + GraphRAG interface
├── graph.py            # Graph construction and query logic
├── ingest.py           # Data scraping, extraction, embeddings, communities
├── requirements.txt    # Python dependencies
├── data/               # Stored scraped and processed datasets
├── demo/               # Demo assets (videos, examples)
└── .env                # API keys (not committed)
```

---

## Tech Stack

### Data + Processing

* Python
* Apify Instagram scraping
* SentenceTransformers embeddings
* LanceDB vector database
* NetworkX graph modeling

### AI + Extraction

* Groq LLM inference
* Structured entity and relationship extraction
* Sentiment and content classification

### Visualization

* Streamlit UI
* Vis.js network visualization

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/Yashkashte5/Brand-Instagram-GraphRAG
cd Brand-Instagram-GraphRAG
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
APIFY_API_TOKEN=your_token_here
```

---

## Running the Application

```bash
streamlit run app.py
```

Typical workflow:

1. Scrape an Instagram account
2. Build embeddings and graph
3. Query brand insights or comparisons
4. Explore graph visualization

---

## Example Use Cases

* Brand positioning analysis
* Competitor comparison
* Hashtag/content strategy insights
* Audience targeting analysis
* Campaign theme detection

---

## Current Limitations

* Currently optimized for limited-scale usage (few accounts recommended)
* Dependent on scraping API stability
* LLM extraction cost/latency considerations
* Graph storage is local (not distributed)

---

## Future Improvements

* Incremental data updates
* Multi‑platform social analytics
* Advanced graph embeddings
* Automated reporting dashboards
* Enterprise graph database integration

---

## License

MIT License

---

## Author

Yash Kashte

---
