# mini_rag_qa

A retrieval-augmented generation (RAG) app that answers questions over PostgreSQL 16 documentation using OpenAI embeddings (text-embedding-3-small) and gpt-4o-mini for generation

## Architecture

```
Indexes Creation (urls.txt)
    → ingestion.py      (UnstructuredURLLoader)
    → chunking.py       (RecursiveCharacterTextSplitter)
    → embeddings.py     (text-embedding-3-small via OpenAI)
    → vector_store.py   (FAISS index, saved to disk)

Query
    → retriever.py      (similarity_search_with_score, top-k)
    → guardrails.py     (score threshold check)
    → prompt_builder.py (system prompt + context + history)
    → ChatOpenAI        (gpt-4o-mini)
    → answer + sources
```

## Data Handling

- Source: PostgreSQL 16 documentation pages listed in `data/urls.txt` (35 URLs)
- Each page is fetched via `UnstructuredURLLoader` and split into 500-token chunks with 100-token overlap
- Chunks are embedded with `text-embedding-3-small` (1536-dim) and stored in a FAISS flat L2 index persisted under `data/processed/faiss_index/`
- Document metadata retains the source page slug (e.g. `sql-select.html`) for citation

## Model Choice

| Component | Model | Reason |
|-----------|-------|--------|
| Embeddings | `text-embedding-3-small` | OpenAI's efficient embedding model, strong retrieval performance, consistent with generation API |
| Generation | `gpt-4o-mini` | Low cost, strong instruction following, good at citing context |

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Set your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# Build the FAISS index (run once)
uv run python scripts/build_index.py

# Launch the app
uv run streamlit run main.py

# Run evaluation
uv run python scripts/run_eval.py
```

## Evaluation

`scripts/run_eval.py` computes two metrics over `data/qa_dataset.json` (15 Q&A pairs):

- **Recall@k** — proxy measure: whether any of the top-k retrieved chunks contains keywords from the question
- **Answer cosine similarity** — embedding cosine similarity between the generated answer and the reference answer

## Limitations

- Recall@k uses keyword overlap as a proxy since there are no gold document IDs; a proper setup would annotate each question with the expected source chunk
- The guardrail is score-based (FAISS L2 distance threshold) and may reject borderline valid questions or pass loosely related ones
- Chat history is passed as raw text; very long conversations will eventually exceed the context window
- Index is static — adding new documentation requires re-running `build_index.py`
- `UnstructuredURLLoader` may fail on pages with heavy JavaScript rendering
