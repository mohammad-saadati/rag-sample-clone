# How to run and use the RAG knowledge base

This is a hybrid RAG knowledge base. You point it at a folder of documents (PDF, Word, HTML, text, images) and then ask questions in English or Persian. Each answer comes back with numbered citations like `[1]`, `[2]`.

Search combines three methods: keyword matching (BM25), vector similarity, and an entity graph. The results are merged and re-ranked before the answer is written.

## 1. Install

```powershell
cd C:\projects\repos\RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

These extras are optional (they're commented out in `requirements.txt`):

- `pip install fastapi uvicorn` if you want the HTTP API
- `pip install python-docx` for `.docx` files
- `pip install pytesseract pillow` plus the Tesseract binary, for OCR on images
- `pip install faiss-cpu` for faster vector search once you pass about 5,000 chunks

## 2. Build a knowledge base

This indexes the sample docs in `data/sample/` into a folder called `kb_store`:

```powershell
python cli.py ingest data/sample --kb kb_store
```

The first run downloads the embedding and reranker models. To run fully offline with no model downloads:

```powershell
python cli.py ingest data/sample --kb kb_store --embedding-model hashing-1024 --reranker heuristic
```

If you run `ingest` again on the same `--kb`, new files are added and changed files are replaced. Other useful flags are `--workers 8` (read files in parallel), `--no-graph`, and `--glossary terms.txt` (one entity per line).

## 3. Ask questions

```powershell
python cli.py ask "What does error E17 mean?" --kb kb_store
python cli.py ask "گارانتی باتری چند سال است؟" --kb kb_store
python cli.py ask "battery" --kb kb_store --filter '{\"type\": \"pdf\", \"year\": {\"$gte\": 2024}}'
```

- `--debug` shows each retriever's ranking, the scores and the timings.
- `--json` prints the raw result.
- `--k N` sets how many sources are used.

The output shows the answer and then a source list. Sources marked with `*` were actually cited in the answer.

Other commands:

```powershell
python cli.py stats --kb kb_store              # index statistics
python cli.py entity VoltaCell --kb kb_store   # an entity's neighbours in the graph
```

## 4. Choose how answers are generated

The app picks a backend in this order:

1. **Claude:** if `ANTHROPIC_API_KEY` is set. You can change the model with `RAG_LLM_MODEL`.
2. **Any OpenAI-compatible server** (for example Ollama): if `RAG_LLM_BASE_URL` is set.
3. **Extractive (no LLM):** if neither is set, it returns the most relevant sentences with citations.

In PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# or, for Ollama:
$env:RAG_LLM_BASE_URL = "http://localhost:11434/v1"; $env:RAG_LLM_MODEL = "qwen2.5"
```

## 5. Run the HTTP API (optional)

```powershell
$env:KB_DIR = "kb_store"
uvicorn api:app --reload
```

On macOS/Linux (bash) the equivalent is `KB_DIR=kb_store uvicorn api:app`.

The API has two endpoints:

- `POST /ask` with a body like `{"question": "...", "filters": {...}, "k": 6}`
- `GET /stats`

Interactive docs are at http://127.0.0.1:8000/docs.

## 6. Add metadata and filter on it

Put a sidecar file next to any document, named after it (for example `report.pdf.meta.json`):

```json
{"department": "legal", "tags": ["policy"], "language": "fa", "year": 2025}
```

You can then filter on those fields with Mongo-style operators:

- comparisons: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- lists and fields: `$in`, `$nin`, `$contains`, `$exists`
- logic: `$and`, `$or`, `$not`

The fields `type`, `year`, `department`, `language`, `tags`, `doc_id` and `filename` are indexed, so filters on them are fast.

## 7. Tests and benchmark

```powershell
python -m pytest -q
python scripts_benchmark.py 100000   # build and query timing on synthetic data
```
