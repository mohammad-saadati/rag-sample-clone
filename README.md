# پایگاه دانش RAG ترکیبی (BM25 + برداری + GraphRAG)

پیاده‌سازی کامل پایپ‌لاین: از فایل‌های خام (PDF، تصویر، Word، HTML، متن) تا پاسخ مستند با ارجاع شماره‌دار. فارسی و انگلیسی هر دو پشتیبانی می‌شوند.

## ساختار

```
rag/
  text.py          نرمال‌سازی فارسی/انگلیسی، توکن‌سازی، تقسیم جمله
  ingest.py        مرحله ۱: خواندن فایل‌ها + OCR + متادیتا (فایل کناری .meta.json)
  chunking.py      مرحله ۲: قطعه‌بندی جمله‌محور با هم‌پوشانی، به تفکیک صفحه
  filters.py       مرحله ۳: فیلتر متادیتا به سبک Mongo با ایندکس معکوس
  bm25.py          مرحله ۴الف: جستجوی واژگانی Okapi BM25
  embeddings.py    مرحله ۴ب: بردارسازی (multilingual-e5 یا جایگزین آفلاین)
  vector_store.py  مرحله ۴ب: ایندکس برداری (FAISS HNSW یا NumPy)
  hybrid.py        مرحله ۵: ادغام رتبه‌ها با RRF
  rerank.py        مرحله ۶: رتبه‌بندی مجدد با cross-encoder
  graph.py         مرحله ۷: GraphRAG — گراف موجودیت‌ها و گسترش همسایگی
  generate.py      مرحله ۸: تولید پاسخ با ارجاع [n]
  pipeline.py      KnowledgeBase: همه‌چیز را به هم وصل می‌کند
cli.py             رابط خط فرمان
api.py             API با FastAPI
tests/             تست‌ها
scripts_benchmark.py  تست مقیاس با داده‌ی مصنوعی
```

## نصب

```bash
pip install -r requirements.txt
```

بدون `sentence-transformers` هم اجرا می‌شود (بردارساز hashing و reranker ابتکاری)، ولی کیفیت با مدل‌های واقعی خیلی بهتر است.

## استفاده

```bash
# ساخت پایگاه دانش (بار اول مدل‌ها دانلود می‌شوند)
python cli.py ingest data/sample --kb kb_store

# حالت کاملاً آفلاین
python cli.py ingest data/sample --kb kb_store --embedding-model hashing-1024 --reranker heuristic

# پرسش
python cli.py ask "What does error E17 mean?" --kb kb_store --debug
python cli.py ask "گارانتی باتری چند سال است؟" --kb kb_store
python cli.py ask "battery" --kb kb_store --filter '{"type": "pdf", "year": {"$gte": 2024}}'

# همسایگی یک موجودیت در گراف
python cli.py entity VoltaCell --kb kb_store

# API
KB_DIR=kb_store uvicorn api:app
```

با اجرای دوباره‌ی `ingest` روی همان `--kb`، فایل‌های جدید اضافه و فایل‌های تغییرکرده جایگزین می‌شوند.

## مدل زبانی برای تولید پاسخ

به ترتیب اولویت:

1. `ANTHROPIC_API_KEY` تنظیم شده باشد → Claude (مدل با `RAG_LLM_MODEL` قابل تغییر است)
2. `RAG_LLM_BASE_URL` تنظیم شده باشد → هر سرور سازگار با OpenAI، مثل Ollama:
   `RAG_LLM_BASE_URL=http://localhost:11434/v1 RAG_LLM_MODEL=qwen2.5`
3. هیچ‌کدام → پاسخ استخراجی (مرتبط‌ترین جمله‌ها با ارجاع)

## متادیتا

کنار هر فایل می‌توانید `نام‌فایل.meta.json` بگذارید:

```json
{"department": "legal", "tags": ["policy"], "language": "fa"}
```

عملگرهای فیلتر: `$eq $ne $gt $gte $lt $lte $in $nin $contains $exists` و `$and $or $not`.
فیلدهای `type, year, department, language, tags, doc_id, filename` ایندکس معکوس دارند و سریع فیلتر می‌شوند.

## برای ۱۰۰ هزار سند

- `pip install faiss-cpu` تا جستجوی برداری با HNSW انجام شود (بالای ۵۰۰۰ قطعه خودکار فعال می‌شود).
- `--workers 8` برای خواندن موازی فایل‌ها.
- روی GPU بردارسازی چند برابر سریع‌تر است (sentence-transformers خودکار از CUDA استفاده می‌کند).
- برای GraphRAG با کیفیت بالاتر در متن فارسی، یا فهرست اصطلاحات بدهید (`--glossary terms.txt`) یا `graph_extractor="llm"` را در `KBConfig` فعال کنید (کندتر و هزینه‌دار).
- در این مقیاس جایگزین‌کردن ذخیره‌سازی pickle با یک پایگاه برداری (Qdrant، pgvector، Weaviate) منطقی است؛ رابط `VectorStore` را فقط کافی است عوض کنید.

`python scripts_benchmark.py 100000` زمان ساخت و پرسش را روی داده‌ی مصنوعی نشان می‌دهد.

## تست

```bash
python -m pytest -q
```
