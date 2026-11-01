# How this project works (simple explanation)

Imagine a librarian who reads a pile of books and then answers questions about them. Each file in this project is one job that librarian does. Below is what each file does, with a tiny example.

---

## The big picture

You give the program some files, like a mower manual or a refund policy. Later you ask a question such as "What does error E17 mean?" and get back:

> Error E17 means the blade is blocked. [1]
> Sources: [1] atlas mower manual (page 1)

The `[1]` tells you which file the answer came from, so you can check it.

The work happens in two phases:

1. **Preparing (once):** read the files, cut them into small pieces and build lookup tables. This is like a librarian reading every book and writing index cards.
2. **Answering (every question):** find the best pieces and write an answer from them.

---

## Phase 1: Preparing

### [text.py](rag/text.py) cleans up text

The same word can be typed in slightly different ways. This file makes them look identical so they can match.

```
"كتاب"  (Arabic letters)   →  "کتاب"  (Persian letters)
"۲۰۲۵"  (Persian digits)   →  "2025"
"The BATTERY is   dead."   →  ["battery", "dead"]
```

The last line shows it also splits a sentence into single words, lowercases them and throws away filler words like "the" and "is". It can also split a paragraph into sentences:

```
"Dr. Lena works here. She is nice."  →  ["Dr. Lena works here.", "She is nice."]
```

It knows the "." after "Dr" isn't the end of a sentence. Every other file uses this one, so they all see text the same way.

### [ingest.py](rag/ingest.py) reads the files

It opens each file and pulls out the plain text.

```
manual.pdf  →  page 1: "Error E17: blade blocked..."
               page 2: "Charging: plug the dock..."
```

It also writes a little label for each file:

```
{ "filename": "manual.pdf", "type": "pdf", "year": 2025 }
```

You can add your own labels by putting a small file next to it, named `manual.pdf.meta.json`:

```json
{ "department": "support" }
```

If the same file appears twice, it keeps only one copy. With `--workers 8`, it reads 8 files at the same time to go faster.

### [chunking.py](rag/chunking.py) cuts text into small pieces

A whole book is too big to hand to the AI, so it cuts each page into pieces of about 200 words, always ending on a full sentence.

```
Page (600 words)  →  piece 1 (words 1–200)
                     piece 2 (words 160–360)   ← starts a bit back
                     piece 3 (words 320–520)
                     ...
```

Each piece starts a little before the previous one ended. That way, a sentence that sits right at a cut still appears whole in one piece. Each piece also remembers its file and page number.

### [bm25.py](rag/bm25.py): search by exact words

This is like the index at the back of a book.

```
"e17"     → piece 4
"battery" → pieces 2, 7, 9
```

Question: "What is E17?" Piece 4 contains "e17", so it ranks first.

Rare words count more. "E17" appears in one place, so it is a strong clue. "Battery" appears everywhere, so it is a weak clue.

### [embeddings.py](rag/embeddings.py): search by meaning

Exact words aren't enough. Take:

```
Question: "How long is the guarantee?"
Text:     "Warranty lasts 2 years."
```

No word is shared, but the meaning is the same. This file turns each piece of text into a list of numbers (for example `[0.12, -0.80, 0.33, ...]`). Texts with similar meaning get similar numbers.

```
"guarantee"  →  [0.9, 0.1, ...]
"warranty"   →  [0.9, 0.2, ...]   ← very close, so a match
"banana"     →  [0.0, 0.8, ...]   ← far away, no match
```

This also works across languages: "گارانتی" and "warranty" end up close together. If the real AI model isn't installed, it uses a simpler built-in version that is weaker but still works.

### [vector_store.py](rag/vector_store.py) stores those number lists

Its job is: "Here is the question's number list. Find the 50 pieces whose lists are closest." With a few thousand pieces it compares against every one. With hundreds of thousands it uses a faster shortcut (FAISS) if that's installed.

### [graph.py](rag/graph.py) is a map of who's connected to whom

It finds names (companies, products, people) and remembers which names appear together.

```
"Northwind sells the Atlas Mower."
"Batteries are supplied by VoltaCell."   (same paragraph)

Map:   Atlas Mower ── Northwind ── VoltaCell
```

Question: "Who makes the battery for the Atlas Mower?" It starts at "Atlas Mower" on the map, walks one step to its neighbors, finds VoltaCell, and brings back the pieces that mention VoltaCell. The other two searches might miss this because those pieces don't say "Atlas Mower".

### [filters.py](rag/filters.py) narrows the search by labels

Before searching, you can say "only look in some files":

```
{"type": "pdf"}                      → only PDF files
{"year": {"$gte": 2024}}             → only year 2024 or later ($gte means "greater than or equal")
{"department": "legal"}              → only files labeled legal
```

It builds lists ahead of time, such as "all PDF pieces = 3, 8, 12", so this step is instant.

---

## Phase 2: Answering

### [hybrid.py](rag/hybrid.py) combines the three searches

Each of the three searches (exact words, meaning, map) returns its own top list:

```
Exact-word search:  A, B, C
Meaning search:     B, D, A
Map search:         B, E
```

B is near the top of all three lists, so B wins. It's like three friends recommending restaurants: the one all three mention is probably good.

```
Combined:  B, A, D, C, E
```

It uses only positions (1st, 2nd, 3rd), not the raw scores, because each search scores on a different scale.

### [rerank.py](rag/rerank.py) takes a careful second look

The combined list has about 30 pieces. A slower, smarter AI model reads the question and each piece side by side and asks "does this piece actually answer the question?" It keeps the best 6.

```
30 rough candidates  →  careful check  →  best 6
```

It's like reading 30 résumés quickly, then interviewing the top 6.

### [generate.py](rag/generate.py) writes the answer

It hands the 6 pieces to an AI (Claude or a local one), numbered:

```
[1] atlas mower manual (page 1): Error E17: blade blocked. Remove grass...
[2] ...
Question: What does error E17 mean?
```

The instructions say: answer only from these pieces, put the `[number]` after every fact, and say "not found" if the answer isn't there. The answer comes back as:

```
Error E17 means the blade is blocked; remove the grass and restart. [1]
```

If no AI is set up, it picks the sentences that best match the question and shows them with their numbers.

### [pipeline.py](rag/pipeline.py) is the manager

It runs every step above in the right order and saves the results to a folder (`kb_store`), so you only prepare once.

```
Prepare:  read → cut → build the 3 search tools + labels → save to folder
Answer:   filter → 3 searches → combine → careful check → write answer
```

It also records how long each step took, for the `--debug` option.

### [\_\_init\_\_.py](rag/__init__.py)

This is the front door. It lets other code write `from rag import KnowledgeBase` without knowing about the inner files.

---

## How you use it

### [cli.py](cli.py) lets you type commands in the terminal

```powershell
python cli.py ingest data/sample --kb kb_store      # prepare (read your files)
python cli.py ask "What does error E17 mean?"       # ask a question
python cli.py stats                                  # how many files/pieces
python cli.py entity VoltaCell                       # who VoltaCell is connected to on the map
```

### [api.py](api.py) does the same over the web

Other programs, like a website or chatbot, can send a question to `http://localhost:8000/ask` and get the answer back.

---

## Extra files

| File | Simple explanation |
|---|---|
| [requirements.txt](requirements.txt) | Shopping list of add-ons to install (`pip install -r requirements.txt`) |
| [data/sample/](data/sample/) | Six small fake documents to try it on: an English manual, a PDF, a Persian warranty, and so on |
| `*.meta.json` | The extra labels for each sample file (department, tags) |
| [tests/test_pipeline.py](tests/test_pipeline.py) | Automatic checks, e.g. "asking about E17 must find the mower manual first" |
| [scripts_benchmark.py](scripts_benchmark.py) | Makes 100,000 fake documents to see how fast it is at large scale |
| [README.md](README.md) / [USAGE.md](USAGE.md) | Instructions, in Persian and English |

---

## Glossary

| Word | Plain meaning |
|---|---|
| RAG | Look things up first, then answer: an AI that searches your files before answering |
| Chunk | One small piece of a document (about 200 words) |
| Embedding / vector | A list of numbers that represents the meaning of a text |
| BM25 | Search by exact words, giving rare words more weight |
| GraphRAG / entity graph | The "who's connected to whom" map |
| Metadata | Labels on a file (type, year, department) |
| RRF (fusion) | Combining several ranked lists, rewarding items that rank high in many |
| Reranker / cross-encoder | The slower, careful second look at the top candidates |
| LLM | The AI that writes the answer (Claude, Llama, etc.) |
| Fallback | A simpler plan B used when the better tool isn't installed |
| Citation `[1]` | A pointer to the source the fact came from |
