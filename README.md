# ResearchIQ

**Local-first agentic research intelligence for academic PDFs.**

ResearchIQ is a Streamlit application that ingests research papers, retrieves
page-level evidence from a local ChromaDB vector store, and runs a real
multi-agent [LangGraph](https://langchain-ai.github.io/langgraph/) workflow to
summarise, analyse, compare, cite, and report on the literature.

Everything runs on your machine:

- **LLM:** Ollama with `qwen3:4b`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** ChromaDB, persisted locally
- **Orchestration:** LangGraph `StateGraph`
- **UI:** Streamlit

No OpenAI, Groq, Anthropic, or other cloud API keys are required. LangSmith
tracing is optional and disabled by default.

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Using the app](#using-the-app)
- [Command line](#command-line)
- [How the workflow works](#how-the-workflow-works)
- [Project layout](#project-layout)
- [Data and privacy](#data-and-privacy)
- [Testing and code quality](#testing-and-code-quality)
- [Evaluation helpers](#evaluation-helpers)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Features

### Document ingestion

- Upload one or many PDFs from the UI, or ingest a file/directory from the CLI.
- Page-by-page extraction with `PyMuPDFLoader`; every chunk keeps its source
  filename, page number, content hash, document ID, and chunk ID.
- Recursive chunking with a default of **1000 characters** and **200 overlap**,
  configurable.
- SHA-256 duplicate detection: re-uploading the same file is skipped.
- Atomic document registry at `data/processed/documents.json`.
- Preview extracted page text before using a paper.
- Permanent removal deletes the stored PDF, processed text, registry record,
  and all Chroma vectors for that document.

### Evidence-grounded retrieval

- Persistent ChromaDB cosine-similarity index.
- Top-K retrieval, optional similarity threshold, and a hard context cap.
- Per-document retrieval for comparisons and reports so every selected paper is
  represented.
- Page-level citations in the form `[Source: paper.pdf, p. 4]`.
- Retrieved evidence is shown in the UI for inspection, not hidden behind the
  answer.

### Six-agent LangGraph workflow

A real compiled `StateGraph` coordinates:

1. **Supervisor Agent** — classifies intent and selects the specialist.
2. **Retrieval Agent** — searches ChromaDB and assembles evidence.
3. **Research Analysis Agent** — summaries, methodology, datasets, algorithms,
   findings, limitations, and research gaps.
4. **Comparison Agent** — structured multi-paper comparison tables.
5. **Citation Validation Agent** — deterministic citation checks plus an LLM
   grounding audit; rejects fabricated citations.
6. **Research Report Agent** — cited Markdown reports and literature reviews.

The graph also contains `direct_response` and `finalize` nodes. Validation can
route back to retrieval for a bounded number of retries. If retrieval returns no
evidence, the workflow returns exactly:

> I could not find sufficient supporting information in the uploaded research papers.

### Seven-page Streamlit workspace

- **Dashboard** — corpus size, system status, library table, recent activity.
- **Document Upload** — multi-PDF ingestion, previews, removal.
- **Research Assistant** — cited conversational Q&A with validation badges.
- **Research Analysis** — seven focused analysis actions on one paper or the
  whole corpus.
- **Paper Comparison** — compare 2–6 papers across a fixed evidence matrix.
- **Research Reports** — reports and literature reviews, saved to disk and
  downloadable as Markdown.
- **Settings** — model selection, retrieval/chunking parameters, validation
  retries, connection diagnostics, and optional LangSmith tracing.

### Engineering details

- Central typed configuration with `pydantic-settings`.
- Runtime preferences persisted separately from secrets.
- Cached embedding model and cached Streamlit services.
- Rotating file logs under `data/processed/logs/`.
- Graceful degradation when Ollama is offline; retrieval and ingestion still
  work without the LLM.
- Atomic JSON writes, content-hash deduplication, and rollback on failed
  ingestion.
- Black + isort formatting, strict pytest markers, and integration tests.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │  Streamlit UI / CLI  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   ApplicationServices │
                    │  (services.py)        │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼────────────────────────┐
        │                      │                        │
┌───────▼────────┐   ┌─────────▼─────────┐   ┌──────────▼────────┐
│ Ingestion      │   │ Retrieval         │   │ LangGraph         │
│ PyMuPDFLoader  │   │ ChromaDB          │   │ StateGraph        │
│ Splitter       │   │ Retriever         │   │ 6 agents          │
│ Registry       │   │ Citations         │   │ Retry + fallback  │
└───────┬────────┘   └─────────┬─────────┘   └──────────┬────────┘
        │                      │                        │
        └──────────────────────┼────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │ Local models                    │
              │ all-MiniLM-L6-v2 + Ollama qwen3 │
              └─────────────────────────────────┘
```

### Graph flow

```text
START
  │
  ▼
supervisor ──► direct_response ──► finalize ──► END
  │
  ├──► retrieval ──► research_analysis ──┐
  │        │                             │
  │        ├──────► comparison ──────────┤
  │        │                             │
  │        ├──────► research_report ─────┤
  │        │                             ▼
  │        └──► (no evidence) ─────► citation_validation
  │                                      │
  │                         ┌────────────┴────────────┐
  │                         │ needs more evidence?    │
  │                         └──────┬─────────────┬────┘
  │                                │ yes         │ no / limit reached
  │                                ▼             ▼
  │                            retrieval      finalize ──► END
  └──► finalize
```

Validation retries are capped by `max_validation_iterations`. The default is 2;
the Settings page allows 0–4. Every failure is normalised into the shared state
and surfaced in the UI instead of crashing the app.

### Shared state

`workflows/state.py` defines the required `ResearchState` fields:

`user_query`, `uploaded_documents`, `document_metadata`, `retrieved_documents`,
`research_intent`, `selected_agent`, `analysis_result`, `comparison_result`,
`validation_result`, `final_response`, `citations`, `iteration_count`.

Additional fields carry conversation history, retrieval context, target
documents, validation attempts, routing reasons, timestamps, and errors.

---

## Requirements

| Component | Version / notes |
| --- | --- |
| Python | 3.12 or newer |
| [uv](https://docs.astral.sh/uv/) | Recommended package manager |
| [Ollama](https://ollama.com/) | Running locally on `http://localhost:11434` |
| Ollama model | `qwen3:4b` (about 2.5 GB download) |
| RAM | 16 GB recommended; CPU-only is supported |
| Disk | About 5 GB for Python packages, models, and the vector store |

The first run downloads the Hugging Face embedding model (about 90 MB). After
that, ingestion and retrieval are fully offline. Ollama must be installed and
the model pulled once.

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repository-url> ResearchIQ_AI_Agent
cd ResearchIQ_AI_Agent
uv sync
```

`uv sync` creates `.venv` and installs the runtime and development dependencies
from `uv.lock`.

If you prefer pip, install into a Python 3.12 environment with
`pip install -e .` and also install the dev group if you want to run tests.

### 2. Start Ollama and pull the model

```bash
ollama serve
```

In a second terminal:

```bash
ollama pull qwen3:4b
```

Verify:

```bash
ollama list
```

### 3. Optional configuration

```bash
cp .env.example .env        # macOS/Linux
copy .env.example .env      # Windows
```

The defaults work without a `.env` file. See [Configuration](#configuration) for
the available values.

### 4. Check the environment

```bash
uv run python main.py check
```

This verifies Ollama connectivity, model installation, the embedding model,
the vector store, and corpus counts.

### 5. Start the app

```bash
uv run python main.py app
```

Or run Streamlit directly:

```bash
uv run streamlit run ui/streamlit_app.py
```

Open <http://localhost:8501>, go to **Document Upload**, and add PDFs.

### 6. Ingest from the command line (optional)

```bash
uv run python main.py ingest "data/my-paper.pdf"
uv run python main.py ingest "data/papers"
uv run python main.py docs
```

### 7. Run the tests

```bash
uv run pytest -q
```

Real Hugging Face embedding test:

```bash
# macOS/Linux
RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_embeddings.py -q

# Windows PowerShell
$env:RUN_INTEGRATION_TESTS='1'; uv run pytest tests/test_embeddings.py -q
```

---

## Configuration

Settings are read from environment variables and an optional `.env` file. The
Settings page persists only non-secret runtime preferences to
`data/processed/runtime_settings.json`; API keys are never written there.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_MODEL` | `qwen3:4b` | Local chat model |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature |
| `OLLAMA_NUM_CTX` | `8192` | Context window |
| `OLLAMA_NUM_PREDICT` | `2000` | Maximum generated tokens per call |
| `OLLAMA_TIMEOUT` | `600` | Per-request timeout in seconds |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `EMBEDDING_BATCH_SIZE` | `32` | Encoding batch size |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_UPLOAD_MB` | `50` | Per-file upload limit |
| `RETRIEVAL_TOP_K` | `4` | Chunks per query (or per paper for comparisons) |
| `SIMILARITY_THRESHOLD` | `0.0` | Discard chunks below this score; 0 disables |
| `MAX_CONTEXT_CHUNKS` | `12` | Hard cap on evidence chunks |
| `MAX_VALIDATION_ITERATIONS` | `2` | Maximum validation-driven retrieval retries |
| `MAX_HISTORY_MESSAGES` | `8` | Conversation turns passed to the supervisor |
| `DATA_DIR` | `./data` | Uploaded PDFs, registry, reports, logs |
| `VECTORSTORE_DIR` | `./vectorstore` | Chroma persistence directory |
| `CHROMA_COLLECTION` | `researchiq_papers` | Chroma collection name |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LANGSMITH_TRACING` | `false` | Enable optional LangSmith tracing |
| `LANGSMITH_API_KEY` | empty | Required only when tracing is enabled |
| `LANGSMITH_PROJECT` | `ResearchIQ` | LangSmith project name |

Runtime preferences saved from the Settings page override these values for
`ollama_model`, retrieval, chunking, validation retries, and LangSmith tracing.
They are validated with the same Pydantic model as environment settings.

### Optional LangSmith tracing

Tracing is strictly opt-in:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=ResearchIQ
```

If no key is present, tracing is automatically disabled. The application never
sends traces unless both a key and the toggle are set.

---

## Using the app

### Dashboard

Shows the number of papers, pages, vector chunks, and local-model status. It
also lists the research library and recent activity, with quick links to
ingestion, chat, and reports.

### Document Upload

1. Drop one or more PDFs (up to 50 MB each by default).
2. Select **Process and index selected PDFs**.
3. Watch per-file progress; duplicates are reported separately from failures.
4. Inspect any indexed paper, preview its first three extracted pages, or remove
   it with its vectors.

Scanned image-only PDFs are rejected with a clear message because the app does
not perform OCR.

### Research Assistant

Ask questions such as:

- "Explain this paper in simple language."
- "What dataset and evaluation metrics were used?"
- "What are the limitations?"
- "Identify possible research gaps."
- "Compare the findings of the uploaded papers."
- "Generate a literature review."

Answers include inline page citations, a citation-validation badge, detailed
sources, and the retrieved evidence. The supervisor routes greetings and
capability questions directly, without retrieval.

### Research Analysis

Choose one paper or the whole corpus, then run a focused action:

- Structured summary
- Methodology analysis
- Dataset extraction
- Algorithm/model extraction
- Key findings and quantitative results
- Limitations
- Research gaps and future directions

### Paper Comparison

Select two to six papers and optionally provide a comparison focus. The result
is a Markdown table covering objectives, problems, methods, datasets, models,
metrics, results, and limitations, followed by similarities, differences, and
comparison caveats. Evidence is retrieved per document so each paper is
represented.

### Research Reports

Generate a research report or literature review across up to eight papers.
Reports are validated, rendered in the UI, saved automatically under
`data/processed/reports/`, and downloadable as Markdown.

### Settings

- Choose any installed Ollama model (not only `qwen3:4b`).
- Tune Top-K, similarity threshold, and validation retries.
- Change chunk size and overlap for **newly ingested** papers.
- Toggle optional LangSmith tracing.
- Refresh Ollama diagnostics and inspect local storage paths.
- Clear conversation history or generated results.

Changing chunking does not re-chunk papers that are already indexed; re-upload
them if you want the new settings applied.

---

## Command line

```bash
uv run python main.py app [--port 8501]
uv run python main.py ingest <pdf-file-or-directory>
uv run python main.py docs
uv run python main.py remove <doc_id> [--yes]
uv run python main.py check
```

- `app` launches Streamlit.
- `ingest` copies PDFs into the managed data folder, indexes them, and reports
  duplicate/failed files.
- `docs` lists document IDs, filenames, page counts, and chunk counts.
- `remove` requires typing `remove` unless `--yes` is passed; it deletes the
  stored PDF, processed text, registry entry, and vectors.
- `check` prints Ollama, embedding, vector-store, and corpus diagnostics.

After installation, the same commands are available through the `researchiq`
entry point:

```bash
uv run researchiq check
```

---

## How the workflow works

### 1. Routing

The Supervisor asks the local model for a strict JSON decision:

```json
{
  "research_intent": "findings",
  "selected_agent": "research_analysis",
  "requires_retrieval": true,
  "retrieval_query": "...",
  "target_document_ids": ["doc_..."],
  "reasoning": "..."
}
```

The response is validated against a fixed intent/agent table. If Ollama is
unavailable or returns malformed JSON, the supervisor falls back to deterministic
keyword routing. UI actions pass an explicit intent hint, so selecting
"Compare selected papers" cannot silently route to a different specialist.

If no papers are indexed, the workflow explains how to upload them instead of
hallucinating an answer.

### 2. Retrieval

The Retrieval Agent chooses a search query, expands it for broad tasks
(comparisons, reports, literature reviews), and calls ChromaDB. Comparison and
report requests retrieve `Top-K` chunks **per selected paper**, then merge and
rank the evidence.

Each chunk carries:

- `doc_id`, `chunk_id`, `chunk_index`
- `filename`
- `page_number` (1-based) and `page` (0-based)
- similarity score

### 3. Specialists

The selected specialist writes Markdown grounded only in the retrieved context.
Prompts enforce the exact citation format and require the model to say when a
detail is not reported.

### 4. Validation

Validation combines two layers:

- **Deterministic checks** parse every `[Source: filename, p. N]` citation and
  verify that the filename/page pair exists in the retrieved evidence.
- **LLM audit** asks the local model whether claims are grounded, whether
  numbers or datasets are invented, and whether more evidence is needed.

Deterministic failures always override a lenient LLM verdict. If the LLM
validation call fails, the deterministic result is used instead of crashing. An
honest abstention using the required insufficient-evidence sentence is treated
as valid.

If validation requests more evidence and the retry budget allows it, the graph
routes back to retrieval with a refined query. After the limit, the answer is
returned with a visible validation caveat.

### 5. Finalisation

The finalizer selects the report, comparison, analysis, or direct response,
cleans any Qwen3 thinking text, and appends a validation note when claims could
not be fully supported. With no evidence, it returns the required fallback
message verbatim.

---

## Project layout

```text
ResearchIQ_AI_Agent/
├── main.py                     # CLI entry point
├── services.py                 # Backend composition root
├── pyproject.toml              # Dependencies and tool configuration
├── .env.example                # Documented local configuration
├── .streamlit/config.toml      # Dark theme and upload limit
├── agents/
│   ├── supervisor_agent.py
│   ├── retrieval_agent.py
│   ├── research_analysis_agent.py
│   ├── comparison_agent.py
│   ├── citation_validation_agent.py
│   ├── research_report_agent.py
│   └── context_utils.py
├── embeddings/
│   └── embedding_model.py      # Cached Hugging Face embeddings
├── evaluation/
│   ├── retrieval_evaluation.py # Hit rate, MRR, recall, precision
│   └── answer_evaluation.py    # Citation coverage and grounding
├── generation/
│   ├── llm.py                  # Ollama client, JSON parsing, health checks
│   └── response_generator.py   # Grounded/direct answers, final cleanup
├── ingestion/
│   ├── pdf_loader.py           # PyMuPDF page extraction
│   ├── text_splitter.py        # Recursive metadata-preserving chunking
│   ├── document_parser.py      # Document records and JSON registry
│   └── ingestion_pipeline.py   # Dedup, embedding, persistence, removal
├── prompts/                    # All agent and RAG prompts
├── retrieval/
│   ├── vector_store.py         # ChromaDB manager
│   └── retriever.py            # Search, context construction, citations
├── ui/
│   ├── streamlit_app.py        # App entry and page router
│   ├── backend.py              # Cached services and health
│   ├── session.py              # Session state and navigation
│   ├── theme.py                # CSS
│   └── components/             # One module per page plus shared helpers
├── utils/
│   ├── config.py               # Settings, overrides, LangSmith config
│   ├── helpers.py              # Hashing, JSON, citations, thinking cleanup
│   ├── logger.py               # Rotating logs
│   ├── errors.py               # Typed application errors
│   └── report_export.py        # Markdown report persistence
├── workflows/
│   ├── state.py                # ResearchState
│   ├── nodes.py                # LangGraph node adapters
│   ├── edges.py                # Conditional routing
│   └── research_graph.py       # Graph compilation and execution
├── tests/                      # Unit and workflow tests
├── data/                       # Local runtime data (git-ignored except samples)
└── vectorstore/                # Chroma persistence (git-ignored)
```

---

## Data and privacy

Local runtime data is kept under the project:

```text
data/
├── uploaded_papers/            # Managed copies of uploaded PDFs
└── processed/
    ├── documents.json          # Authoritative document registry
    ├── runtime_settings.json   # Non-secret UI preferences
    ├── reports/                # Generated Markdown reports
    ├── logs/                   # Rotating application logs
    └── <doc_id>.json           # Extracted page text for previews
vectorstore/
└── chroma.sqlite3              # Persistent embeddings and metadata
```

These paths are git-ignored so papers, logs, and generated reports do not end up
in version control.

Privacy properties:

- PDFs, extracted text, embeddings, queries, and answers stay on the machine.
- No cloud LLM or embedding API is called.
- The first embedding-model download contacts Hugging Face; afterwards the model
  is cached locally.
- LangSmith is the only optional network integration and is disabled unless both
  a key and the toggle are set.
- Ollama receives prompts and evidence locally at `localhost:11434`.

---

## Testing and code quality

```bash
uv run pytest -q                       # unit + workflow tests
uv run pytest --cov --cov-report=term  # coverage report
uv run black --check .
uv run isort --check-only .
```

Integration tests are opt-in:

```bash
# macOS/Linux
RUN_INTEGRATION_TESTS=1 uv run pytest -q

# Windows PowerShell
$env:RUN_INTEGRATION_TESTS='1'; uv run pytest -q
```

The test suite covers:

- PDF validation, page metadata, chunking, deduplication, registry, removal
- Embedding caching and error handling
- Retrieval thresholds, per-document search, citations, context
- Every agent, including malformed JSON fallbacks and citation fabrication
- Full LangGraph routing, streaming, all six agents, retries, fallback
  abstention, and graceful failure

---

## Evaluation helpers

`evaluation/retrieval_evaluation.py` computes document-level **hit rate, MRR,
recall, and precision** from a JSON case file:

```json
[
  {
    "query": "What dataset was used?",
    "relevant_doc_ids": ["doc_..."],
    "relevant_pages": [3]
  }
]
```

`evaluation/answer_evaluation.py` computes citation coverage, fabricated-citation
counts, abstention rate, and valid-answer rate for generated responses.

These modules are library helpers; they are intentionally not exposed as a
separate UI page.

---

## Performance notes

`qwen3:4b` is a reasoning model. On CPU it can spend a significant number of
tokens thinking before it produces an answer, and it may take **minutes per
agent call**. ResearchIQ is configured for that reality:

- `OLLAMA_TIMEOUT` defaults to **600 seconds**.
- `OLLAMA_NUM_PREDICT` defaults to **2000 tokens**, enough for reasoning plus an
  answer.
- The model is kept warm for 10 minutes.
- The UI shows live agent progress while the graph runs.

If your machine is faster, reduce the timeout. If calls are still timing out,
increase `OLLAMA_TIMEOUT` and `OLLAMA_NUM_PREDICT`. Using a smaller non-reasoning
model is supported through **Settings → Ollama model**, but `qwen3:4b` is the
documented and tested default.

Practical tips:

- Keep the first query short while Ollama loads the model into memory.
- Use Top-K 3–4 on 16 GB machines.
- Avoid comparing more than four papers at once on CPU.
- Enable a GPU build of PyTorch and set `EMBEDDING_DEVICE=cuda` if available.
- Close other large applications; the model itself needs several GB of RAM.

---

## Troubleshooting

### `Ollama is not reachable`

Start the server and verify the endpoint:

```bash
ollama serve
curl http://localhost:11434/api/tags
```

If you changed the port, set `OLLAMA_BASE_URL` in `.env`.

### `Model 'qwen3:4b' is not installed`

```bash
ollama pull qwen3:4b
ollama list
```

### The app is stuck on "Loading ResearchIQ models"

The first run downloads the embedding model and opens ChromaDB. Check
`data/processed/logs/researchiq.log`. Ensure the machine has internet access for
the first Hugging Face download.

### Generation times out

Increase `OLLAMA_TIMEOUT` and, if the answer is being cut off, increase
`OLLAMA_NUM_PREDICT`. CPU inference with a reasoning model is slow.

### A PDF is rejected as "not a valid PDF"

The file may be encrypted, corrupted, or not actually a PDF. Only text-based
PDFs are supported. Scanned image-only papers need OCR before upload.

### The answer says "I could not find sufficient supporting information"

This is the required grounded fallback. It means retrieval returned no chunks
that passed the current filters. Confirm the paper is indexed, lower the
similarity threshold in Settings, or ask a question closer to the paper's text.

### Citations are flagged as invalid

The validation agent compares inline citations against retrieved pages. If the
model cites a page that was not retrieved, the answer is marked for review. Open
the retrieval context panel to see which pages were actually available.

### Duplicate upload was skipped

Deduplication is based on the file's SHA-256 hash. The same content under a
different filename is still a duplicate. Remove the existing paper first if you
want to re-index it with different chunk settings.

---

## Limitations

- Text-based PDFs only; no OCR for scanned documents.
- The default embedding model is English-focused.
- Retrieval quality depends on chunking and the question; inspect retrieved
  context when an answer seems incomplete.
- `qwen3:4b` can be slow on CPU. Its reasoning tokens are separated from the
  final answer where the Ollama template supports it, and stray thinking blocks
  are stripped, but no local 4B model is error-free.
- Citation validation checks page-level support, not sentence-level entailment.
- Conversation state is per browser session and is not persisted after restart.
- Reports and extracted page previews are stored locally in plain text/Markdown.
- The app is designed for a single local user; it is not a multi-tenant server.

---

## License

MIT. See `LICENSE`.
