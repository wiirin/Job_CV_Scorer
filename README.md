# CV Checker

An AI-powered resume screening tool that ranks candidate CVs against job
descriptions using **Claude Sonnet 4.6 on Google Vertex AI**, with a
built-in **GDPR-compliant PII removal layer** that strips personal data
from every CV *before* it is sent to the model. PDF CVs are supported
end-to-end via **PyMuPDF text extraction with a Tesseract OCR fallback**
for scanned documents.

The project ships as a 4-page Streamlit app:

| Page                  | What it does                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **CV Score Rank**     | Loads every JD in `job_description/` and every CV in `cv/`, anonymizes the CVs and ranks them per JD.                   |
| **GDPR Audit Log**    | Displays `pii_log.json` — a full audit trail of every PII entity that was detected and redacted.                        |
| **Job ⇄ CV Score**    | Free-text playground: paste any JD + CV, click *Remove PII*, then *Calculate Score*.                                    |
| **PDF CV Analysis**   | Upload a PDF CV → extract text (OCR fallback) → remove PII → score against your pasted job description in one click.    |

---

## Architecture

```
┌──────────────┐    ┌────────────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  PDF / TXT   │ ─▶ │ PDFTextExtractor   │ ─▶ │ PIIRemovalService │ ─▶ │  RelevancyScorer     │ ─▶ score (0.0–1.0)
│  CV input    │    │ (PyMuPDF + OCR     │    │ (spaCy NER +      │    │  (Claude Sonnet 4.6  │
└──────────────┘    │  via Tesseract)    │    │  regex patterns)  │    │   on Vertex AI)      │
                    └────────────────────┘    └─────────┬─────────┘    └──────────────────────┘
                                                        │
                                                        ▼
                                                pii_log.json
                                              (GDPR audit trail)
```

### Repository layout

```
cv-checker/
├── app.py                       # Streamlit UI — 4 pages, no business logic
├── services/
│   ├── __init__.py              # Re-exports the public API
│   ├── pii_remover.py           # PIIRemovalService (spaCy NER + regex)
│   ├── relevancy_scorer.py      # RelevancyScorer (Claude Sonnet 4.6 / Vertex)
│   └── ocr_pdf_to_text.py       # PDFTextExtractor (PyMuPDF + Tesseract OCR)
├── cv/                          # Sample CVs (plain text)
├── job_description/             # Sample job descriptions (plain text)
├── pii_log.json                 # Persistent GDPR audit log
├── pyproject.toml               # uv / pip deps
└── README.md
```

### Key modules

| Module                              | Purpose                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `services.pii_remover`              | `PIIRemovalService` — spaCy + regex PII detection, writes `pii_log.json`.                              |
| `services.relevancy_scorer`         | `RelevancyScorer` — prompt-engineered JSON-only Claude call on Vertex.                                 |
| `services.ocr_pdf_to_text`          | `PDFTextExtractor` / `extract_text_from_pdf()` — text-layer first, OCR fallback for scanned pages.     |
| `app.py`                            | Streamlit UI (4 pages). Imports everything from `services`.                                            |

### How PDF extraction works

`services.ocr_pdf_to_text.PDFTextExtractor` runs a two-stage pipeline:

1. **Native text layer** — open the PDF with **PyMuPDF** (`fitz`) and pull
   the embedded text from each page. This is fast, lossless and works
   for the vast majority of digitally-generated CVs (PDFs exported from
   Word, Pages, Google Docs, LaTeX, etc.).
2. **OCR fallback** — for any page whose native text layer is empty or
   suspiciously short (< 40 chars), the page is rendered to a high-DPI
   bitmap and passed to **Tesseract OCR** via `pytesseract`. The result
   is merged back in.

You can also tick **Force OCR** in the sidebar to skip the text layer
entirely, which is useful for fully-scanned CVs that contain phantom
"selectable" text (e.g. PDF/A wrappers around scanned images).

> **About `langextract`** — the [google/langextract](https://github.com/google/langextract)
> library is for **structured information extraction** from text using LLMs
> (Gemini, OpenAI, Ollama, …). It is *not* an OCR library. The right
> open-source toolchain for PDF → text is PyMuPDF + Tesseract, which is
> what we use here. `langextract` would be a great later addition on top
> of the cleaned text — for example, to extract a structured
> `{name, skills, experience}` JSON of each candidate.

### How PII removal works

The `PIIRemovalService` combines:

- **spaCy NER** (`en_core_web_sm`) → detects `PERSON`, `GPE`, `LOC`, `FAC`.
- **Regex patterns** → emails, phone numbers, social URLs, `@` handles, dates of birth.
- **Keyword heuristics** → address lines (street, zip…), photo references.

Every detected span is replaced with `[REDACTED]` and appended to `pii_log.json`
with its type, value, replacement token, offsets, confidence and timestamp.
This satisfies GDPR **Article 5(1)(c)** (data minimization), **Article 25**
(data protection by design) and **Article 30** (records of processing).

### How scoring works

`RelevancyScorer` sends a carefully prompted request to `claude-sonnet-4-6`
on Vertex AI and forces a **strict JSON response**:

```json
{
  "score": 0.87,
  "verdict": "Strong match",
  "reasoning": "…",
  "matched_skills": ["Python", "FastAPI", "AWS"],
  "missing_skills": ["PCI compliance"],
  "experience_fit": "match"
}
```

If the Vertex call fails (no auth, no network, bad project, …) the scorer
falls back to a keyword-overlap heuristic so the UI still works end-to-end
for demos.

---

## Prerequisites

- **Python 3.12 or newer** (`python3 --version` to check).
- A **Google Cloud project** with the Vertex AI API enabled and
  access to the `claude-sonnet-4-6` model in the Vertex AI Model Garden.
- The **gcloud CLI** for local credentials:
  <https://cloud.google.com/sdk/docs/install>
- *(Optional but recommended)* **Tesseract OCR** — only needed if you
  want to OCR scanned PDFs on the *PDF CV Analysis* page. Text-based
  PDFs work without it.

Enable the API and Claude model on your project once:

```bash
gcloud services enable aiplatform.googleapis.com
# Then, in the Cloud Console, visit the Model Garden and enable
# "Claude Sonnet 4.6" for your project.
```

Authenticate your machine so the Anthropic SDK can talk to Vertex:

```bash
gcloud auth application-default login
```

Install the Tesseract binary if you plan to OCR scanned PDFs:

```bash
# macOS
brew install tesseract

# Debian / Ubuntu
sudo apt-get install -y tesseract-ocr

# Windows (PowerShell, with Chocolatey)
choco install tesseract
```

---

## Option A — Run with plain `pip` (no uv)

```bash
# 1. Clone / download the project, then cd into it
cd cv-checker

# 2. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows PowerShell

# 3. Install dependencies
pip install --upgrade pip
pip install "anthropic[vertex]" streamlit spacy pandas \
            pymupdf pytesseract pillow

# 4. Download the spaCy English model (used by the PII remover)
python -m spacy download en_core_web_sm

# 5. Launch the app
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it
in your browser.

## Option B — Run with `uv` (faster, reproducible)

If you have [uv](https://docs.astral.sh/uv/) installed:

```bash
cd cv-checker
uv sync                                  # install exact versions from uv.lock
uv run python -m spacy download en_core_web_sm
uv run streamlit run app.py
```

`uv.lock` is committed, so every teammate gets the same dependency graph.

---

## Configuring Vertex AI from the UI

Once the app is running, open the **sidebar** and fill in the
*Vertex AI config* section:

- **GCloud project ID** — e.g. `my-company-ai`
- **Region** — any region that serves `claude-sonnet-4-6`, e.g.
  `europe-west1`, `us-east5`

Click **Apply**. The scorer is rebuilt and the score cache is cleared, so
the next *Run scoring* call uses your credentials.

You can also pre-fill these via environment variables before launching:

```bash
export VERTEX_PROJECT_ID=my-company-ai
export VERTEX_REGION=europe-west1
```

If `gcloud auth application-default login` has been run, the Anthropic SDK
picks up the credentials automatically — no API key required.

---

## Using the app

### 1. CV Score Rank

1. Put any plain-text job descriptions under `job_description/` and CVs
   under `cv/` (one file per document).
2. Open the *CV Score Rank* page.
3. Click **Run scoring** in the sidebar.
4. For every job description the app renders a 3-column layout — the JD
   on the left, the **anonymized** CV in the middle, and the 0.0–1.0
   Claude score on the right, sorted best-first. Each score has a
   *Why?* expander with matched / missing skills.

### 2. GDPR Audit Log

Every anonymization run appends to `pii_log.json`. The page shows:

- Headline metrics (candidates logged, total PII entities, distinct
  categories).
- One expandable panel per candidate with a table of every redaction
  (Type, Value, Replaced With, Timestamp).
- A collapsible raw-JSON view for deeper inspection.

Use *Regenerate from sample CVs* or *Clear audit log* in the sidebar for
demo resets.

### 3. Job ⇄ CV Score (free text)

Paste any job description (left) and CV (right), then:

- **🧹 Remove PII from CV** — runs the anonymizer and shows the cleaned
  text plus a table of everything that was redacted.
- **🎯 Calculate Score** — scores the (anonymized) CV against the JD
  using Claude and returns score + verdict + reasoning.
- **Reset** — clears both boxes.

### 4. PDF CV Analysis (upload + OCR)

The newest page glues the whole pipeline together for real PDFs:

1. **Upload a PDF CV** on the left, **paste a job description** on the right.
2. Tweak the OCR controls in the sidebar if needed:
   - **Force OCR on every page** — for fully-scanned CVs that have a
     bogus/empty text layer.
   - **OCR render DPI** — 300 is a good default; bump to 400–600 for
     hard-to-read scans.
   - **OCR language(s)** — Tesseract language codes, e.g. `eng`,
     `eng+fra`, `eng+tha` (the matching `*.traineddata` must be installed).
3. Click **📄 Extract & Anonymize** — the app:
   - extracts the text via PyMuPDF (OCR fallback per page),
   - removes PII with `PIIRemovalService`,
   - appends an entry to `pii_log.json`,
   - shows extraction stats, the anonymized text, and the redaction table.
4. Click **🎯 Calculate Score** — sends the anonymized text + your JD to
   Claude on Vertex and renders score, verdict, matched/missing skills.

---

## Troubleshooting

| Symptom                                                                    | Likely cause / fix                                                                                             |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: en_core_web_sm`                                      | Run `python -m spacy download en_core_web_sm`.                                                                 |
| Scores all read *"Heuristic (offline)"*                                    | The Vertex call failed — check project ID, region, and `gcloud auth application-default login`.                |
| `PermissionDenied: 403` from Vertex                                        | Enable **Vertex AI API** and *Claude Sonnet 4.6* in the Model Garden for your project.                         |
| `NotFound: model claude-sonnet-4-6` / `404`                                | Your chosen region does not serve this model — try `europe-west1`, `us-east5`, or `us-east1`.                  |
| Streamlit can't find `app.py`                                              | Make sure you ran `streamlit run app.py` from the project root.                                                |
| Audit log page is empty                                                    | Visit *CV Score Rank* once, or click *Regenerate from sample CVs* in the sidebar of the audit page.            |
| PDF page returns "no extractable text" / OCR skipped                       | Install Tesseract (`brew install tesseract` / `apt-get install tesseract-ocr`) **and** tick *Force OCR* if the PDF is scanned. |
| `ImportError: PyMuPDF is required for PDF text extraction`                 | Run `pip install pymupdf` (or `uv sync` to pick it up from `pyproject.toml`).                                  |

---

## Notes for production use

- Swap `en_core_web_sm` for `en_core_web_trf` (transformer-based) for
  higher-accuracy NER on real CVs.
- Replace the regex/keyword PII layer with **Microsoft Presidio** or a
  custom model trained on CV data.
- Store `pii_log.json` in an encrypted, append-only store (e.g. BigQuery,
  S3 with Object Lock) rather than on local disk.
- Add authentication / tenant isolation in Streamlit before exposing it.
- Pin `temperature=0` (already the default) and validate the JSON schema
  strictly on every response.
