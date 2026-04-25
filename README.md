# CV Checker

An AI-powered resume screening tool that scores candidate CVs against job
descriptions using **Claude Sonnet 4.6**, with a built-in
**GDPR-compliant PII removal layer** that strips personal data from every CV
*before* it is sent to the model. PDF CVs are supported end-to-end via
**PyMuPDF text extraction with a Tesseract OCR fallback** for scanned
documents, and candidates can optionally share a **cover letter** (PDF or
pasted text) to receive direct, professionally worded writing feedback in
addition to the CV ⇄ JD score.

Two Claude providers are supported out of the box and can be switched from
the sidebar:

- **Anthropic Claude API** (just an API key — easiest to get started)
- **Google Vertex AI** (Claude Sonnet 4.6 served from your GCP project)

The project ships as a 4-page Streamlit app:

| Page                                | What it does                                                                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PDF CV Analysis**                 | Upload a PDF CV → extract text (OCR fallback) → remove PII → score against your pasted JD. Optionally add a cover letter (PDF or text) for feedback.  |
| **Job ⇄ CV text Score**             | Free-text playground: paste any JD + CV, click *Remove PII*, then *Calculate Score*.                                                                  |
| **Rank from sample CVs–Job Desc.**  | Loads every JD in `job_description/` and every CV in `cv/`, anonymizes the CVs and ranks them per JD.                                                 |
| **GDPR Audit Log**                  | Displays `pii_log.json` — a full audit trail of every PII entity that was detected and redacted.                                                      |

---

## Architecture

```
┌──────────────┐    ┌────────────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  PDF / TXT   │ ─▶ │ PDFTextExtractor   │ ─▶ │ PIIRemovalService │ ─▶ │  RelevancyScorer     │ ─▶ score (0.0–1.0)
│  CV input    │    │ (PyMuPDF + OCR     │    │ (spaCy NER +      │    │  (Claude Sonnet 4.6) │
└──────────────┘    │  via Tesseract)    │    │  regex patterns)  │    └──────────────────────┘
                    └────────┬───────────┘    └─────────┬─────────┘
                             │                          │
                             │                          ▼
                             │                    pii_log.json
                             │                  (GDPR audit trail)
                             │
                             ▼ (optional)
                    ┌────────────────────────┐
                    │  CoverLetterAnalyzer   │ ─▶ professional writing feedback
                    │  (Claude Sonnet 4.6)   │    (verdict, priority revisions, …)
                    └────────────────────────┘
```

### Repository layout

```
cv-checker/
├── app.py                       # Streamlit UI — 4 pages, no business logic
├── services/
│   ├── __init__.py              # Re-exports the public API
│   ├── pii_remover.py           # PIIRemovalService (spaCy NER + regex)
│   ├── relevancy_scorer.py      # RelevancyScorer (Claude Sonnet 4.6)
│   ├── ocr_pdf_to_text.py       # PDFTextExtractor (PyMuPDF + Tesseract OCR)
│   └── cv_cover_letter.py       # CoverLetterAnalyzer (Claude Sonnet 4.6)
├── cv/                          # Sample CVs (plain text)
├── job_description/             # Sample job descriptions (plain text)
├── pii_log.json                 # Persistent GDPR audit log
├── pyproject.toml               # uv / pip deps
└── README.md
```

### Key modules

| Module                              | Purpose                                                                                                              |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `services.pii_remover`              | `PIIRemovalService` — spaCy + regex PII detection, writes `pii_log.json`.                                            |
| `services.relevancy_scorer`         | `RelevancyScorer` — prompt-engineered JSON-only Claude call (Vertex AI or Claude API).                               |
| `services.ocr_pdf_to_text`          | `PDFTextExtractor` / `extract_text_from_pdf()` — text-layer first, OCR fallback for scanned pages.                   |
| `services.cv_cover_letter`          | `CoverLetterAnalyzer` — Claude-powered cover letter feedback (CV + JD aware), professional and constructive in tone. |
| `app.py`                            | Streamlit UI (4 pages). Imports everything from `services`.                                                          |

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

> **Note on cover letters:** the cover letter is *not* anonymized before
> being sent to the model. It is the candidate's own document and they are
> asking for feedback on the actual writing — redacting the salutation, for
> example, would prevent the model from commenting on whether the opening
> works. CVs continue to be anonymized as before.

### How CV scoring works

`RelevancyScorer` sends a carefully prompted request to `claude-sonnet-4-6`
(via Vertex AI or the Anthropic Claude API) and forces a **strict JSON
response**:

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

If the Claude call fails (no auth, no network, bad project, missing API
key, …) the scorer falls back to a keyword-overlap heuristic so the UI
still works end-to-end for demos.

### How cover letter analysis works

`CoverLetterAnalyzer` lives in `services.cv_cover_letter` and shares the
same provider configuration as the scorer (Vertex AI or Claude API — no
extra credentials needed). When the candidate provides a cover letter on
the *PDF CV Analysis* page, the analyzer sends Claude the **JD + CV +
cover letter** together with a system prompt instructing it to act as an
experienced career coach and senior hiring manager. The voice is
**direct, professional, and constructive** — specific about what to
refine, but measured in word choice (no harsh or dismissive language).

The response is a strict JSON payload:

```json
{
  "score": 0.62,
  "verdict": "Solid",
  "send_recommendation": "Minor revisions recommended",
  "overall_assessment": "The letter is clear and professional, but the opening could be more tailored to the role…",
  "strengths": ["Cites a concrete project outcome in paragraph two"],
  "improvements": [
    "Reference the team or product by name in the opening.",
    "Replace generic phrasing such as 'passionate about technology' with a concrete example."
  ],
  "priority_revisions": [
    "Tailor the opening paragraph to the specific role and team."
  ],
  "jd_alignment": "Addresses the data-pipeline requirement well; does not mention the stakeholder-management aspect.",
  "cv_alignment": "Adds useful narrative around the most recent role rather than restating bullets.",
  "tone": "Professional but slightly generic",
  "bottom_line": "Recommend revising the opening and tightening the alignment to the role before sending."
}
```

The UI surfaces a single *Recommendation* banner (green when the letter is
genuinely send-ready, neutral blue otherwise), a **Priority revisions**
list of the most impactful changes to address first, plus separate
**Suggested refinements** and **Strengths** columns. Cover letter feedback
only appears when the user actually provides a letter — the existing
CV-only flow is unchanged.

---

## Prerequisites

- **Python 3.12 or newer** (`python3 --version` to check).
- **Claude access**, via *either*:
  - an **Anthropic API key** (`sk-ant-…`) — get one at
    [console.anthropic.com](https://console.anthropic.com), *or*
  - a **Google Cloud project** with the Vertex AI API enabled and access
    to the `claude-sonnet-4-6` model in the Vertex AI Model Garden,
    plus the **gcloud CLI** for local credentials
    (<https://cloud.google.com/sdk/docs/install>).
- *(Optional but recommended)* **Tesseract OCR** — only needed if you
  want to OCR scanned PDFs on the *PDF CV Analysis* page. Text-based
  PDFs (CV or cover letter) work without it.

### If you use Vertex AI

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

### If you OCR scanned PDFs

Install the Tesseract binary:

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

## Configuring the Claude provider from the UI

Once the app is running, open the **sidebar** under *Claude provider* and
pick one of the two backends:

### Anthropic Claude API (easiest)

- **Anthropic API key** — paste your `sk-ant-…` key.

### Google Vertex AI

- **GCloud project ID** — e.g. `my-company-ai`
- **Region** — any region that serves `claude-sonnet-4-6`, e.g.
  `europe-west1`, `us-east5`

Click **Apply**. The scorer **and** the cover letter analyzer are both
rebuilt and the score cache is cleared, so the next call uses your new
credentials.

You can also pre-fill these via environment variables before launching:

```bash
# Claude API
export ANTHROPIC_API_KEY=sk-ant-...

# OR Vertex AI
export VERTEX_PROJECT_ID=my-company-ai
export VERTEX_REGION=europe-west1
```

If `gcloud auth application-default login` has been run, the Anthropic
Vertex SDK picks up the credentials automatically — no API key required
for the Vertex backend.

---

## Using the app

### 1. PDF CV Analysis (upload + OCR + optional cover letter)

The flagship page glues the whole pipeline together for real PDFs:

1. **Upload a PDF CV** on the left, **paste a job description** on the right.
2. Tweak the OCR controls in the sidebar if needed:
   - **Force OCR on every page** — for fully-scanned CVs (or cover
     letters) that have a bogus/empty text layer.
   - **OCR render DPI** — 300 is a good default; bump to 400–600 for
     hard-to-read scans.
   - **OCR language(s)** — Tesseract language codes, e.g. `eng`,
     `eng+fra`, `eng+tha` (the matching `*.traineddata` must be installed).
3. *(Optional)* In the **Cover letter (optional)** section, choose:
   - **Skip** — no cover letter feedback (default).
   - **Paste text** — drop the cover letter into the text area.
   - **Upload PDF** — same OCR pipeline is reused to extract the text.
4. Click **📄 Extract & Anonymize** — the app:
   - extracts the CV text via PyMuPDF (OCR fallback per page),
   - removes PII with `PIIRemovalService`,
   - appends an entry to `pii_log.json`,
   - extracts the cover letter PDF too if one was uploaded,
   - shows extraction stats, the anonymized text, and the redaction table.
5. Click **🎯 Calculate Score** — sends the anonymized CV + your JD to
   Claude and renders the relevancy score, verdict, matched/missing skills.
   If a cover letter is present, the **Cover Letter Feedback** panel also
   appears with:
   - a quality score and verdict (`Excellent` / `Strong` / `Solid` /
     `Developing` / `Early draft`),
   - a one-line **Recommendation** banner (`Ready to send` /
     `Minor revisions recommended` / `Substantial revisions recommended`),
   - **Priority revisions** — the most impactful changes to address first,
   - **Suggested refinements** and **Strengths** columns,
   - alignment notes against both the JD and the CV.

If you upload a cover letter PDF *after* clicking *Extract & Anonymize*,
just click *Calculate Score* — the cover letter is extracted lazily so you
don't have to redo the CV step.

### 2. Job ⇄ CV text Score (free text)

Paste any job description (left) and CV (right), then:

- **🧹 Remove PII from CV** — runs the anonymizer and shows the cleaned
  text plus a table of everything that was redacted.
- **🎯 Calculate Score** — scores the (anonymized) CV against the JD
  using Claude and returns score + verdict + reasoning.
- **Reset** — clears both boxes.

### 3. Rank from sample CVs–Job Description

1. Put any plain-text job descriptions under `job_description/` and CVs
   under `cv/` (one file per document).
2. Open the page and click **Run scoring** in the sidebar.
3. For every job description the app renders a 3-column layout — the JD
   on the left, the **anonymized** CV in the middle, and the 0.0–1.0
   Claude score on the right, sorted best-first. Each score has a
   *Why?* expander with matched / missing skills.

### 4. GDPR Audit Log

Every anonymization run appends to `pii_log.json`. The page shows:

- Headline metrics (candidates logged, total PII entities, distinct
  categories).
- One expandable panel per candidate with a table of every redaction
  (Type, Value, Replaced With, Timestamp).
- A collapsible raw-JSON view for deeper inspection.

Use *Regenerate from sample CVs* or *Clear audit log* in the sidebar for
demo resets.

---

## Troubleshooting

| Symptom                                                                    | Likely cause / fix                                                                                                              |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: en_core_web_sm`                                      | Run `python -m spacy download en_core_web_sm`.                                                                                  |
| Scores all read *"Heuristic (offline)"*                                    | The Claude call failed — check API key (Claude API) **or** project ID + region + `gcloud auth application-default login` (Vertex). |
| Cover letter feedback says *"Could not generate cover letter feedback right now"* | Same as above — the analyzer reuses the scorer's provider config. Re-check credentials in the sidebar.                          |
| `PermissionDenied: 403` from Vertex                                        | Enable **Vertex AI API** and *Claude Sonnet 4.6* in the Model Garden for your project.                                          |
| `NotFound: model claude-sonnet-4-6` / `404`                                | Your chosen region does not serve this model — try `europe-west1`, `us-east5`, or `us-east1`.                                   |
| `RuntimeError: Claude API provider selected but no API key was provided`   | Paste your `sk-ant-…` key in the sidebar (or export `ANTHROPIC_API_KEY`).                                                       |
| Streamlit can't find `app.py`                                              | Make sure you ran `streamlit run app.py` from the project root.                                                                 |
| Audit log page is empty                                                    | Visit *Rank from sample CVs–Job Description* once, or click *Regenerate from sample CVs* in the sidebar of the audit page.      |
| PDF page returns "no extractable text" / OCR skipped                       | Install Tesseract (`brew install tesseract` / `apt-get install tesseract-ocr`) **and** tick *Force OCR* if the PDF is scanned.  |
| Cover letter PDF produced no text                                          | Same as above — tick *Force OCR* in the sidebar, or switch to *Paste text* for the cover letter.                                |
| `ImportError: PyMuPDF is required for PDF text extraction`                 | Run `pip install pymupdf` (or `uv sync` to pick it up from `pyproject.toml`).                                                   |

---

## Notes for production use

- Swap `en_core_web_sm` for `en_core_web_trf` (transformer-based) for
  higher-accuracy NER on real CVs.
- Replace the regex/keyword PII layer with **Microsoft Presidio** or a
  custom model trained on CV data.
- Store `pii_log.json` in an encrypted, append-only store (e.g. BigQuery,
  S3 with Object Lock) rather than on local disk.
- Add authentication / tenant isolation in Streamlit before exposing it.
- Pin `temperature=0` (already the default for both the scorer and the
  cover letter analyzer) and validate the JSON schema strictly on every
  response.
- Cover letters are not anonymized before being sent to Claude (by design —
  see the note in *How PII removal works*). If you need to retain cover
  letters server-side, store them under the same encryption / retention
  controls as raw CVs.
