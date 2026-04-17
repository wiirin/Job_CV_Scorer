# CV Checker

An AI-powered resume screening tool that ranks candidate CVs against job
descriptions using **Claude Sonnet 4.6 on Google Vertex AI**, with a
built-in **GDPR-compliant PII removal layer** that strips personal data
from every CV *before* it is sent to the model.

The project ships as a 3-page Streamlit app:

| Page                | What it does                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| **CV Score Rank**   | Loads every JD in `job_description/` and every CV in `cv/`, anonymizes the CVs and ranks them per JD.       |
| **GDPR Audit Log**  | Displays `pii_log.json` — a full audit trail of every PII entity that was detected and redacted.            |
| **Job ⇄ CV Score**  | Free-text playground: paste any JD + CV, click *Remove PII*, then *Calculate Score*.                        |

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Raw CV /    │ ──▶ │ PIIRemovalService │ ──▶ │  RelevancyScorer    │ ──▶ score (0.0–1.0)
│  Job Desc.   │     │ (spaCy NER +      │     │  (Claude Sonnet 4.6 │
└──────────────┘     │  regex patterns)  │     │   on Vertex AI)     │
                     └────────┬──────────┘     └─────────────────────┘
                              │
                              ▼
                      pii_log.json
                    (GDPR audit trail)
```

### Key files

| File                    | Purpose                                                                  |
| ----------------------- | ------------------------------------------------------------------------ |
| `pii_remover.py`        | `PIIRemovalService` — spaCy + regex PII detection, writes `pii_log.json`.|
| `relevancy_scorer.py`   | `RelevancyScorer` — prompt-engineered JSON-only Claude call on Vertex.   |
| `app.py`                | Streamlit UI (3 pages).                                                  |
| `cv/`                   | Sample CVs (plain text).                                                 |
| `job_description/`      | Sample job descriptions (plain text).                                    |
| `pii_log.json`          | Persistent audit log, (re)generated on every ranking run.                |

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
pip install "anthropic[vertex]" streamlit spacy pandas

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
