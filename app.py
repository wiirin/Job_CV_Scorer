"""
CV Checker — Streamlit app.

Four pages (in display order):
  1. PDF CV Analysis                       — upload a PDF CV, OCR → PII removal → score vs JD.
  2. Job ⇄ CV text Score                   — free-text playground (remove PII + score).
  3. Rank from sample CVs-Job Description  — rank candidate CVs for each selected job description.
  4. GDPR Audit Log                        — browse the pii_log.json produced during anonymization.

All heavy lifting lives in the ``services/`` package; this file is just the UI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from services import (
    PROVIDER_CLAUDE_API,
    PROVIDER_VERTEX,
    AnonymizationResult,
    CoverLetterAnalyzer,
    CoverLetterFeedback,
    PDFExtractionResult,
    PDFTextExtractor,
    PIIRemovalService,
    RelevancyScore,
    RelevancyScorer,
)

# ──────────────────────────────────────────────
# Constants / paths
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CV_DIR = ROOT / "cv"
JOB_DIR = ROOT / "job_description"
AUDIT_LOG = ROOT / "pii_log.json"

PAGE_RANK = "Rank from sample CVs-Job Description"
PAGE_AUDIT = "GDPR Audit Log"
PAGE_PLAYGROUND = "Job ⇄ CV text Score"
PAGE_PDF = "PDF CV Analysis"


# ──────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pii_service() -> PIIRemovalService:
    return PIIRemovalService(audit_log_path=AUDIT_LOG)


@st.cache_resource(show_spinner=False)
def get_pdf_extractor(force_ocr: bool, ocr_dpi: int, ocr_language: str) -> PDFTextExtractor:
    """Build (or reuse) a PDFTextExtractor for the requested config."""
    return PDFTextExtractor(
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        force_ocr=force_ocr,
    )


@st.cache_resource(show_spinner=False)
def _build_scorer(
    provider: str,
    project_id: str,
    region: str,
    api_key: str,
) -> RelevancyScorer:
    """Build a scorer keyed by all config inputs so edits invalidate the cache."""
    return RelevancyScorer(
        provider=provider or PROVIDER_VERTEX,
        project_id=project_id or None,
        region=region or None,
        api_key=api_key or None,
    )


def get_scorer() -> RelevancyScorer:
    cfg = st.session_state.get("scorer_cfg", {})
    return _build_scorer(
        cfg.get("provider", PROVIDER_VERTEX),
        cfg.get("project_id", ""),
        cfg.get("region", ""),
        cfg.get("api_key", ""),
    )


@st.cache_resource(show_spinner=False)
def _build_cover_letter_analyzer(
    provider: str,
    project_id: str,
    region: str,
    api_key: str,
) -> CoverLetterAnalyzer:
    """Cover letter analyzer that piggy-backs on the scorer's provider config."""
    return CoverLetterAnalyzer(
        provider=provider or PROVIDER_VERTEX,
        project_id=project_id or None,
        region=region or None,
        api_key=api_key or None,
    )


def get_cover_letter_analyzer() -> CoverLetterAnalyzer:
    cfg = st.session_state.get("scorer_cfg", {})
    return _build_cover_letter_analyzer(
        cfg.get("provider", PROVIDER_VERTEX),
        cfg.get("project_id", ""),
        cfg.get("region", ""),
        cfg.get("api_key", ""),
    )


@st.cache_data(show_spinner=False)
def load_job_descriptions() -> Dict[str, str]:
    """Return {job_name: text} for every file in /job_description."""
    jobs: Dict[str, str] = {}
    if not JOB_DIR.exists():
        return jobs
    for path in sorted(JOB_DIR.iterdir()):
        if path.is_file() and path.suffix not in {".py", ".pyc"}:
            try:
                jobs[path.name] = path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                continue
    return jobs


@st.cache_data(show_spinner=False)
def load_cvs() -> Dict[str, str]:
    """Return {cv_name: raw_text} for every file in /cv."""
    cvs: Dict[str, str] = {}
    if not CV_DIR.exists():
        return cvs
    for path in sorted(CV_DIR.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            try:
                cvs[path.name] = path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                continue
    return cvs


@st.cache_data(show_spinner=False)
def anonymize_all_cvs(_service: PIIRemovalService, cvs: Dict[str, str]) -> Dict[str, AnonymizationResult]:
    """Anonymize every CV once and persist a fresh audit log for the demo."""
    _service.clear_audit_log()
    results: Dict[str, AnonymizationResult] = {}
    for idx, (name, text) in enumerate(cvs.items(), start=1):
        candidate_id = f"cv-{idx:03d}"
        results[name] = _service.anonymize(text, candidate_id=candidate_id, log_to_file=True)
    return results


def load_audit_log() -> List[Dict]:
    if not AUDIT_LOG.exists() or AUDIT_LOG.stat().st_size == 0:
        return []
    try:
        with AUDIT_LOG.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# ──────────────────────────────────────────────
# Scoring helpers (session-cached)
# ──────────────────────────────────────────────
def score_cv_job(cv_text: str, jd_text: str) -> RelevancyScore:
    """Score a CV/JD pair, caching results in st.session_state."""
    cache = st.session_state.setdefault("score_cache", {})
    key = (hash(cv_text), hash(jd_text))
    if key in cache:
        return cache[key]

    scorer = get_scorer()
    result = scorer.score(cv_text=cv_text, job_description=jd_text)
    cache[key] = result
    return result


# ──────────────────────────────────────────────
# Page 1 — CV Score Rank
# ──────────────────────────────────────────────
def page_rank() -> None:
    st.title("Rank from sample CVs-Job Description")
    st.caption(
        "Select one or more job descriptions. The app removes PII from every "
        "sample CV and then asks Claude Sonnet 4.6 for a 0.0 – 1.0 fit score "
        "(via your configured provider — Claude API or Vertex AI)."
    )

    jobs = load_job_descriptions()
    cvs = load_cvs()

    if not jobs or not cvs:
        st.warning("No job descriptions or CVs found under `job_description/` or `cv/`.")
        return

    anonymized = anonymize_all_cvs(get_pii_service(), cvs)

    with st.sidebar:
        st.subheader("Ranking controls")
        st.caption(f"Scoring **{len(cvs)}** CVs against **{len(jobs)}** jobs.")
        run = st.button("Run scoring", type="primary", use_container_width=True)
        if st.button("Clear score cache", use_container_width=True):
            st.session_state.pop("score_cache", None)
            st.success("Cache cleared.")

    if not run and "score_cache" not in st.session_state:
        st.info("Press **Run scoring** in the sidebar to compute scores.")
        return

    for job_name in jobs.keys():
        jd_text = jobs[job_name]
        st.markdown("---")
        st.subheader(f"📄 {job_name}")

        # Score + sort all CVs for this JD
        rows: List[Tuple[str, AnonymizationResult, RelevancyScore]] = []
        with st.spinner(f"Scoring {len(anonymized)} CVs against {job_name}…"):
            for cv_name, anon in anonymized.items():
                result = score_cv_job(anon.anonymized_text, jd_text)
                rows.append((cv_name, anon, result))
        rows.sort(key=lambda r: r[2].score, reverse=True)

        # 3-column layout mirroring the slide template
        header = st.columns([3, 5, 2])
        header[0].markdown("**Job Description**")
        header[1].markdown("**CV – Without PII**")
        header[2].markdown("**Score**")

        for idx, (cv_name, anon, result) in enumerate(rows):
            cols = st.columns([3, 5, 2])
            if idx == 0:
                with cols[0]:
                    with st.container(border=True):
                        st.markdown(f"**{job_name}**")
                        st.text_area(
                            label="jd",
                            value=jd_text,
                            height=260,
                            key=f"jd-{job_name}",
                            label_visibility="collapsed",
                        )
            else:
                cols[0].markdown("&nbsp;", unsafe_allow_html=True)

            with cols[1]:
                with st.container(border=True):
                    st.markdown(f"**{cv_name}**  ·  _{anon.entity_count} PII removed_")
                    st.text_area(
                        label="cv",
                        value=anon.anonymized_text,
                        height=220,
                        key=f"cv-{job_name}-{cv_name}",
                        label_visibility="collapsed",
                    )
            with cols[2]:
                with st.container(border=True):
                    st.metric(label="Fit", value=f"{result.score:.2f}")
                    if result.verdict:
                        st.caption(result.verdict)
                    st.progress(result.score)
                    if result.reasoning:
                        with st.expander("Why?"):
                            st.write(result.reasoning)
                            if result.matched_skills:
                                st.markdown("**Matched**")
                                st.write(", ".join(result.matched_skills))
                            if result.missing_skills:
                                st.markdown("**Missing**")
                                st.write(", ".join(result.missing_skills))


# ──────────────────────────────────────────────
# Page 2 — GDPR Audit Log
# ──────────────────────────────────────────────
def page_audit() -> None:
    st.title("🔒 GDPR Compliance – PII Audit Log")
    st.write(
        "This log tracks all Personally Identifiable Information (PII) that was "
        "detected and removed from CVs before AI scoring. This ensures GDPR "
        "compliance by:"
    )
    st.markdown(
        "- **Article 5(1)(c)**: Data minimization — only necessary data is processed\n"
        "- **Article 25**: Data protection by design — PII removed before scoring\n"
        "- **Article 30**: Records of processing activities — full audit trail"
    )

    with st.sidebar:
        st.subheader("Audit controls")
        if st.button("Regenerate from sample CVs", use_container_width=True):
            anonymize_all_cvs.clear()
            anonymize_all_cvs(get_pii_service(), load_cvs())
            st.success("Regenerated audit log.")
        if st.button("Clear audit log", use_container_width=True):
            get_pii_service().clear_audit_log()
            st.success("Audit log cleared.")

    entries = load_audit_log()
    if not entries:
        st.info(
            "No audit entries yet. Visit **Rank from sample CVs-Job Description** "
            "(or click *Regenerate from sample CVs* in the sidebar) to populate the log."
        )
        return

    total_pii = sum(e.get("entity_count", 0) for e in entries)
    all_cats = sorted({c for e in entries for c in e.get("categories", [])})

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates logged", len(entries))
    c2.metric("PII entities removed", total_pii)
    c3.metric("Distinct PII categories", len(all_cats))

    st.markdown("---")

    for entry in entries:
        header = (
            f"Candidate `{entry.get('candidate_id', 'unknown')}` — "
            f"{entry.get('entity_count', 0)} PII entities removed"
        )
        with st.expander(header, expanded=False):
            meta_cols = st.columns(3)
            meta_cols[0].caption(f"Timestamp: {entry.get('timestamp', '')}")
            meta_cols[1].caption(
                f"Categories: {', '.join(entry.get('categories', [])) or '—'}"
            )
            meta_cols[2].caption(
                f"Chars: {entry.get('original_length', 0)} → {entry.get('anonymized_length', 0)}"
            )

            ent_rows = [
                {
                    "Type": ent.get("type", ""),
                    "Value (truncated)": _truncate(ent.get("value", ""), 60),
                    "Replaced With": ent.get("replaced_with", "[REDACTED]"),
                    "Timestamp": ent.get("timestamp", ""),
                }
                for ent in entry.get("entities", [])
            ]
            if ent_rows:
                st.dataframe(pd.DataFrame(ent_rows), use_container_width=True, hide_index=False)
            else:
                st.caption("No PII entities recorded for this candidate.")

    with st.expander("Raw pii_log.json", expanded=False):
        st.code(json.dumps(entries, indent=2, ensure_ascii=False), language="json")


def _truncate(text: str, limit: int) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ──────────────────────────────────────────────
# Page 3 — Job ⇄ CV Score (free text playground)
# ──────────────────────────────────────────────
def page_playground() -> None:
    st.title("Job ⇄ CV text Score")
    st.caption(
        "Paste any job description and CV text, remove PII, then compute a fit score."
    )

    ss = st.session_state
    ss.setdefault("pg_jd", "")
    ss.setdefault("pg_cv", "")
    ss.setdefault("pg_cv_clean", "")
    ss.setdefault("pg_anon_result", None)
    ss.setdefault("pg_score", None)

    left, right = st.columns(2)
    with left:
        st.subheader("Job Description")
        ss.pg_jd = st.text_area(
            "Paste job description",
            value=ss.pg_jd,
            height=320,
            label_visibility="collapsed",
        )
    with right:
        st.subheader("CV")
        ss.pg_cv = st.text_area(
            "Paste candidate CV",
            value=ss.pg_cv,
            height=320,
            label_visibility="collapsed",
        )

    btn1, btn2, btn3 = st.columns([1, 1, 1])
    remove_clicked = btn1.button(
        "🧹 Remove PII from CV",
        use_container_width=True,
        disabled=not ss.pg_cv.strip(),
    )
    score_clicked = btn2.button(
        "🎯 Calculate Score",
        type="primary",
        use_container_width=True,
        disabled=not (ss.pg_jd.strip() and (ss.pg_cv_clean or ss.pg_cv).strip()),
    )
    if btn3.button("Reset", use_container_width=True):
        for k in ("pg_jd", "pg_cv", "pg_cv_clean", "pg_anon_result", "pg_score"):
            ss[k] = "" if k.startswith("pg_") and isinstance(ss[k], str) else None
        st.rerun()

    if remove_clicked:
        with st.spinner("Detecting and redacting PII…"):
            anon = get_pii_service().anonymize(
                ss.pg_cv, candidate_id="playground", log_to_file=True
            )
        ss.pg_anon_result = anon
        ss.pg_cv_clean = anon.anonymized_text
        ss.pg_score = None  # invalidate previous score
        st.success(anon.summary)

    if score_clicked:
        cv_for_scoring = ss.pg_cv_clean or ss.pg_cv
        if not ss.pg_cv_clean:
            st.info("Tip: click **Remove PII from CV** first for a GDPR-compliant flow.")
        with st.spinner("Scoring CV against job description…"):
            ss.pg_score = get_scorer().score(
                cv_text=cv_for_scoring, job_description=ss.pg_jd
            )

    # ── Output panels ───────────────────────────
    if ss.pg_anon_result is not None:
        st.markdown("### Anonymized CV")
        a, b = st.columns([3, 2])
        with a:
            st.text_area(
                "Anonymized CV",
                value=ss.pg_cv_clean,
                height=260,
                label_visibility="collapsed",
            )
        with b:
            anon: AnonymizationResult = ss.pg_anon_result
            st.metric("PII entities removed", anon.entity_count)
            st.caption(
                "Categories: " + (", ".join(anon.pii_categories_found) or "—")
            )
            if anon.entities_removed:
                df = pd.DataFrame(
                    [
                        {
                            "Type": e.label,
                            "Value": _truncate(e.text, 40),
                            "Replaced": e.replaced_with,
                        }
                        for e in anon.entities_removed
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

    if ss.pg_score is not None:
        st.markdown("### Relevancy Score")
        score: RelevancyScore = ss.pg_score
        m1, m2 = st.columns([1, 3])
        with m1:
            st.metric("Fit score", f"{score.score:.2f}")
            st.progress(score.score)
            if score.verdict:
                st.caption(score.verdict)
        with m2:
            if score.reasoning:
                st.markdown("**Reasoning**")
                st.write(score.reasoning)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Matched skills**")
                st.write(", ".join(score.matched_skills) or "—")
            with cols[1]:
                st.markdown("**Missing skills**")
                st.write(", ".join(score.missing_skills) or "—")
            if score.experience_fit:
                st.caption(f"Experience fit: {score.experience_fit}")
        with st.expander("Raw model response"):
            st.code(score.raw_response or json.dumps(score.to_dict(), indent=2))


# ──────────────────────────────────────────────
# Page 4 — PDF CV Analysis (upload → OCR → PII → score)
# ──────────────────────────────────────────────
def page_pdf() -> None:
    st.title("PDF CV Analysis")
    st.caption(
        "Upload a candidate's CV (PDF). The app extracts the text "
        "(OCR fallback for scanned pages), strips PII for GDPR compliance, "
        "and scores it against your pasted job description. Optionally "
        "share a cover letter (PDF or pasted text) to also receive "
        "constructive writing feedback."
    )

    ss = st.session_state
    ss.setdefault("pdf_jd", "")
    ss.setdefault("pdf_extraction", None)
    ss.setdefault("pdf_anon", None)
    ss.setdefault("pdf_score", None)
    ss.setdefault("pdf_filename", "")
    ss.setdefault("pdf_cover_mode", "Skip")
    ss.setdefault("pdf_cover_text", "")
    ss.setdefault("pdf_cover_pdf_text", "")
    ss.setdefault("pdf_cover_filename", "")
    ss.setdefault("pdf_cover_feedback", None)

    with st.sidebar:
        st.subheader("PDF / OCR settings")
        force_ocr = st.checkbox(
            "Force OCR on every page",
            value=False,
            help="Tick this for fully-scanned PDFs that have no text layer.",
        )
        ocr_dpi = st.slider(
            "OCR render DPI", min_value=150, max_value=600, value=300, step=50,
            help="Higher = better OCR quality, slower extraction.",
        )
        ocr_language = st.text_input(
            "OCR language(s)",
            value=PDFTextExtractor.DEFAULT_LANGUAGE,
            help="Tesseract language codes, e.g. 'eng' or 'eng+fra'.",
        )

    upload_col, jd_col = st.columns([1, 1])
    with upload_col:
        st.subheader("1. Upload PDF CV")
        uploaded = st.file_uploader(
            "Drag-and-drop or browse",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
    with jd_col:
        st.subheader("2. Paste Job Description")
        ss.pdf_jd = st.text_area(
            "Job description",
            value=ss.pdf_jd,
            height=220,
            label_visibility="collapsed",
            placeholder="Paste the job description here…",
        )

    cover_uploaded = _render_cover_letter_inputs(ss)

    btn1, btn2, btn3 = st.columns([1, 1, 1])
    extract_clicked = btn1.button(
        "📄 Extract & Anonymize",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )
    score_clicked = btn2.button(
        "🎯 Calculate Score",
        use_container_width=True,
        disabled=not (ss.pdf_anon is not None and ss.pdf_jd.strip()),
        help=(
            "Scores the CV against the JD. If a cover letter is provided, also "
            "generates writing feedback."
        ),
    )
    if btn3.button("Reset PDF page", use_container_width=True):
        for k in (
            "pdf_extraction",
            "pdf_anon",
            "pdf_score",
            "pdf_filename",
            "pdf_cover_feedback",
            "pdf_cover_pdf_text",
            "pdf_cover_filename",
            "pdf_cover_text",
        ):
            ss[k] = "" if isinstance(ss.get(k), str) else None
        ss.pdf_cover_mode = "Skip"
        st.rerun()

    if extract_clicked and uploaded is not None:
        ss.pdf_filename = uploaded.name
        extractor = get_pdf_extractor(force_ocr, ocr_dpi, ocr_language.strip() or "eng")
        try:
            with st.spinner(f"Extracting text from {uploaded.name}…"):
                pdf_bytes = uploaded.getvalue()
                extraction = extractor.extract(pdf_bytes)
        except ImportError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"PDF extraction failed: {exc}")
            return

        ss.pdf_extraction = extraction
        ss.pdf_score = None  # invalidate any previous score
        ss.pdf_cover_feedback = None  # invalidate stale cover letter feedback

        for w in extraction.warnings:
            st.warning(w)
        if not extraction.text.strip():
            st.error(
                "No text could be extracted from this PDF. If it's a scanned "
                "document, install Tesseract and tick **Force OCR**."
            )
            ss.pdf_anon = None
            return

        with st.spinner("Removing PII…"):
            anon = get_pii_service().anonymize(
                extraction.text,
                candidate_id=f"pdf-{Path(uploaded.name).stem}",
                log_to_file=True,
            )
        ss.pdf_anon = anon
        st.success(
            f"Extracted {extraction.char_count} chars from "
            f"{extraction.page_count} page(s). {anon.summary}"
        )

        # If the user uploaded a cover letter PDF too, extract it now so it's
        # ready when they click "Calculate Score".
        if ss.pdf_cover_mode == "Upload PDF" and cover_uploaded is not None:
            try:
                with st.spinner(f"Extracting cover letter from {cover_uploaded.name}…"):
                    cl_extraction = extractor.extract(cover_uploaded.getvalue())
            except Exception as exc:
                st.warning(f"Cover letter extraction failed: {exc}")
            else:
                ss.pdf_cover_pdf_text = cl_extraction.text
                ss.pdf_cover_filename = cover_uploaded.name
                if not cl_extraction.text.strip():
                    st.warning(
                        "Cover letter PDF produced no text. Try ticking "
                        "**Force OCR** in the sidebar, or paste the letter as text."
                    )
                else:
                    st.info(
                        f"Cover letter ready ({cl_extraction.char_count} chars "
                        f"from {cl_extraction.page_count} page(s))."
                    )

    if score_clicked and ss.pdf_anon is not None:
        with st.spinner("Scoring CV against job description…"):
            ss.pdf_score = get_scorer().score(
                cv_text=ss.pdf_anon.anonymized_text,
                job_description=ss.pdf_jd,
            )

        # Lazy-extract a cover letter PDF the user uploaded after the
        # initial Extract & Anonymize step (so they don't have to click it
        # twice just to add a cover letter).
        if (
            ss.pdf_cover_mode == "Upload PDF"
            and cover_uploaded is not None
            and (
                not ss.pdf_cover_pdf_text
                or ss.pdf_cover_filename != cover_uploaded.name
            )
        ):
            extractor = get_pdf_extractor(force_ocr, ocr_dpi, ocr_language.strip() or "eng")
            try:
                with st.spinner(f"Extracting cover letter from {cover_uploaded.name}…"):
                    cl_extraction = extractor.extract(cover_uploaded.getvalue())
            except Exception as exc:
                st.warning(f"Cover letter extraction failed: {exc}")
            else:
                ss.pdf_cover_pdf_text = cl_extraction.text
                ss.pdf_cover_filename = cover_uploaded.name

        cover_letter_text = _current_cover_letter_text(ss)
        if cover_letter_text.strip():
            with st.spinner("Reading your cover letter and preparing feedback…"):
                ss.pdf_cover_feedback = get_cover_letter_analyzer().analyze(
                    cover_letter=cover_letter_text,
                    cv_text=ss.pdf_anon.anonymized_text,
                    job_description=ss.pdf_jd,
                )
        else:
            ss.pdf_cover_feedback = None

    # ── Output panels ─────────────────────────────────
    extraction: PDFExtractionResult | None = ss.pdf_extraction
    anon: AnonymizationResult | None = ss.pdf_anon
    score: RelevancyScore | None = ss.pdf_score

    if extraction is not None:
        st.markdown("---")
        st.markdown(f"### Extraction — `{ss.pdf_filename}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pages", extraction.page_count)
        m2.metric("Characters", extraction.char_count)
        m3.metric("Method", extraction.method)
        m4.metric("OCR pages", len(extraction.ocr_pages))

        with st.expander("Raw extracted text", expanded=False):
            st.text_area(
                "raw",
                value=extraction.text,
                height=240,
                label_visibility="collapsed",
            )

    if anon is not None:
        st.markdown("### Anonymized CV (sent to the scorer)")
        a, b = st.columns([3, 2])
        with a:
            st.text_area(
                "Anonymized CV",
                value=anon.anonymized_text,
                height=260,
                label_visibility="collapsed",
            )
        with b:
            st.metric("PII entities removed", anon.entity_count)
            st.caption(
                "Categories: " + (", ".join(anon.pii_categories_found) or "—")
            )
            if anon.entities_removed:
                df = pd.DataFrame(
                    [
                        {
                            "Type": e.label,
                            "Value": _truncate(e.text, 40),
                            "Replaced": e.replaced_with,
                        }
                        for e in anon.entities_removed
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

    if score is not None:
        st.markdown("### Relevancy Score")
        m1, m2 = st.columns([1, 3])
        with m1:
            st.metric("Fit score", f"{score.score:.2f}")
            st.progress(score.score)
            if score.verdict:
                st.caption(score.verdict)
        with m2:
            if score.reasoning:
                st.markdown("**Reasoning**")
                st.write(score.reasoning)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Matched skills**")
                st.write(", ".join(score.matched_skills) or "—")
            with cols[1]:
                st.markdown("**Missing skills**")
                st.write(", ".join(score.missing_skills) or "—")
            if score.experience_fit:
                st.caption(f"Experience fit: {score.experience_fit}")
        with st.expander("Raw model response"):
            st.code(score.raw_response or json.dumps(score.to_dict(), indent=2))

    feedback: CoverLetterFeedback | None = ss.pdf_cover_feedback
    if feedback is not None:
        _render_cover_letter_feedback(feedback)


# ──────────────────────────────────────────────
# Cover letter helpers (used by page_pdf)
# ──────────────────────────────────────────────
def _render_cover_letter_inputs(ss):
    """Render the optional cover letter section.

    Returns the uploaded ``UploadedFile`` (or ``None``) so the caller can
    extract its text during the "Extract & Anonymize" step.
    """
    st.subheader("3. Cover letter (optional)")
    st.caption(
        "Share your cover letter to receive constructive writing feedback "
        "in addition to the CV ⇄ JD score. Skip this if you don't have one."
    )

    mode_options = ["Skip", "Paste text", "Upload PDF"]
    current = ss.get("pdf_cover_mode", "Skip")
    if current not in mode_options:
        current = "Skip"
    ss.pdf_cover_mode = st.radio(
        "How would you like to share it?",
        options=mode_options,
        index=mode_options.index(current),
        horizontal=True,
        key="pdf_cover_mode_radio",
    )

    cover_uploaded = None
    if ss.pdf_cover_mode == "Paste text":
        ss.pdf_cover_text = st.text_area(
            "Paste your cover letter",
            value=ss.get("pdf_cover_text", ""),
            height=200,
            placeholder="Dear Hiring Manager,\n\nI'm excited to apply for…",
            key="pdf_cover_text_area",
        )
    elif ss.pdf_cover_mode == "Upload PDF":
        cover_uploaded = st.file_uploader(
            "Upload your cover letter (PDF)",
            type=["pdf"],
            accept_multiple_files=False,
            key="pdf_cover_upload",
            help=(
                "Text will be extracted when you click **Extract & Anonymize**. "
                "It's used only to generate writing feedback for you."
            ),
        )
        if ss.get("pdf_cover_pdf_text"):
            with st.expander(
                f"Extracted cover letter text — {ss.get('pdf_cover_filename', 'cover_letter.pdf')}",
                expanded=False,
            ):
                st.text_area(
                    "cover-letter-extracted",
                    value=ss.pdf_cover_pdf_text,
                    height=180,
                    label_visibility="collapsed",
                )
    else:
        ss.pdf_cover_text = ""
        ss.pdf_cover_pdf_text = ""
        ss.pdf_cover_filename = ""

    return cover_uploaded


def _current_cover_letter_text(ss) -> str:
    """Return whichever cover letter source is currently active (may be empty)."""
    mode = ss.get("pdf_cover_mode", "Skip")
    if mode == "Paste text":
        return ss.get("pdf_cover_text", "") or ""
    if mode == "Upload PDF":
        return ss.get("pdf_cover_pdf_text", "") or ""
    return ""


def _render_cover_letter_feedback(feedback: CoverLetterFeedback) -> None:
    """Render the cover letter analyzer output as a professional feedback panel."""
    st.markdown("---")
    st.markdown("### ✉️ Cover Letter Feedback")

    head_left, head_right = st.columns([1, 3])
    with head_left:
        st.metric("Letter quality", f"{feedback.score:.2f}")
        st.progress(feedback.score)
        if feedback.verdict:
            st.caption(feedback.verdict)
        if feedback.tone:
            st.caption(f"Tone: {feedback.tone}")
    with head_right:
        if feedback.overall_assessment:
            st.markdown("**Overall assessment**")
            st.write(feedback.overall_assessment)

    # Recommendation banner — green when genuinely ready, neutral info
    # otherwise. We avoid a red/error tier to keep the tone professional
    # rather than adversarial; the message itself carries the directness.
    bottom_line = feedback.bottom_line or feedback.send_recommendation
    if bottom_line:
        if feedback.is_send_ready:
            st.success(f"**Recommendation:** {bottom_line}")
        else:
            st.info(f"**Recommendation:** {bottom_line}")

    if (
        feedback.send_recommendation
        and feedback.send_recommendation not in (bottom_line or "")
    ):
        st.caption(f"Status: {feedback.send_recommendation}")

    if feedback.priority_revisions:
        st.markdown("**Priority revisions**")
        st.caption("Address these first to most strengthen the letter.")
        for item in feedback.priority_revisions:
            st.markdown(f"- {item}")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Suggested refinements**")
        if feedback.improvements:
            for item in feedback.improvements:
                st.markdown(f"- {item}")
        else:
            st.caption("No further refinements suggested by the reviewer.")
    with cols[1]:
        st.markdown("**Strengths**")
        if feedback.strengths:
            for item in feedback.strengths:
                st.markdown(f"- {item}")
        else:
            st.caption(
                "The reviewer did not highlight specific standout strengths."
            )

    align_cols = st.columns(2)
    with align_cols[0]:
        if feedback.jd_alignment:
            st.markdown("**Alignment with the job description**")
            st.write(feedback.jd_alignment)
    with align_cols[1]:
        if feedback.cv_alignment:
            st.markdown("**How it complements your CV**")
            st.write(feedback.cv_alignment)

    with st.expander("Raw model response"):
        st.code(
            feedback.raw_response or json.dumps(feedback.to_dict(), indent=2)
        )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="CV Checker",
        page_icon="📄",
        layout="wide",
    )

    with st.sidebar:
        st.title("📄 CV Checker")
        st.caption("AI-powered resume screening with GDPR-compliant PII removal.")
        page = st.radio(
            "Navigation",
            options=[PAGE_PDF, PAGE_PLAYGROUND, PAGE_RANK, PAGE_AUDIT],
            index=0,
        )
        st.markdown("---")
        _render_scorer_config()
        st.markdown("---")

    if page == PAGE_PDF:
        page_pdf()
    elif page == PAGE_PLAYGROUND:
        page_playground()
    elif page == PAGE_RANK:
        page_rank()
    else:
        page_audit()


def _render_scorer_config() -> None:
    """Sidebar form for picking the Claude provider (Vertex AI or Claude API)."""
    import os

    cfg = st.session_state.setdefault(
        "scorer_cfg",
        {
            "provider": (
                PROVIDER_CLAUDE_API if os.getenv("ANTHROPIC_API_KEY") else PROVIDER_VERTEX
            ),
            "project_id": os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "",
            "region": os.getenv("VERTEX_REGION") or RelevancyScorer.DEFAULT_REGION,
            "api_key": os.getenv("ANTHROPIC_API_KEY") or "",
        },
    )

    st.subheader("Claude provider")
    st.caption("Pick how the Claude Sonnet 4.6 scorer is called.")

    provider_labels = {
        PROVIDER_CLAUDE_API: "Anthropic Claude API (API key)",
        PROVIDER_VERTEX: "Google Vertex AI (GCP)",
    }
    provider_options = [PROVIDER_CLAUDE_API, PROVIDER_VERTEX]
    current_provider = cfg.get("provider", PROVIDER_VERTEX)
    if current_provider not in provider_options:
        current_provider = PROVIDER_VERTEX

    # NOTE: keep the provider radio OUTSIDE the form so changing it triggers an
    # immediate rerun and the matching credential fields appear right away.
    # Widgets inside st.form do not cause reruns until the form is submitted.
    provider = st.radio(
        "Provider",
        options=provider_options,
        index=provider_options.index(current_provider),
        format_func=lambda p: provider_labels[p],
        key="scorer_cfg_provider_radio",
    )

    with st.form("scorer-cfg", clear_on_submit=False, border=False):
        if provider == PROVIDER_CLAUDE_API:
            api_key = st.text_input(
                "Anthropic API key",
                value=cfg.get("api_key", ""),
                type="password",
                placeholder="sk-ant-...",
                help="Stored in this Streamlit session only. Get one at console.anthropic.com.",
            )
            project_id = cfg.get("project_id", "")
            region = cfg.get("region", "") or RelevancyScorer.DEFAULT_REGION
        else:
            project_id = st.text_input(
                "GCloud project ID",
                value=cfg.get("project_id", ""),
                placeholder="my-gcp-project",
            )
            region = st.text_input(
                "Region",
                value=cfg.get("region", "") or RelevancyScorer.DEFAULT_REGION,
                placeholder="europe-west1",
                help="Any Vertex AI region that serves claude-sonnet-4-6.",
            )
            api_key = cfg.get("api_key", "")

        submitted = st.form_submit_button("Apply", use_container_width=True)

    if submitted:
        new_cfg = {
            "provider": provider,
            "project_id": (project_id or "").strip(),
            "region": (region or "").strip() or RelevancyScorer.DEFAULT_REGION,
            "api_key": (api_key or "").strip(),
        }
        changed = any(new_cfg[k] != cfg.get(k, "") for k in new_cfg)
        cfg.update(new_cfg)

        if provider == PROVIDER_CLAUDE_API and not cfg["api_key"]:
            st.warning("No API key set — scoring will fail until you provide one.")
        if provider == PROVIDER_VERTEX and not cfg["project_id"]:
            st.warning(
                "No GCloud project set — Vertex calls will fall back to your "
                "default ADC project (if any)."
            )

        if changed:
            _build_scorer.clear()
            _build_cover_letter_analyzer.clear()
            st.session_state.pop("score_cache", None)
            st.success("Scorer config updated — scorer + score cache reset.")
        else:
            st.info("No changes.")

    status: List[str] = [f"provider `{provider_labels[cfg['provider']]}`"]
    if cfg["provider"] == PROVIDER_CLAUDE_API:
        status.append("API key set" if cfg.get("api_key") else "no API key")
    else:
        status.append(
            f"project `{cfg['project_id']}`" if cfg.get("project_id") else "no project set"
        )
        status.append(f"region `{cfg['region']}`")
    st.caption(" · ".join(status))


if __name__ == "__main__":
    main()
