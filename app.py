"""
CV Checker — Streamlit app.

Three pages:
  1. CV Score Rank  — rank candidate CVs for each selected job description.
  2. GDPR Audit Log — browse the pii_log.json produced during anonymization.
  3. Job ⇄ CV Score — free-text playground (remove PII + score).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from pii_remover import AnonymizationResult, PIIRemovalService
from relevancy_scorer import RelevancyScore, RelevancyScorer

# ──────────────────────────────────────────────
# Constants / paths
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CV_DIR = ROOT / "cv"
JOB_DIR = ROOT / "job_description"
AUDIT_LOG = ROOT / "pii_log.json"

PAGE_RANK = "CV Score Rank"
PAGE_AUDIT = "GDPR Audit Log"
PAGE_PLAYGROUND = "Job ⇄ CV Score"


# ──────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pii_service() -> PIIRemovalService:
    return PIIRemovalService(audit_log_path=AUDIT_LOG)


@st.cache_resource(show_spinner=False)
def _build_scorer(project_id: str, region: str) -> RelevancyScorer:
    """Build a scorer keyed by (project_id, region) so edits invalidate the cache."""
    return RelevancyScorer(
        project_id=project_id or None,
        region=region or None,
    )


def get_scorer() -> RelevancyScorer:
    cfg = st.session_state.get("vertex_cfg", {})
    return _build_scorer(cfg.get("project_id", ""), cfg.get("region", ""))


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
    st.title("CV Score Rank")
    st.caption(
        "Select one or more job descriptions. The app removes PII from every "
        "CV and then asks Claude Sonnet 4.6 (on Vertex AI) for a 0.0 – 1.0 fit score."
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
            "No audit entries yet. Visit **CV Score Rank** (or click *Regenerate from "
            "sample CVs* in the sidebar) to populate the log."
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
    st.title("Job ⇄ CV Score")
    st.caption(
        "Paste any job description and CV, remove PII, then compute a fit score."
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
            options=[PAGE_RANK, PAGE_AUDIT, PAGE_PLAYGROUND],
            index=0,
        )
        st.markdown("---")
        _render_vertex_config()
        st.markdown("---")

    if page == PAGE_RANK:
        page_rank()
    elif page == PAGE_AUDIT:
        page_audit()
    else:
        page_playground()


def _render_vertex_config() -> None:
    """Sidebar form for the Vertex AI project_id / region used by the scorer."""
    import os

    cfg = st.session_state.setdefault(
        "vertex_cfg",
        {
            "project_id": os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "",
            "region": os.getenv("VERTEX_REGION") or RelevancyScorer.DEFAULT_REGION,
        },
    )

    st.subheader("Vertex AI config")
    st.caption("Used by the Claude Sonnet 4.6 scorer.")
    with st.form("vertex-cfg", clear_on_submit=False, border=False):
        project_id = st.text_input(
            "GCloud project ID",
            value=cfg["project_id"],
            placeholder="my-gcp-project",
        )
        region = st.text_input(
            "Region",
            value=cfg["region"],
            placeholder="europe-west1",
            help="Any Vertex AI region that serves claude-sonnet-4-6.",
        )
        submitted = st.form_submit_button("Apply", use_container_width=True)

    if submitted:
        changed = (project_id.strip() != cfg["project_id"]) or (region.strip() != cfg["region"])
        cfg["project_id"] = project_id.strip()
        cfg["region"] = region.strip() or RelevancyScorer.DEFAULT_REGION
        if changed:
            _build_scorer.clear()
            st.session_state.pop("score_cache", None)
            st.success("Vertex config updated — scorer + score cache reset.")
        else:
            st.info("No changes.")

    status = []
    if cfg["project_id"]:
        status.append(f"project `{cfg['project_id']}`")
    else:
        status.append("no project set")
    status.append(f"region `{cfg['region']}`")
    st.caption(" · ".join(status))


if __name__ == "__main__":
    main()
