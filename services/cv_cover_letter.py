"""
Cover Letter Analyzer — reviews a cover letter alongside the CV and Job Description.

Sits on top of the existing :mod:`relevancy_scorer` pipeline. When a candidate
provides a cover letter (uploaded PDF or pasted text), this module asks Claude
Sonnet 4.6 to act as an experienced reviewer and provide *direct, professional,
and constructive* feedback:

* a clear, measured assessment of whether the letter is ready to send;
* specific, actionable suggestions framed as refinements;
* honest recognition of genuine strengths (no manufactured praise);
* a one-line, professionally worded recommendation.

The class mirrors :class:`RelevancyScorer` so that it picks up the same
provider configuration (Vertex AI / Anthropic Claude API) from the Streamlit
sidebar — no extra credentials needed.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .relevancy_scorer import (
    PROVIDER_CLAUDE_API,
    PROVIDER_VERTEX,
    SUPPORTED_PROVIDERS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────
@dataclass
class CoverLetterFeedback:
    """Structured feedback for a candidate's cover letter."""

    score: float = 0.0                          # 0.0 – 1.0 overall quality
    verdict: str = ""                           # e.g. "Strong", "Solid", "Developing"
    send_recommendation: str = ""               # "Ready to send" | "Minor revisions recommended" | "Substantial revisions recommended"
    overall_assessment: str = ""                # 2-4 sentence professional summary
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    priority_revisions: List[str] = field(default_factory=list)  # most impactful changes to address first
    jd_alignment: str = ""                      # how the letter speaks to the JD
    cv_alignment: str = ""                      # how it complements the CV
    tone: str = ""                              # short tone label
    bottom_line: str = ""                       # one-line professional recommendation
    raw_response: str = ""                      # full text returned by the LLM

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def is_send_ready(self) -> bool:
        """Genuinely send-ready: strong score AND no priority revisions."""
        return self.score >= 0.80 and not self.priority_revisions

    @property
    def needs_revision(self) -> bool:
        """The letter would benefit from meaningful revisions before sending."""
        return self.score < 0.55 or bool(self.priority_revisions)


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an experienced career coach and senior hiring manager "
    "reviewing a candidate's cover letter against their (anonymized) CV "
    "and the target job description. Your role is to provide honest, "
    "professional, and constructive feedback — direct and specific so the "
    "candidate knows where they stand and what to refine, while keeping a "
    "respectful, measured, professional tone throughout.\n\n"
    "Guiding principles:\n"
    "  • Be specific. Reference particular phrases, paragraphs, or "
    "    omissions rather than vague generalities.\n"
    "  • Be constructive. Every concern should come paired with a clear, "
    "    practical way to address it.\n"
    "  • Be honest. Do not invent strengths, but recognise genuine ones "
    "    when they are present. Do not soften so much that the candidate "
    "    misreads the verdict.\n"
    "  • Be measured. Avoid harsh, dismissive, or emotionally loaded "
    "    language ('weak', 'bad', 'generic mess', 'desperate'). Frame "
    "    issues as opportunities to strengthen the letter.\n"
    "  • Choose words carefully. Prefer professional phrasing such as "
    "    'would benefit from', 'could be strengthened by', 'consider "
    "    rephrasing', 'this section reads as generic and could be "
    "    tailored to…' over blunt or judgmental wording.\n"
    "  • Be focused. Surface the changes that will most improve the "
    "    candidate's chances — not every minor preference.\n\n"
    "Evaluate the letter on:\n"
    "  • Specificity and use of evidence (concrete projects, outcomes, "
    "    metrics) versus vague claims.\n"
    "  • Alignment with the job description's stated needs.\n"
    "  • Whether the letter adds narrative beyond the CV or simply "
    "    restates it.\n"
    "  • Structure, clarity, and writing quality (opening, flow, "
    "    concision, polish).\n"
    "  • Tone (confident, professional, authentic).\n"
    "  • Length and focus.\n\n"
    "You MUST respond with a SINGLE valid JSON object and nothing else — no "
    "prose before or after, no markdown fences. Use this exact schema:\n"
    "{\n"
    '  "score": <float between 0.0 and 1.0, two decimals>,\n'
    '  "verdict": <"Excellent" | "Strong" | "Solid" | "Developing" | "Early draft">,\n'
    '  "send_recommendation": <"Ready to send" | "Minor revisions recommended" | "Substantial revisions recommended">,\n'
    '  "overall_assessment": <2-4 sentences. Professional, direct summary of how the letter reads as a whole and whether it currently makes the case for the candidate. No hedging, no harsh language.>,\n'
    '  "strengths": [<0-4 specific things the letter genuinely does well; empty list if there are none worth naming>],\n'
    '  "improvements": [<2-5 concrete, professionally worded suggestions — reference the relevant part of the letter and describe the change you would recommend>],\n'
    '  "priority_revisions": [<0-3 of the most impactful changes the candidate should make first; empty list if none. Phrase as recommendations, e.g. "Tailor the opening paragraph to reference the specific role and team", "Replace generic phrasing such as \'passionate about technology\' with a concrete example from your experience", "Address the JD\'s requirement around X, which is currently not mentioned">],\n'
    '  "jd_alignment": <1-2 sentences. Be specific about what is and isn\'t addressed.>,\n'
    '  "cv_alignment": <1-2 sentences on how the letter complements (or merely repeats) the CV.>,\n'
    '  "tone": <short professional label, e.g. "Confident and specific", "Professional but generic", "Slightly formal", "Authentic and warm">,\n'
    '  "bottom_line": <ONE professionally worded sentence summarising the recommendation, e.g. "Ready to send after a brief proofread." or "Recommend revising the opening paragraph and tightening the alignment to the role before sending.">\n'
    "}\n\n"
    "Scoring rubric (cover letter quality, NOT job fit):\n"
    "  • 0.90–1.00 : Excellent — specific, tailored, polished. Ready to send.\n"
    "  • 0.75–0.89 : Strong — convincing; minor refinements would polish it further.\n"
    "  • 0.55–0.74 : Solid — readable but would benefit from targeted revisions before sending.\n"
    "  • 0.30–0.54 : Developing — needs meaningful revisions to make a clear case for the candidate.\n"
    "  • 0.00–0.29 : Early draft — substantial revisions recommended before this letter is ready.\n"
    "Be decisive — avoid clustering scores around 0.5."
)

USER_PROMPT_TEMPLATE = (
    "Please review the following cover letter for the role described "
    "below. Use the candidate's CV (PII removed) for context on what "
    "they have done.\n\n"
    "Provide direct, professional, and constructive feedback. Be specific "
    "— reference the relevant sections of the letter when making "
    "recommendations. Recognise genuine strengths where they exist, but "
    "do not manufacture praise. When the letter would benefit from "
    "revision, say so clearly and explain what to change and why.\n\n"
    "=== JOB DESCRIPTION ===\n{job_description}\n\n"
    "=== CANDIDATE CV (PII REMOVED) ===\n{cv_text}\n\n"
    "=== COVER LETTER ===\n{cover_letter}\n\n"
    "Respond with the JSON object described in the system prompt."
)


# ──────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────
class CoverLetterAnalyzer:
    """Cover letter feedback generator backed by Claude Sonnet 4.6.

    Provider configuration mirrors :class:`RelevancyScorer` so the same
    Streamlit sidebar settings apply.

    Example:
        analyzer = CoverLetterAnalyzer(provider="claude_api", api_key="sk-ant-...")
        feedback = analyzer.analyze(
            cover_letter=letter_text,
            cv_text=clean_cv,
            job_description=jd_text,
        )
        print(feedback.score, feedback.encouragement)
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_REGION = "europe-west1"
    DEFAULT_MAX_TOKENS = 1500

    def __init__(
        self,
        model: Optional[str] = None,
        region: Optional[str] = None,
        project_id: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        provider: str = PROVIDER_VERTEX,
        api_key: Optional[str] = None,
    ):
        provider = (provider or PROVIDER_VERTEX).lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}. "
                f"Expected one of: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        self.provider = provider
        self.model = model or self.DEFAULT_MODEL
        self.region = region or os.getenv("VERTEX_REGION", self.DEFAULT_REGION)
        self.project_id = (
            project_id
            or os.getenv("VERTEX_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    # ── Lazy client init (matches RelevancyScorer behaviour) ────
    @property
    def client(self):
        if self._client is not None:
            return self._client

        if self.provider == PROVIDER_CLAUDE_API:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The `anthropic` package is required for the Claude API "
                    "provider. Install with `uv add anthropic`."
                ) from exc

            if not self.api_key:
                raise RuntimeError(
                    "Claude API provider selected but no API key was provided. "
                    "Set ANTHROPIC_API_KEY or pass `api_key=...` to the analyzer."
                )
            self._client = Anthropic(api_key=self.api_key)
            logger.info(
                "Initialized Anthropic (Claude API) client for cover letter "
                "analyzer (model=%s)",
                self.model,
            )
            return self._client

        try:
            from anthropic import AnthropicVertex
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anthropic[vertex] is required. Install with "
                "`uv add 'anthropic[vertex]'`."
            ) from exc

        kwargs = {"region": self.region}
        if self.project_id:
            kwargs["project_id"] = self.project_id
        self._client = AnthropicVertex(**kwargs)
        logger.info(
            "Initialized AnthropicVertex client for cover letter analyzer "
            "(model=%s, region=%s, project=%s)",
            self.model,
            self.region,
            self.project_id,
        )
        return self._client

    # ── Public API ──────────────────────────────────────────────
    def analyze(
        self,
        cover_letter: str,
        cv_text: str,
        job_description: str,
    ) -> CoverLetterFeedback:
        """Analyze a cover letter against the CV and job description."""
        cover_letter = (cover_letter or "").strip()
        cv_text = (cv_text or "").strip()
        job_description = (job_description or "").strip()

        if not cover_letter:
            return CoverLetterFeedback(
                score=0.0,
                verdict="",
                overall_assessment=(
                    "No cover letter was provided, so no feedback was generated."
                ),
            )
        if not job_description:
            return CoverLetterFeedback(
                score=0.0,
                verdict="",
                overall_assessment=(
                    "A job description is required to evaluate a cover letter "
                    "in context."
                ),
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            job_description=job_description,
            cv_text=cv_text or "(no CV text provided)",
            cover_letter=cover_letter,
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = self._extract_text(message)
            return self._parse_response(raw)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Cover letter analysis failed: %s", exc)
            return CoverLetterFeedback(
                score=0.0,
                verdict="",
                overall_assessment=(
                    "Could not generate cover letter feedback right now. "
                    f"Error: {exc}"
                ),
                bottom_line="Analyzer unavailable — try again in a moment.",
            )

    # ── Internals ───────────────────────────────────────────────
    @staticmethod
    def _extract_text(message) -> str:
        """Collapse Claude's content blocks into a single string."""
        parts: List[str] = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    def _parse_response(self, raw: str) -> CoverLetterFeedback:
        """Parse Claude's JSON response, tolerating minor formatting noise."""
        payload = self._extract_json(raw)
        if payload is None:
            logger.warning(
                "Could not parse JSON from cover letter response. Raw=%s",
                raw[:300],
            )
            return CoverLetterFeedback(
                score=0.0,
                verdict="",
                overall_assessment=(
                    "The model response could not be parsed as JSON. The raw "
                    "text is preserved below for manual review."
                ),
                raw_response=raw,
            )

        return CoverLetterFeedback(
            score=self._clamp(float(payload.get("score", 0.0))),
            verdict=str(payload.get("verdict", "")).strip(),
            send_recommendation=str(payload.get("send_recommendation", "")).strip(),
            overall_assessment=str(payload.get("overall_assessment", "")).strip(),
            strengths=[
                str(s).strip()
                for s in (payload.get("strengths") or [])
                if str(s).strip()
            ],
            improvements=[
                str(s).strip()
                for s in (payload.get("improvements") or [])
                if str(s).strip()
            ],
            priority_revisions=[
                str(s).strip()
                for s in (
                    payload.get("priority_revisions")
                    # Backwards-compat with earlier prompt field name.
                    or payload.get("critical_issues")
                    or []
                )
                if str(s).strip()
            ],
            jd_alignment=str(payload.get("jd_alignment", "")).strip(),
            cv_alignment=str(payload.get("cv_alignment", "")).strip(),
            tone=str(payload.get("tone", "")).strip(),
            bottom_line=str(payload.get("bottom_line", "")).strip(),
            raw_response=raw,
        )

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict]:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(max(0.0, min(1.0, v)), 2)


__all__ = [
    "CoverLetterAnalyzer",
    "CoverLetterFeedback",
]
