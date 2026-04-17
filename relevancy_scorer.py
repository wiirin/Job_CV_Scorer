"""
Relevancy Scorer — CV ⇄ Job Description matching via Claude Sonnet 4.6 on Vertex.

Uses prompt engineering to coerce Claude into returning a strict JSON payload
containing a float score in [0.0, 1.0] plus a short rationale and skill
breakdown. The scorer is resilient to slightly malformed JSON and provides a
deterministic stub fallback for local/offline development.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────
@dataclass
class RelevancyScore:
    """Structured output returned by the scorer."""
    score: float                        # 0.0 – 1.0
    verdict: str = ""                   # short label e.g. "Strong match"
    reasoning: str = ""                 # 1-3 sentence explanation
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    experience_fit: str = ""            # under / match / over-qualified
    raw_response: str = ""              # full text returned by the LLM

    def to_dict(self) -> Dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a senior technical recruiter and hiring manager. Your job is to "
    "evaluate how well a candidate's (anonymized) CV matches a given job "
    "description. Be rigorous, fair and unbiased: ignore any residual personal "
    "information and focus purely on skills, experience, seniority and domain fit.\n\n"
    "You MUST respond with a SINGLE valid JSON object and nothing else — no prose "
    "before or after, no markdown fences. Use this exact schema:\n"
    "{\n"
    '  "score": <float between 0.0 and 1.0, two decimals>,\n'
    '  "verdict": <"Strong match" | "Good match" | "Partial match" | "Weak match" | "No match">,\n'
    '  "reasoning": <1-3 sentence justification>,\n'
    '  "matched_skills": [<skills from the JD that the CV clearly demonstrates>],\n'
    '  "missing_skills": [<required skills from the JD that are absent>],\n'
    '  "experience_fit": <"under-qualified" | "match" | "over-qualified">\n'
    "}\n\n"
    "Scoring rubric:\n"
    "  • 0.90–1.00 : Exceptional fit — all must-haves + most nice-to-haves, right seniority.\n"
    "  • 0.75–0.89 : Strong fit — all must-haves, right seniority, some gaps in nice-to-haves.\n"
    "  • 0.55–0.74 : Partial fit — most must-haves but notable gaps or seniority mismatch.\n"
    "  • 0.30–0.54 : Weak fit — significant skill or domain gaps.\n"
    "  • 0.00–0.29 : Poor fit — role or domain mismatch.\n"
    "Be decisive and avoid clustering every score around 0.5."
)

USER_PROMPT_TEMPLATE = (
    "Evaluate the following candidate against the job description.\n\n"
    "=== JOB DESCRIPTION ===\n{job_description}\n\n"
    "=== CANDIDATE CV (PII REMOVED) ===\n{cv_text}\n\n"
    "Respond with the JSON object described in the system prompt."
)


# ──────────────────────────────────────────────
# Scorer
# ──────────────────────────────────────────────
class RelevancyScorer:
    """CV ⇄ Job relevancy scorer backed by Claude Sonnet 4.6 on Vertex AI.

    Example:
        scorer = RelevancyScorer()
        result = scorer.score(cv_text=clean_cv, job_description=jd_text)
        print(result.score, result.verdict)
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_REGION = "europe-west1"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        region: Optional[str] = None,
        project_id: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ):
        self.model = model
        self.region = region or os.getenv("VERTEX_REGION", self.DEFAULT_REGION)
        self.project_id = project_id or os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    # ── Lazy client init ────────────────────────────────────────
    @property
    def client(self):
        if self._client is None:
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
                "Initialized AnthropicVertex client (model=%s, region=%s, project=%s)",
                self.model, self.region, self.project_id,
            )
        return self._client

    # ── Public API ──────────────────────────────────────────────
    def score(self, cv_text: str, job_description: str) -> RelevancyScore:
        """Score a CV against a job description and return a RelevancyScore."""
        cv_text = (cv_text or "").strip()
        job_description = (job_description or "").strip()

        if not cv_text or not job_description:
            return RelevancyScore(
                score=0.0,
                verdict="No match",
                reasoning="Missing CV or job description input.",
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            job_description=job_description,
            cv_text=cv_text,
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
        except Exception as exc:  # pragma: no cover
            logger.exception("Scoring failed, falling back to heuristic: %s", exc)
            return self._heuristic_fallback(cv_text, job_description, error=str(exc))

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

    def _parse_response(self, raw: str) -> RelevancyScore:
        """Parse the model's JSON response, tolerating minor formatting noise."""
        payload = self._extract_json(raw)
        if payload is None:
            logger.warning("Could not parse JSON from model response. Raw=%s", raw[:300])
            # Last-ditch: look for a bare decimal as the score.
            m = re.search(r"(\d+\.\d+)", raw)
            guess = float(m.group(1)) if m else 0.0
            return RelevancyScore(
                score=self._clamp(guess),
                verdict="",
                reasoning="Model response could not be parsed as JSON.",
                raw_response=raw,
            )

        return RelevancyScore(
            score=self._clamp(float(payload.get("score", 0.0))),
            verdict=str(payload.get("verdict", "")).strip(),
            reasoning=str(payload.get("reasoning", "")).strip(),
            matched_skills=list(payload.get("matched_skills", []) or []),
            missing_skills=list(payload.get("missing_skills", []) or []),
            experience_fit=str(payload.get("experience_fit", "")).strip(),
            raw_response=raw,
        )

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict]:
        if not raw:
            return None
        # Try direct parse first.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Strip code fences if present.
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass
        # Fall back: first {...} block.
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

    # ── Offline fallback (keyword overlap) ──────────────────────
    @staticmethod
    def _heuristic_fallback(cv_text: str, job_description: str, error: str = "") -> RelevancyScore:
        """Very light keyword-overlap scorer used when the API call fails."""
        def tokens(text: str) -> set:
            return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\+\#\.\-]{2,}", text)}

        jd_tokens = tokens(job_description)
        cv_tokens = tokens(cv_text)
        if not jd_tokens:
            return RelevancyScore(score=0.0, verdict="No match", reasoning="Empty JD.")
        overlap = jd_tokens & cv_tokens
        score = round(min(1.0, len(overlap) / max(20, len(jd_tokens) * 0.5)), 2)
        return RelevancyScore(
            score=score,
            verdict="Heuristic (offline)",
            reasoning=(
                "Vertex call unavailable; used a keyword-overlap heuristic "
                f"({len(overlap)} tokens matched). Error: {error}"
            ),
            matched_skills=sorted(list(overlap))[:10],
        )
