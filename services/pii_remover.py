"""
NER-based PII Removal Service (GDPR Compliance Layer)

This module uses spaCy's Named Entity Recognition to identify and remove
Personally Identifiable Information (PII) from CV text before AI scoring.

PII categories handled:
  - Names (PERSON entities via spaCy NER)
  - Email addresses (regex pattern)
  - Phone numbers (regex pattern)
  - Physical addresses (GPE, LOC, FAC entities + keyword detection)
  - Photo references (keyword detection)

Production Notes:
  - Replace en_core_web_sm with en_core_web_trf for higher accuracy
  - Add custom NER model trained on CV-specific data
  - Consider Presidio (Microsoft) for enterprise PII detection
  - Store audit logs in encrypted database
  - Add consent management integration
"""
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


DEFAULT_AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "pii_log.json"


# ──────────────────────────────────────────────
# PII Detection Result
# ──────────────────────────────────────────────
@dataclass
class PIIEntity:
    """A detected PII entity in the text."""
    text: str
    label: str          # PERSON, EMAIL, PHONE, ADDRESS, PHOTO
    start: int          # Character start position
    end: int            # Character end position
    confidence: float = 1.0
    replaced_with: str = "[REDACTED]"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "type": self.label,
            "value": self.text,
            "replaced_with": self.replaced_with,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }


@dataclass
class AnonymizationResult:
    """Result of the anonymization process."""
    original_text: str
    anonymized_text: str
    entities_removed: List[PIIEntity] = field(default_factory=list)
    entity_count: int = 0
    pii_categories_found: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        cats = ", ".join(self.pii_categories_found) if self.pii_categories_found else "None"
        return f"Removed {self.entity_count} PII entities. Categories: {cats}"


# ──────────────────────────────────────────────
# Regex Patterns for PII Detection
# ──────────────────────────────────────────────
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'
)

PHONE_PATTERN = re.compile(
    r'(?:\+?\d{1,3}[\s\-.]?)?'     # Country code
    r'(?:\(?\d{1,4}\)?[\s\-.]?)?'   # Area code
    r'\d{2,4}[\s\-.]?'              # First part
    r'\d{3,4}'                       # Second part
    r'(?:[\s\-.]?\d{1,4})?'         # Optional extension
)

# Common address indicators
ADDRESS_KEYWORDS = [
    "street", "st.", "avenue", "ave.", "road", "rd.", "boulevard", "blvd.",
    "drive", "dr.", "lane", "ln.", "court", "ct.", "suite", "ste.",
    "apartment", "apt.", "floor", "building", "bldg.",
    "zip", "postal", "postcode", "p.o. box",
]

# Photo/image reference keywords
PHOTO_KEYWORDS = [
    "photo", "headshot", "portrait", "picture", "image", "photograph",
    "profile photo", "profile picture", "profile image",
]

# URL pattern (LinkedIn, personal sites with PII)
URL_PATTERN = re.compile(
    r'https?://(?:www\.)?(?:linkedin\.com/in/|facebook\.com/|twitter\.com/|github\.com/)\S+'
)

# Social media handles
SOCIAL_PATTERN = re.compile(
    r'@[A-Za-z0-9_]{2,30}'
)

# Date of birth patterns
DOB_PATTERN = re.compile(
    r'\b(?:date\s+of\s+birth|d\.?o\.?b\.?|born)\s*:?\s*'
    r'(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
    re.IGNORECASE
)


# ──────────────────────────────────────────────
# NER PII Removal Service
# ──────────────────────────────────────────────
class PIIRemovalService:
    """
    GDPR-compliant PII removal using NER + regex patterns.

    Usage:
        service = PIIRemovalService()
        result = service.anonymize(cv_text)
        clean_text = result.anonymized_text
    """

    REPLACEMENT_TOKEN = "[REDACTED]"

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        audit_log_path: Union[str, Path] = DEFAULT_AUDIT_LOG_PATH,
    ):
        """
        Initialize the PII removal service.

        Args:
            spacy_model: spaCy model name. Use 'en_core_web_trf' for
                         transformer-based higher accuracy in production.
            audit_log_path: Path to the JSON file used to persist the
                         GDPR-compliant audit trail of redactions.
        """
        self.nlp = None
        self.spacy_model = spacy_model
        self.audit_log_path = Path(audit_log_path)
        self._load_model()

    def _load_model(self):
        """Load spaCy NLP model with graceful fallback."""
        try:
            import spacy
            self.nlp = spacy.load(self.spacy_model)
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
        except OSError:
            logger.warning(
                f"spaCy model '{self.spacy_model}' not found. "
                f"Falling back to regex-only PII detection. "
                f"Run: python -m spacy download {self.spacy_model}"
            )
            self.nlp = None

    def anonymize(
        self,
        text: str,
        candidate_id: Optional[str] = None,
        log_to_file: bool = True,
    ) -> AnonymizationResult:
        """
        Remove all PII from the given text.

        Pipeline:
          1. spaCy NER detection (names, locations, organizations)
          2. Regex-based detection (email, phone, URLs, DOB)
          3. Keyword-based detection (addresses, photo references)
          4. Replace all detected PII with [REDACTED]

        Args:
            text: Raw CV text containing potential PII

        Returns:
            AnonymizationResult with cleaned text and audit trail
        """
        if not text or not text.strip():
            return AnonymizationResult(
                original_text=text,
                anonymized_text=text,
            )

        entities: List[PIIEntity] = []

        # ── Step 1: spaCy NER Detection ──
        entities.extend(self._detect_ner_entities(text))

        # ── Step 2: Regex-based Detection ──
        entities.extend(self._detect_emails(text))
        entities.extend(self._detect_phones(text))
        entities.extend(self._detect_urls(text))
        entities.extend(self._detect_social_handles(text))
        entities.extend(self._detect_dob(text))

        # ── Step 3: Keyword-based Detection ──
        entities.extend(self._detect_address_lines(text))
        entities.extend(self._detect_photo_references(text))

        # ── Step 4: Deduplicate and sort (longest first for clean replacement) ──
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

        # ── Step 5: Replace PII with redaction tokens ──
        anonymized_text = self._replace_entities(text, entities)

        # ── Build result ──
        categories = list(set(e.label for e in entities))
        result = AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized_text,
            entities_removed=entities,
            entity_count=len(entities),
            pii_categories_found=categories,
        )

        logger.info(f"PII Anonymization: {result.summary}")

        if log_to_file:
            try:
                self.append_audit_entry(result, candidate_id=candidate_id)
            except Exception as exc:  # audit must never break the pipeline
                logger.error(f"Failed to write PII audit log: {exc}")

        return result

    # ──────────────────────────────────────────
    # GDPR Audit Log (pii_log.json)
    # ──────────────────────────────────────────
    def append_audit_entry(
        self,
        result: "AnonymizationResult",
        candidate_id: Optional[str] = None,
    ) -> Dict:
        """Append a redaction record to the JSON audit log.

        The log is a list of entries shaped like::

            {
                "candidate_id": "cv-006",
                "timestamp": "2026-04-17T09:17:59",
                "entity_count": 12,
                "categories": ["NAME", "EMAIL", ...],
                "entities": [{"type": "NAME", "value": "...", ...}, ...]
            }
        """
        entry = {
            "candidate_id": candidate_id or f"candidate-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.utcnow().isoformat(),
            "entity_count": result.entity_count,
            "categories": sorted(set(result.pii_categories_found)),
            "original_length": len(result.original_text),
            "anonymized_length": len(result.anonymized_text),
            "entities": [ent.to_dict() for ent in result.entities_removed],
        }

        log_data = self._load_audit_log()
        log_data.append(entry)
        self._write_audit_log(log_data)
        return entry

    def _load_audit_log(self) -> List[Dict]:
        if not self.audit_log_path.exists() or self.audit_log_path.stat().st_size == 0:
            return []
        try:
            with self.audit_log_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            logger.warning("Audit log is not a list; starting fresh.")
            return []
        except json.JSONDecodeError:
            logger.warning("Audit log corrupted; starting fresh.")
            return []

    def _write_audit_log(self, data: List[Dict]) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def clear_audit_log(self) -> None:
        """Clear all audit records (useful for demo resets)."""
        self._write_audit_log([])

    # ──────────────────────────────────────────
    # Detection Methods
    # ──────────────────────────────────────────
    def _detect_ner_entities(self, text: str) -> List[PIIEntity]:
        """Use spaCy NER to detect named entities."""
        if self.nlp is None:
            return []

        entities = []
        doc = self.nlp(text)

        # Target entity labels for PII
        pii_labels = {"PERSON", "GPE", "LOC", "FAC"}

        for ent in doc.ents:
            if ent.label_ in pii_labels:
                entities.append(PIIEntity(
                    text=ent.text,
                    label=self._map_spacy_label(ent.label_),
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,  # spaCy typical confidence
                ))

        return entities

    def _detect_emails(self, text: str) -> List[PIIEntity]:
        """Detect email addresses using regex."""
        return [
            PIIEntity(
                text=match.group(),
                label="EMAIL",
                start=match.start(),
                end=match.end(),
            )
            for match in EMAIL_PATTERN.finditer(text)
        ]

    def _detect_phones(self, text: str) -> List[PIIEntity]:
        """Detect phone numbers using regex."""
        entities = []
        for match in PHONE_PATTERN.finditer(text):
            phone = match.group().strip()
            # Filter out short numbers that are likely not phones
            digits = re.sub(r'\D', '', phone)
            if len(digits) >= 7:
                entities.append(PIIEntity(
                    text=phone,
                    label="PHONE",
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _detect_urls(self, text: str) -> List[PIIEntity]:
        """Detect social/personal profile URLs."""
        return [
            PIIEntity(
                text=match.group(),
                label="URL",
                start=match.start(),
                end=match.end(),
            )
            for match in URL_PATTERN.finditer(text)
        ]

    def _detect_social_handles(self, text: str) -> List[PIIEntity]:
        """Detect social media handles (@username)."""
        return [
            PIIEntity(
                text=match.group(),
                label="SOCIAL",
                start=match.start(),
                end=match.end(),
            )
            for match in SOCIAL_PATTERN.finditer(text)
        ]

    def _detect_dob(self, text: str) -> List[PIIEntity]:
        """Detect date of birth patterns."""
        return [
            PIIEntity(
                text=match.group(),
                label="DOB",
                start=match.start(),
                end=match.end(),
            )
            for match in DOB_PATTERN.finditer(text)
        ]

    def _detect_address_lines(self, text: str) -> List[PIIEntity]:
        """Detect physical address lines using keyword matching."""
        entities = []
        lines = text.split('\n')
        offset = 0

        for line in lines:
            line_lower = line.lower().strip()
            if any(kw in line_lower for kw in ADDRESS_KEYWORDS):
                start = text.find(line, offset)
                if start != -1:
                    entities.append(PIIEntity(
                        text=line.strip(),
                        label="ADDRESS",
                        start=start,
                        end=start + len(line.rstrip()),
                        confidence=0.70,
                    ))
            offset += len(line) + 1  # +1 for newline

        return entities

    def _detect_photo_references(self, text: str) -> List[PIIEntity]:
        """Detect photo/image references in CV text."""
        entities = []
        text_lower = text.lower()

        for keyword in PHOTO_KEYWORDS:
            start = 0
            while True:
                idx = text_lower.find(keyword, start)
                if idx == -1:
                    break
                entities.append(PIIEntity(
                    text=text[idx:idx + len(keyword)],
                    label="PHOTO",
                    start=idx,
                    end=idx + len(keyword),
                    confidence=0.75,
                ))
                start = idx + 1

        return entities

    # ──────────────────────────────────────────
    # Utility Methods
    # ──────────────────────────────────────────
    @staticmethod
    def _map_spacy_label(label: str) -> str:
        """Map spaCy entity labels to our PII categories."""
        mapping = {
            "PERSON": "NAME",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "FAC": "ADDRESS",
            "ORG": "ORGANIZATION",
        }
        return mapping.get(label, label)

    @staticmethod
    def _deduplicate_entities(entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove overlapping entities, keeping the longer match."""
        if not entities:
            return []

        # Sort by start position, then by length (longest first)
        sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        result = [sorted_ents[0]]

        for ent in sorted_ents[1:]:
            last = result[-1]
            # Skip if this entity overlaps with the last kept one
            if ent.start < last.end:
                continue
            result.append(ent)

        return result

    def _replace_entities(self, text: str, entities: List[PIIEntity]) -> str:
        """Replace detected PII entities with redaction token."""
        if not entities:
            return text

        # Sort by position in reverse to replace from end to start
        sorted_ents = sorted(entities, key=lambda e: e.start, reverse=True)

        result = text
        for ent in sorted_ents:
            result = result[:ent.start] + self.REPLACEMENT_TOKEN + result[ent.end:]

        # Clean up multiple consecutive redaction tokens
        while f"{self.REPLACEMENT_TOKEN} {self.REPLACEMENT_TOKEN}" in result:
            result = result.replace(
                f"{self.REPLACEMENT_TOKEN} {self.REPLACEMENT_TOKEN}",
                self.REPLACEMENT_TOKEN
            )

        return result

    def get_pii_summary(self, result: AnonymizationResult) -> Dict:
        """
        Generate a GDPR compliance summary of PII removal.

        Returns dict suitable for audit logging.
        """
        category_counts = {}
        for ent in result.entities_removed:
            category_counts[ent.label] = category_counts.get(ent.label, 0) + 1

        return {
            "total_pii_entities": result.entity_count,
            "categories": category_counts,
            "categories_found": result.pii_categories_found,
            "text_length_original": len(result.original_text),
            "text_length_anonymized": len(result.anonymized_text),
            "anonymization_complete": True,
        }


