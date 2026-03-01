"""Extract structured clinical concepts from clinical note text.

Uses local LLM (Ollama/vLLM) to identify conditions, symptoms, procedures,
medications, body sites, and lab values — with negation and qualifier detection.

Falls back to regex-based extraction if LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import re

from core.api.shared import ask_llm_nothink, strip_llm_wrapper
from core.validation.clinical_models import ClinicalConcept, ConceptCategory

log = logging.getLogger("noesis.validation")


async def extract_clinical_concepts(
    note_text: str,
    max_concepts: int = 30,
) -> list[ClinicalConcept]:
    """Extract clinical concepts from a clinical note.

    Uses the local LLM (with thinking disabled) to parse clinical language,
    handling negation, qualifiers, and categorization. Falls back to regex
    if LLM unavailable.

    Args:
        note_text: Raw clinical note text.
        max_concepts: Maximum number of concepts to extract.

    Returns:
        List of ClinicalConcept with term, category, negated, qualifier.
    """
    truncated = note_text[:3000]

    prompt = f"""You are a clinical NLP system. Extract medical concepts from this clinical note.

For each concept, return:
- "term": the clinical term as written (normalize abbreviations: RLQ → right lower quadrant)
- "category": one of: condition, symptom, procedure, medication, body_site, lab_value
- "negated": true if the concept is denied/absent/negative (e.g., "denies chest pain", "no fever")
- "qualifier": any modifier (e.g., "acute", "bilateral", "mild", "chronic") or null

Clinical note:
{truncated}

Rules:
- Extract up to {max_concepts} concepts
- Separate composite findings (e.g., "fever and chills" → two concepts)
- Lab values should include the value when present (e.g., "WBC 15000")
- Medications should include dosage if mentioned (e.g., "metformin 500mg")
- Body sites mentioned as standalone anatomical references are body_site category
- Symptoms are subjective complaints; conditions are assessed diagnoses
- Be thorough — capture every clinically relevant concept

Return ONLY a JSON array. Example:
[
  {{"term": "acute appendicitis", "category": "condition", "negated": false, "qualifier": "acute"}},
  {{"term": "right lower quadrant pain", "category": "symptom", "negated": false, "qualifier": null}},
  {{"term": "chest pain", "category": "symptom", "negated": true, "qualifier": null}}
]

Return ONLY the JSON array, nothing else."""

    try:
        answer = await ask_llm_nothink(prompt, temperature=0.1, max_tokens=2000, timeout=90.0)
        answer = strip_llm_wrapper(answer)

        # Try to extract JSON array if answer has extra text around it
        if not answer.startswith("["):
            bracket_match = re.search(r"\[.*\]", answer, re.DOTALL)
            if bracket_match:
                answer = bracket_match.group(0)

        raw = json.loads(answer)
        if not isinstance(raw, list):
            log.warning("LLM returned non-list for concept extraction")
            return _fallback_extract(note_text, max_concepts)

        concepts = []
        for item in raw[:max_concepts]:
            if not isinstance(item, dict) or "term" not in item:
                continue
            try:
                category = ConceptCategory(item.get("category", "symptom"))
            except ValueError:
                category = ConceptCategory.SYMPTOM
            concepts.append(
                ClinicalConcept(
                    term=item["term"].strip(),
                    category=category,
                    negated=bool(item.get("negated", False)),
                    qualifier=item.get("qualifier"),
                )
            )

        return concepts if concepts else _fallback_extract(note_text, max_concepts)

    except Exception as e:
        log.info("LLM clinical concept extraction unavailable, using fallback: %s", e)
        return _fallback_extract(note_text, max_concepts)


# ── Fallback regex extraction ────────────────────────────────────────────

# Common clinical abbreviations → expanded forms
_ABBREVIATIONS = {
    "rlq": "right lower quadrant",
    "ruq": "right upper quadrant",
    "llq": "left lower quadrant",
    "luq": "left upper quadrant",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "ha": "headache",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "uti": "urinary tract infection",
    "bph": "benign prostatic hyperplasia",
    "gerd": "gastroesophageal reflux disease",
    "ckd": "chronic kidney disease",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
}

# Negation patterns
_NEGATION_RE = re.compile(
    r"(?:no|denies|denied|negative for|without|absent|not|non)[- ]",
    re.IGNORECASE,
)

# Lab value pattern: "WBC 15,000" or "glucose 250 mg/dL" or "HbA1c 8.5%"
_LAB_RE = re.compile(
    r"\b(WBC|RBC|Hgb|Hct|PLT|BUN|Cr|glucose|HbA1c|A1c|Na|K|Cl|CO2|Ca|Mg|"
    r"albumin|AST|ALT|ALP|bilirubin|TSH|T4|PSA|INR|PT|PTT|ESR|CRP|troponin|"
    r"BNP|lactate|lipase|amylase)\s*(?:of\s+)?(\d[\d,]*\.?\d*)\s*(%|mg/dL|g/dL|"
    r"mEq/L|mmol/L|U/L|ng/mL|cells/mcL|K/uL|sec|mIU/L)?",
    re.IGNORECASE,
)

# Medication pattern: common drug names
_MEDICATION_TERMS = {
    "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
    "omeprazole", "pantoprazole", "metoprolol", "losartan", "gabapentin",
    "levothyroxine", "prednisone", "amoxicillin", "azithromycin",
    "ciprofloxacin", "ibuprofen", "acetaminophen", "aspirin", "warfarin",
    "heparin", "enoxaparin", "furosemide", "hydrochlorothiazide",
    "albuterol", "fluticasone", "montelukast", "sertraline", "fluoxetine",
    "escitalopram", "duloxetine", "tramadol", "oxycodone", "morphine",
    "hydrocodone", "clopidogrel", "apixaban", "rivaroxaban",
}


# Common clinical terms → category (for fallback extraction)
_CLINICAL_TERMS: dict[str, ConceptCategory] = {
    # Conditions/diagnoses
    "appendicitis": ConceptCategory.CONDITION,
    "pneumonia": ConceptCategory.CONDITION,
    "diabetes": ConceptCategory.CONDITION,
    "diabetes mellitus": ConceptCategory.CONDITION,
    "type 2 diabetes": ConceptCategory.CONDITION,
    "type 1 diabetes": ConceptCategory.CONDITION,
    "hypertension": ConceptCategory.CONDITION,
    "heart failure": ConceptCategory.CONDITION,
    "atrial fibrillation": ConceptCategory.CONDITION,
    "myocardial infarction": ConceptCategory.CONDITION,
    "stroke": ConceptCategory.CONDITION,
    "asthma": ConceptCategory.CONDITION,
    "cellulitis": ConceptCategory.CONDITION,
    "sepsis": ConceptCategory.CONDITION,
    "fracture": ConceptCategory.CONDITION,
    "abscess": ConceptCategory.CONDITION,
    "cholecystitis": ConceptCategory.CONDITION,
    "diverticulitis": ConceptCategory.CONDITION,
    "pancreatitis": ConceptCategory.CONDITION,
    "urinary tract infection": ConceptCategory.CONDITION,
    "kidney disease": ConceptCategory.CONDITION,
    # Symptoms/findings
    "pain": ConceptCategory.SYMPTOM,
    "abdominal pain": ConceptCategory.SYMPTOM,
    "chest pain": ConceptCategory.SYMPTOM,
    "headache": ConceptCategory.SYMPTOM,
    "fever": ConceptCategory.SYMPTOM,
    "nausea": ConceptCategory.SYMPTOM,
    "vomiting": ConceptCategory.SYMPTOM,
    "diarrhea": ConceptCategory.SYMPTOM,
    "cough": ConceptCategory.SYMPTOM,
    "shortness of breath": ConceptCategory.SYMPTOM,
    "dizziness": ConceptCategory.SYMPTOM,
    "fatigue": ConceptCategory.SYMPTOM,
    "swelling": ConceptCategory.SYMPTOM,
    "tenderness": ConceptCategory.SYMPTOM,
    "rebound tenderness": ConceptCategory.SYMPTOM,
    "guarding": ConceptCategory.SYMPTOM,
    "edema": ConceptCategory.SYMPTOM,
    "rash": ConceptCategory.SYMPTOM,
    "dyspnea": ConceptCategory.SYMPTOM,
    "tachycardia": ConceptCategory.SYMPTOM,
    "hypotension": ConceptCategory.SYMPTOM,
    "chills": ConceptCategory.SYMPTOM,
    "weight loss": ConceptCategory.SYMPTOM,
    "weight gain": ConceptCategory.SYMPTOM,
    "inflammation": ConceptCategory.SYMPTOM,
    # Body sites
    "abdomen": ConceptCategory.BODY_SITE,
    "appendix": ConceptCategory.BODY_SITE,
    "chest": ConceptCategory.BODY_SITE,
    "lung": ConceptCategory.BODY_SITE,
    "heart": ConceptCategory.BODY_SITE,
    "liver": ConceptCategory.BODY_SITE,
    "kidney": ConceptCategory.BODY_SITE,
    "gallbladder": ConceptCategory.BODY_SITE,
    "pancreas": ConceptCategory.BODY_SITE,
    "colon": ConceptCategory.BODY_SITE,
}

# Procedure patterns for fallback extraction
_PROCEDURE_PATTERNS: list[tuple[re.Pattern | str, str]] = [
    (r"\bappendectomy\b", "appendectomy"),
    (r"\bcholecystectomy\b", "cholecystectomy"),
    (r"\bcolonoscopy\b", "colonoscopy"),
    (r"\bendoscopy\b", "endoscopy"),
    (r"\bct\s+(?:scan\s+)?(?:of\s+)?(?:the\s+)?(\w+)", "CT scan"),
    (r"\bmri\s+(?:of\s+)?(?:the\s+)?(\w+)", "MRI"),
    (r"\bx-?ray\b", "X-ray"),
    (r"\bultrasound\b", "ultrasound"),
    (r"\bbiopsy\b", "biopsy"),
    (r"\bintubation\b", "intubation"),
    (r"\bcabg\b", "coronary artery bypass graft"),
    (r"\bangioplasty\b", "angioplasty"),
    (r"\bdialysis\b", "dialysis"),
    (r"\btransfusion\b", "transfusion"),
]


def _fallback_extract(text: str, max_concepts: int) -> list[ClinicalConcept]:
    """Regex-based clinical concept extraction when LLM is unavailable."""
    concepts: list[ClinicalConcept] = []
    text_lower = text.lower()
    seen_terms: set[str] = set()

    def _add(term: str, category: ConceptCategory, negated: bool = False, qualifier: str | None = None):
        key = term.lower().strip()
        if key and key not in seen_terms and len(concepts) < max_concepts:
            seen_terms.add(key)
            concepts.append(ClinicalConcept(term=term, category=category, negated=negated, qualifier=qualifier))

    # Extract lab values
    for match in _LAB_RE.finditer(text):
        lab_name = match.group(1)
        lab_value = match.group(2)
        unit = match.group(3) or ""
        _add(f"{lab_name} {lab_value}{unit}".strip(), ConceptCategory.LAB_VALUE)

    # Extract medications
    for med in _MEDICATION_TERMS:
        if re.search(rf"\b{re.escape(med)}\b", text_lower):
            _add(med, ConceptCategory.MEDICATION)

    # Extract abbreviation-based concepts
    for abbrev, expanded in _ABBREVIATIONS.items():
        if re.search(rf"\b{re.escape(abbrev)}\b", text_lower):
            negated = bool(re.search(rf"(?:no|denies|without)\s+{re.escape(abbrev)}", text_lower))
            _add(expanded, ConceptCategory.CONDITION, negated=negated)

    # Extract clinical terms from free text
    for term, category in _CLINICAL_TERMS.items():
        if term in text_lower:
            # Check negation context (20 chars before the term)
            idx = text_lower.index(term)
            prefix = text_lower[max(0, idx - 25):idx]
            negated = bool(_NEGATION_RE.search(prefix))

            # Check for qualifier
            qualifier = None
            for q in ("acute", "chronic", "mild", "moderate", "severe", "bilateral", "recurrent"):
                if q in prefix or (idx + len(term) < len(text_lower) and q in text_lower[idx:idx + len(term) + 15]):
                    qualifier = q
                    break

            _add(term, category, negated=negated, qualifier=qualifier)

    # Extract procedure terms from common patterns
    for pattern, proc_term in _PROCEDURE_PATTERNS:
        if re.search(pattern, text_lower):
            _add(proc_term, ConceptCategory.PROCEDURE)

    return concepts
