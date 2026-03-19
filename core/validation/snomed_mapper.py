"""Map extracted clinical concepts to SNOMED CT and RxNorm codes.

Uses embedding similarity (pgvector) to find candidate SNOMED and RxNorm
concepts, then verifies via Neo4j that the semantic tag or term type
matches the concept category.
"""

from __future__ import annotations

import json
import logging

from core.graph.neo4j_client import get_session
from core.rag.embeddings import embed_text
from core.rag.vector_store import PgVectorStore
from core.validation.clinical_models import (
    ClinicalConcept,
    ConceptCategory,
    MappedConcept,
)

log = logging.getLogger("noesis.validation")

_store = PgVectorStore()

# Category → acceptable SNOMED semantic tags
_CATEGORY_TAGS: dict[ConceptCategory, set[str]] = {
    ConceptCategory.CONDITION: {
        "disorder", "finding", "clinical finding", "morphologic abnormality",
        "situation", "event",
    },
    ConceptCategory.SYMPTOM: {
        "finding", "clinical finding", "disorder", "situation",
        "observable entity",
    },
    ConceptCategory.PROCEDURE: {
        "procedure", "regime/therapy",
    },
    ConceptCategory.MEDICATION: {
        "substance", "product", "pharmaceutical / biologic product",
        "medicinal product", "clinical drug", "medicinal product form",
    },
    ConceptCategory.BODY_SITE: {
        "body structure", "cell structure", "morphologic abnormality",
    },
    ConceptCategory.LAB_VALUE: {
        "observable entity", "procedure", "qualifier value",
    },
}

# Minimum confidence to consider a mapping valid
_MIN_CONFIDENCE = 0.65

# Common clinical abbreviations → expanded terms for better embedding
_ABBREVIATION_EXPANSIONS: dict[str, str] = {
    # Lab values
    "wbc": "white blood cell count",
    "rbc": "red blood cell count",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet count",
    "bun": "blood urea nitrogen",
    "cr": "creatinine",
    "ef": "ejection fraction",
    "inr": "international normalized ratio",
    "pt": "prothrombin time",
    "ptt": "partial thromboplastin time",
    "esr": "erythrocyte sedimentation rate",
    "crp": "c-reactive protein",
    "bnp": "brain natriuretic peptide",
    "tsh": "thyroid stimulating hormone",
    "psa": "prostate specific antigen",
    "hba1c": "hemoglobin a1c",
    "a1c": "hemoglobin a1c",
    "ast": "aspartate aminotransferase",
    "alt": "alanine aminotransferase",
    "alp": "alkaline phosphatase",
    # Cardiac anatomy
    "lad": "left anterior descending coronary artery",
    "rca": "right coronary artery",
    "lcx": "left circumflex coronary artery",
    "lv": "left ventricle",
    "rv": "right ventricle",
    "la": "left atrium",
    "ra": "right atrium",
    "lvef": "left ventricular ejection fraction",
    # Common clinical
    "rlq": "right lower quadrant of abdomen",
    "ruq": "right upper quadrant of abdomen",
    "llq": "left lower quadrant of abdomen",
    "luq": "left upper quadrant of abdomen",
}

# RxNorm TTYs that indicate a valid medication match
_RXNORM_MED_TTYS = {
    "in", "pin", "min", "bn", "scd", "sbd",
    "scdc", "sbdc", "scdf", "sbdf", "psn", "sy", "tmsy",
}


async def map_concepts_to_snomed(
    concepts: list[ClinicalConcept],
) -> list[MappedConcept]:
    """Map a list of clinical concepts to SNOMED CT codes.

    For each concept:
      1. Embed the term and search pgvector for ontology matches
      2. Verify the best match's semantic tag is compatible
      3. Fetch ICD-10 codes from Neo4j if available
      4. Compute a confidence score

    Returns:
        List of MappedConcept in same order as input.
    """
    results: list[MappedConcept] = []

    for concept in concepts:
        mapped = await _map_single_concept(concept)
        results.append(mapped)

    mapped_count = sum(1 for m in results if m.sctid or m.rxcui)
    log.info(
        "Mapped %d/%d concepts to SNOMED CT / RxNorm",
        mapped_count, len(concepts),
    )
    return results


async def _map_single_concept(concept: ClinicalConcept) -> MappedConcept:
    """Map a single clinical concept to SNOMED CT."""
    # Expand abbreviations for better embedding quality
    embed_term = _expand_abbreviations(concept.term)

    try:
        # Step 1: Embed and search ontology vectors
        query_vec = embed_text(embed_term)
        candidates = await _store.search(
            query_vec, k=5, source_types=["ontology"]
        )

        if not candidates:
            return MappedConcept(concept=concept)

        # Step 2: Score and filter candidates
        best_match = None
        best_score = 0.0
        acceptable_tags = _CATEGORY_TAGS.get(concept.category, set())

        # Normalize scores relative to the top candidate
        raw_scores = [c.get("score", 0.0) for c in candidates]
        max_raw = max(raw_scores) if raw_scores else 1.0

        for candidate in candidates:
            raw_score = candidate.get("score", 0.0)
            meta = candidate.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)

            ontology = meta.get("ontology", "snomed")
            confidence = _normalize_score(raw_score, max_raw)

            if ontology == "rxnorm":
                # RxNorm match — check TTY for medication category
                rxcui = meta.get("rxcui", "")
                tty = meta.get("tty", "").lower()
                is_med = concept.category == ConceptCategory.MEDICATION

                if is_med and tty in _RXNORM_MED_TTYS:
                    confidence = min(1.0, confidence + 0.1)
                elif is_med:
                    confidence *= 0.9  # Unknown TTY but still medication category
                else:
                    confidence *= 0.5  # RxNorm match for non-medication category

                if confidence > best_score:
                    best_score = confidence
                    best_match = {
                        "rxcui": rxcui,
                        "sctid": None,
                        "snomed_term": candidate.get("text", ""),
                        "semantic_tag": f"rxnorm:{tty}",
                        "confidence": round(confidence, 3),
                        "source_ontology": "rxnorm",
                    }
            else:
                # SNOMED match — existing logic
                semantic_tag = meta.get("semantic_tag", "").lower()
                sctid = meta.get("sctid", "")

                tag_match = semantic_tag in acceptable_tags
                if tag_match:
                    confidence = min(1.0, confidence + 0.1)
                elif acceptable_tags:
                    confidence *= 0.7

                if confidence > best_score:
                    best_score = confidence
                    best_match = {
                        "rxcui": None,
                        "sctid": sctid,
                        "snomed_term": candidate.get("text", ""),
                        "semantic_tag": semantic_tag,
                        "confidence": round(confidence, 3),
                        "source_ontology": "snomed",
                    }

        if not best_match or best_score < _MIN_CONFIDENCE:
            return MappedConcept(concept=concept)

        # Step 3: Fetch ICD-10 codes
        if best_match["source_ontology"] == "rxnorm" and best_match.get("rxcui"):
            # For RxNorm matches, resolve SNOMED cross-ref first
            sctid, icd10_codes = await _resolve_rxnorm_to_snomed(
                best_match["rxcui"]
            )
            best_match["sctid"] = sctid
        else:
            icd10_codes = await _fetch_icd10_codes(best_match["sctid"])

        return MappedConcept(
            concept=concept,
            sctid=best_match["sctid"],
            snomed_term=best_match["snomed_term"],
            semantic_tag=best_match["semantic_tag"],
            confidence=best_match["confidence"],
            icd10_codes=icd10_codes,
            rxcui=best_match.get("rxcui"),
            source_ontology=best_match.get("source_ontology", "snomed"),
        )

    except Exception as e:
        log.warning("Failed to map concept '%s': %s", concept.term, e)
        return MappedConcept(concept=concept)


async def _resolve_rxnorm_to_snomed(rxcui: str) -> tuple[str | None, list[str]]:
    """Resolve an RxNorm concept to its SNOMED cross-reference and ICD-10 codes.

    Traverses: RxNormConcept -[:MAPS_TO_SNOMED]-> SnomedConcept -[:MAPS_TO]-> ICD10Code

    Returns:
        Tuple of (sctid or None, list of ICD-10 codes)
    """
    try:
        async with get_session() as session:
            result = await session.run(
                """
                MATCH (r:RxNormConcept {rxcui: $rxcui})-[:MAPS_TO_SNOMED]->(s:SnomedConcept)
                OPTIONAL MATCH (s)-[:MAPS_TO]->(i:ICD10Code)
                RETURN s.sctid AS sctid, collect(DISTINCT i.code) AS codes
                LIMIT 1
                """,
                rxcui=rxcui,
            )
            record = await result.single()
            if record and record["sctid"]:
                codes = [c for c in record["codes"] if c][:5]
                return record["sctid"], codes
    except Exception as e:
        log.debug("RxNorm→SNOMED resolution failed for %s: %s", rxcui, e)
    return None, []


async def _fetch_icd10_codes(sctid: str) -> list[str]:
    """Fetch ICD-10 codes mapped to a SNOMED concept."""
    try:
        async with get_session() as session:
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[:MAPS_TO]->(i:ICD10Code)
                RETURN i.code AS code
                ORDER BY code
                LIMIT 5
                """,
                sctid=sctid,
            )
            records = [record.data() async for record in result]
            return [r["code"] for r in records]
    except Exception as e:
        log.debug("ICD-10 lookup failed for %s: %s", sctid, e)
        return []


def _normalize_score(raw_score: float, max_score: float) -> float:
    """Normalize pgvector inner product score to 0-1 confidence.

    pgvector <#> (negative inner product) scores are inverted by our
    PgVectorStore.search(), so higher = more similar.

    For nomic-embed-text-v1.5 with inner product, raw scores are
    typically 300-500+. We normalize relative to the top candidate:
    the best match gets ~0.85, others scale proportionally.
    """
    if max_score <= 0:
        return 0.0
    if raw_score <= 0:
        return 0.0
    # Scale so top candidate ≈ 0.85 (leaving room for tag match boost)
    ratio = raw_score / max_score
    return round(max(0.0, min(1.0, ratio * 0.85)), 3)


def _expand_abbreviations(term: str) -> str:
    """Expand clinical abbreviations in a term for better embedding quality.

    Handles patterns like:
      "WBC 15000"  → "white blood cell count 15000"
      "EF 35%"     → "ejection fraction 35%"
      "LAD"        → "left anterior descending coronary artery"
    """
    import re

    words = term.split()
    expanded = []
    for word in words:
        # Check if the word (lowered, stripped of punctuation) is an abbreviation
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean in _ABBREVIATION_EXPANSIONS:
            expanded.append(_ABBREVIATION_EXPANSIONS[clean])
        else:
            expanded.append(word)
    return " ".join(expanded)
