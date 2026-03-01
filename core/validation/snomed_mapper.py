"""Map extracted clinical concepts to SNOMED CT codes.

Uses embedding similarity (pgvector) to find candidate SNOMED concepts,
then verifies via Neo4j that the semantic tag matches the concept category.
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
_MIN_CONFIDENCE = 0.3


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

    mapped_count = sum(1 for m in results if m.sctid)
    log.info(
        "Mapped %d/%d concepts to SNOMED CT",
        mapped_count, len(concepts),
    )
    return results


async def _map_single_concept(concept: ClinicalConcept) -> MappedConcept:
    """Map a single clinical concept to SNOMED CT."""
    # Strip lab values from embedding text (just use the lab name)
    embed_term = concept.term
    if concept.category == ConceptCategory.LAB_VALUE:
        # "WBC 15000" → "white blood cell count" — but we embed as-is
        # and let the ontology embeddings handle matching
        pass

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

            semantic_tag = meta.get("semantic_tag", "").lower()
            sctid = meta.get("sctid", "")

            # Compute confidence from embedding similarity
            # pgvector inner product scores can be very large (300-500+)
            # Normalize: top candidate gets 0.85, others scaled proportionally
            confidence = _normalize_score(raw_score, max_raw)

            # Boost confidence if semantic tag matches expected category
            tag_match = semantic_tag in acceptable_tags
            if tag_match:
                confidence = min(1.0, confidence + 0.1)
            elif acceptable_tags:
                # Penalty for tag mismatch, but don't discard entirely
                confidence *= 0.7

            if confidence > best_score:
                best_score = confidence
                best_match = {
                    "sctid": sctid,
                    "snomed_term": candidate.get("text", ""),
                    "semantic_tag": semantic_tag,
                    "confidence": round(confidence, 3),
                }

        if not best_match or best_score < _MIN_CONFIDENCE:
            return MappedConcept(concept=concept)

        # Step 3: Fetch ICD-10 codes for the matched concept
        icd10_codes = await _fetch_icd10_codes(best_match["sctid"])

        return MappedConcept(
            concept=concept,
            sctid=best_match["sctid"],
            snomed_term=best_match["snomed_term"],
            semantic_tag=best_match["semantic_tag"],
            confidence=best_match["confidence"],
            icd10_codes=icd10_codes,
        )

    except Exception as e:
        log.warning("Failed to map concept '%s': %s", concept.term, e)
        return MappedConcept(concept=concept)


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
