from __future__ import annotations

from core.validation.models import (
    ClaimEvidence,
    ClaimStatus,
    ValidatedClaim,
    ValidationResult,
)

# Thresholds for evidence classification
SUPPORT_THRESHOLD = 0.5
WEAK_THRESHOLD = 0.2


async def validate_claims(
    claims: list[str],
    user_question: str,
) -> ValidationResult:
    """Validate each claim against the knowledge base and graph.

    For each claim:
      1. Retrieve evidence from RAG (vector search + reranker)
      2. Check concept coverage in Neo4j
      3. Determine supported/unsupported status
    """
    from core.rag.retriever import retrieve_context, index_memory_cards

    validated: list[ValidatedClaim] = []

    for claim_text in claims:
        # --- Step 1: Evidence retrieval ---
        evidence_list: list[ClaimEvidence] = []
        best_score = 0.0
        try:
            raw_docs = retrieve_context(claim_text, k=3)
            for doc in raw_docs:
                score = doc.get("rerank_score", doc.get("score", 0.0))
                norm_score = _normalize_score(score, raw_docs)

                if norm_score > WEAK_THRESHOLD:
                    relationship = (
                        "supports" if norm_score > SUPPORT_THRESHOLD else "tangential"
                    )
                    evidence_list.append(
                        ClaimEvidence(
                            doc_id=doc["doc_id"],
                            text=doc["text"],
                            relevance_score=round(norm_score, 3),
                            relationship=relationship,
                        )
                    )
                    best_score = max(best_score, norm_score)
        except Exception:
            pass

        # --- Step 2: Graph concept coverage ---
        concepts_found: list[str] = []
        concepts_missing: list[str] = []
        try:
            from core.graph.concept_extractor import extract_concepts
            from core.graph import queries as gq

            claim_concepts = await extract_concepts(claim_text, max_concepts=3)
            for concept_name in claim_concepts:
                results = await gq.search_nodes(concept_name, limit=1)
                if results.nodes:
                    concepts_found.append(concept_name)
                else:
                    concepts_missing.append(concept_name)
        except Exception:
            pass

        # --- Step 3: Determine status ---
        status, confidence = _compute_status(
            evidence_list, best_score, concepts_found, concepts_missing
        )

        validated.append(
            ValidatedClaim(
                text=claim_text,
                status=status,
                confidence=round(confidence, 3),
                evidence=evidence_list,
                graph_concepts_found=concepts_found,
                graph_concepts_missing=concepts_missing,
            )
        )

    # --- Aggregate ---
    supported = sum(1 for v in validated if v.status == ClaimStatus.SUPPORTED)
    unsupported = sum(1 for v in validated if v.status == ClaimStatus.UNSUPPORTED)
    contradicted = sum(1 for v in validated if v.status == ClaimStatus.CONTRADICTED)

    summary_score = (
        sum(v.confidence for v in validated) / len(validated) if validated else 0.0
    )

    if summary_score >= 0.7:
        label = "High"
    elif summary_score >= 0.4:
        label = "Medium"
    elif summary_score > 0.1:
        label = "Low"
    else:
        label = "Ungrounded"

    total = len(validated)
    detail = f"{supported}/{total} claims supported"
    if contradicted:
        detail += f", {contradicted} contradicted"
    if unsupported:
        detail += f", {unsupported} unsupported"

    return ValidationResult(
        claims=validated,
        summary_score=round(summary_score, 3),
        supported_count=supported,
        unsupported_count=unsupported,
        contradicted_count=contradicted,
        label=label,
        detail=detail,
    )


def _normalize_score(score: float, all_docs: list[dict]) -> float:
    """Normalize a reranker/similarity score to 0-1 range."""
    if abs(score) <= 1.5:
        return max(0.0, min(1.0, score))
    # Large scores (dot products) — normalize relative to max
    max_s = (
        max(abs(d.get("rerank_score", d.get("score", 0))) for d in all_docs)
        if all_docs
        else 1.0
    )
    if max_s == 0:
        return 0.0
    return max(0.0, min(1.0, score / max_s))


def _compute_status(
    evidence: list[ClaimEvidence],
    best_score: float,
    found: list[str],
    missing: list[str],
) -> tuple[ClaimStatus, float]:
    """Determine claim status and confidence from evidence signals."""
    contradictions = [e for e in evidence if e.relationship == "contradicts"]
    if contradictions:
        return ClaimStatus.CONTRADICTED, max(
            e.relevance_score for e in contradictions
        )

    supports = [e for e in evidence if e.relationship == "supports"]

    # Strong evidence + graph concepts found
    if supports and best_score > SUPPORT_THRESHOLD:
        graph_bonus = 0.1 if found else 0.0
        confidence = min(1.0, best_score + graph_bonus)
        return ClaimStatus.SUPPORTED, confidence

    # Weak evidence but concepts exist in graph
    if found and len(found) > len(missing):
        return ClaimStatus.SUPPORTED, min(1.0, 0.4 + (0.1 * len(found)))

    # No meaningful evidence found
    if not supports and not found:
        return ClaimStatus.UNSUPPORTED, 0.1

    # Marginal case
    return ClaimStatus.UNSUPPORTED, 0.25
