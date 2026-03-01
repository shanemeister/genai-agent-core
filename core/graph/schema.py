from __future__ import annotations

from core.graph.neo4j_client import get_session

SCHEMA_STATEMENTS = [
    # User artifact constraints
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT memorycard_id IF NOT EXISTS FOR (m:MemoryCard) REQUIRE m.card_id IS UNIQUE",
    "CREATE CONSTRAINT diagram_id IF NOT EXISTS FOR (d:Diagram) REQUIRE d.diagram_id IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT chatsession_id IF NOT EXISTS FOR (cs:ChatSession) REQUIRE cs.session_id IS UNIQUE",
    "CREATE CONSTRAINT mindfileentry_id IF NOT EXISTS FOR (mf:MindFileEntry) REQUIRE mf.entry_id IS UNIQUE",
    # SNOMED CT ontology
    "CREATE CONSTRAINT snomed_concept_sctid IF NOT EXISTS FOR (s:SnomedConcept) REQUIRE s.sctid IS UNIQUE",
    "CREATE CONSTRAINT ontology_name IF NOT EXISTS FOR (o:Ontology) REQUIRE o.name IS UNIQUE",
    "CREATE CONSTRAINT icd10_code IF NOT EXISTS FOR (i:ICD10Code) REQUIRE i.code IS UNIQUE",
    # B-tree index for semantic tag filtering
    "CREATE INDEX snomed_semantic_tag IF NOT EXISTS FOR (s:SnomedConcept) ON (s.semantic_tag)",
]

# Full-text indexes use different syntax — run separately after constraints
FULLTEXT_STATEMENTS = [
    "CREATE FULLTEXT INDEX snomed_term_search IF NOT EXISTS FOR (s:SnomedConcept) ON EACH [s.fsn, s.preferred_term]",
]


async def ensure_schema() -> None:
    """Create constraints and indexes idempotently on startup."""
    async with get_session() as session:
        for stmt in SCHEMA_STATEMENTS:
            await session.run(stmt)
        for stmt in FULLTEXT_STATEMENTS:
            try:
                await session.run(stmt)
            except Exception:
                # Full-text index may already exist — ignore
                pass
