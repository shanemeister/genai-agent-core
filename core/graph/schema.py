from __future__ import annotations

from core.graph.neo4j_client import get_session

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT memorycard_id IF NOT EXISTS FOR (m:MemoryCard) REQUIRE m.card_id IS UNIQUE",
    "CREATE CONSTRAINT diagram_id IF NOT EXISTS FOR (d:Diagram) REQUIRE d.diagram_id IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT chatsession_id IF NOT EXISTS FOR (cs:ChatSession) REQUIRE cs.session_id IS UNIQUE",
]


async def ensure_schema() -> None:
    """Create constraints and indexes idempotently on startup."""
    async with get_session() as session:
        for stmt in SCHEMA_STATEMENTS:
            await session.run(stmt)
