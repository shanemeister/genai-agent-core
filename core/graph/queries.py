from __future__ import annotations

from typing import Optional

from core.graph.models import GraphData, GraphEdge, GraphNode
from core.graph.neo4j_client import get_session


def _sanitize_props(props: dict) -> dict:
    """Convert Neo4j-specific types (DateTime, Duration, etc.) to JSON-safe values."""
    clean = {}
    for k, v in props.items():
        if hasattr(v, "iso_format"):
            clean[k] = v.iso_format()
        elif hasattr(v, "isoformat"):
            clean[k] = v.isoformat()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _node_from_record(node) -> GraphNode:
    """Convert a Neo4j node to a GraphNode."""
    labels = list(node.labels)
    label = labels[0] if labels else "Unknown"
    props = dict(node)
    name = props.pop("name", props.pop("text", props.pop("card_id", str(node.element_id))))
    return GraphNode(
        id=node.element_id,
        label=label,
        name=str(name),
        properties=_sanitize_props(props),
    )


def _edge_from_record(rel) -> GraphEdge:
    """Convert a Neo4j relationship to a GraphEdge."""
    return GraphEdge(
        id=rel.element_id,
        source=rel.start_node.element_id,
        target=rel.end_node.element_id,
        label=rel.type,
        properties=_sanitize_props(dict(rel)),
    )


async def get_full_graph(limit: int = 100) -> GraphData:
    """Return all nodes and edges up to a limit."""
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
            LIMIT $limit
            """,
            limit=limit,
        )
        async for record in result:
            n = record["n"]
            if n and n.element_id not in nodes_map:
                nodes_map[n.element_id] = _node_from_record(n)

            m = record["m"]
            if m and m.element_id not in nodes_map:
                nodes_map[m.element_id] = _node_from_record(m)

            r = record["r"]
            if r and r.element_id not in edges_map:
                edges_map[r.element_id] = _edge_from_record(r)

    return GraphData(
        nodes=list(nodes_map.values()),
        edges=list(edges_map.values()),
    )


async def get_neighbors(node_id: str, depth: int = 1, limit: int = 25) -> GraphData:
    """Expand neighborhood around a node."""
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (start) WHERE elementId(start) = $node_id
            CALL {
                WITH start
                MATCH path = (start)-[*1..$depth]-(neighbor)
                UNWIND relationships(path) AS r
                UNWIND nodes(path) AS n
                RETURN DISTINCT n, r
                LIMIT $limit
            }
            RETURN n, r
            """,
            node_id=node_id,
            depth=depth,
            limit=limit,
        )
        async for record in result:
            n = record["n"]
            if n and n.element_id not in nodes_map:
                nodes_map[n.element_id] = _node_from_record(n)

            r = record["r"]
            if r and r.element_id not in edges_map:
                edges_map[r.element_id] = _edge_from_record(r)

    return GraphData(
        nodes=list(nodes_map.values()),
        edges=list(edges_map.values()),
    )


async def search_nodes(query: str, limit: int = 20) -> GraphData:
    """Search nodes by name/text using case-insensitive contains."""
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($query)
               OR toLower(coalesce(n.description, '')) CONTAINS toLower($query)
               OR toLower(coalesce(n.text, '')) CONTAINS toLower($query)
            WITH n LIMIT $limit
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m
            """,
            query=query,
            limit=limit,
        )
        async for record in result:
            n = record["n"]
            if n and n.element_id not in nodes_map:
                nodes_map[n.element_id] = _node_from_record(n)

            m = record["m"]
            if m and m.element_id not in nodes_map:
                nodes_map[m.element_id] = _node_from_record(m)

            r = record["r"]
            if r and r.element_id not in edges_map:
                edges_map[r.element_id] = _edge_from_record(r)

    return GraphData(
        nodes=list(nodes_map.values()),
        edges=list(edges_map.values()),
    )


async def create_concept(
    name: str,
    description: Optional[str] = None,
    source: Optional[str] = None,
) -> GraphNode:
    """Create or merge a Concept node."""
    async with get_session() as session:
        result = await session.run(
            """
            MERGE (c:Concept {name: $name})
            SET c.description = coalesce($description, c.description)
            SET c.source = coalesce($source, c.source)
            SET c.updated_at = datetime()
            RETURN c
            """,
            name=name.strip().lower(),
            description=description,
            source=source,
        )
        record = await result.single()
        return _node_from_record(record["c"])


async def create_relationship(
    source_id: str,
    target_id: str,
    rel_type: str = "RELATED_TO",
    strength: float = 0.5,
) -> GraphEdge:
    """Create a relationship between two nodes by element ID."""
    async with get_session() as session:
        result = await session.run(
            f"""
            MATCH (a) WHERE elementId(a) = $source_id
            MATCH (b) WHERE elementId(b) = $target_id
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.strength = $strength
            SET r.updated_at = datetime()
            RETURN r
            """,
            source_id=source_id,
            target_id=target_id,
            strength=strength,
        )
        record = await result.single()
        if not record:
            raise ValueError("One or both nodes not found")
        return _edge_from_record(record["r"])


async def sync_memory_card(card) -> GraphNode:
    """Upsert a MemoryCard node and link to extracted concepts."""
    async with get_session() as session:
        # Upsert the memory card node
        result = await session.run(
            """
            MERGE (m:MemoryCard {card_id: $card_id})
            SET m.text = $text,
                m.category = $category,
                m.scope = $scope,
                m.approval = $approval,
                m.name = $name,
                m.updated_at = datetime()
            RETURN m
            """,
            card_id=card.id,
            text=card.text,
            category=card.category.value if hasattr(card.category, "value") else str(card.category),
            scope=card.scope.value if hasattr(card.scope, "value") else str(card.scope),
            approval=card.approval.value if hasattr(card.approval, "value") else str(card.approval),
            name=card.text[:60] + ("..." if len(card.text) > 60 else ""),
        )
        record = await result.single()
        node = _node_from_record(record["m"])

    # Try to extract and link concepts
    try:
        from core.graph.concept_extractor import extract_concepts
        concepts = await extract_concepts(card.text)
        async with get_session() as session:
            for concept_name in concepts:
                await session.run(
                    """
                    MERGE (c:Concept {name: $concept_name})
                    SET c.updated_at = datetime()
                    WITH c
                    MATCH (m:MemoryCard {card_id: $card_id})
                    MERGE (m)-[:MENTIONS]->(c)
                    """,
                    concept_name=concept_name.strip().lower(),
                    card_id=card.id,
                )
    except Exception:
        pass  # Concept extraction is best-effort

    return node


async def get_stats() -> dict:
    """Return node and edge counts."""
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (n)
            WITH count(n) AS node_count
            OPTIONAL MATCH ()-[r]->()
            RETURN node_count, count(r) AS edge_count
            """
        )
        record = await result.single()
        return {
            "node_count": record["node_count"] if record else 0,
            "edge_count": record["edge_count"] if record else 0,
        }


async def record_chat_session(
    session_id: str,
    user_message: str,
    assistant_response: str,
    model: Optional[str] = None,
    processing_time: Optional[float] = None,
    retrieved_doc_ids: Optional[list[str]] = None,
) -> GraphNode:
    """Record a chat interaction as a ChatSession node with lineage edges."""
    async with get_session() as session:
        result = await session.run(
            """
            MERGE (cs:ChatSession {session_id: $session_id})
            SET cs.user_message = $user_message,
                cs.assistant_response = left($assistant_response, 500),
                cs.model = $model,
                cs.processing_time = $processing_time,
                cs.name = left($user_message, 60),
                cs.created_at = datetime()
            RETURN cs
            """,
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            model=model,
            processing_time=processing_time,
        )
        record = await result.single()
        node = _node_from_record(record["cs"])

    # Link to retrieved memory cards used as context
    if retrieved_doc_ids:
        async with get_session() as session:
            for doc_id in retrieved_doc_ids:
                if doc_id.startswith("memory:"):
                    card_id = doc_id.replace("memory:", "")
                    await session.run(
                        """
                        MATCH (cs:ChatSession {session_id: $session_id})
                        MATCH (m:MemoryCard {card_id: $card_id})
                        MERGE (cs)-[:USED_CONTEXT]->(m)
                        """,
                        session_id=session_id,
                        card_id=card_id,
                    )

    # Extract concepts from the user message and link
    try:
        from core.graph.concept_extractor import extract_concepts
        concepts = await extract_concepts(user_message, max_concepts=3)
        async with get_session() as session:
            for concept_name in concepts:
                await session.run(
                    """
                    MERGE (c:Concept {name: $concept_name})
                    SET c.updated_at = datetime()
                    WITH c
                    MATCH (cs:ChatSession {session_id: $session_id})
                    MERGE (cs)-[:DISCUSSES]->(c)
                    """,
                    concept_name=concept_name.strip().lower(),
                    session_id=session_id,
                )
    except Exception:
        pass

    return node


async def record_diagram_lineage(
    diagram_id: str,
    source_session_id: str,
    prompt: str,
    diagram_code: str,
    model: Optional[str] = None,
) -> GraphNode:
    """Record a diagram and link it to the chat session that produced it."""
    async with get_session() as session:
        result = await session.run(
            """
            MERGE (d:Diagram {diagram_id: $diagram_id})
            SET d.prompt = left($prompt, 200),
                d.code = $diagram_code,
                d.model = $model,
                d.name = left($prompt, 60),
                d.created_at = datetime()
            WITH d
            MATCH (cs:ChatSession {session_id: $source_session_id})
            MERGE (cs)-[:PRODUCED]->(d)
            RETURN d
            """,
            diagram_id=diagram_id,
            prompt=prompt,
            diagram_code=diagram_code,
            model=model,
            source_session_id=source_session_id,
        )
        record = await result.single()
        if not record:
            raise ValueError("Source chat session not found")
        return _node_from_record(record["d"])


async def link_memory_to_session(card_id: str, session_id: str) -> None:
    """Link a proposed memory card back to the chat session that inspired it."""
    async with get_session() as session:
        await session.run(
            """
            MATCH (cs:ChatSession {session_id: $session_id})
            MATCH (m:MemoryCard {card_id: $card_id})
            MERGE (cs)-[:PROPOSED]->(m)
            """,
            session_id=session_id,
            card_id=card_id,
        )


async def get_artifact_lineage(node_id: str) -> GraphData:
    """Get the full lineage chain for an artifact (all connected artifacts)."""
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (start) WHERE elementId(start) = $node_id
            CALL {
                WITH start
                MATCH path = (start)-[*1..3]-(connected)
                WHERE any(label IN labels(connected)
                    WHERE label IN ['ChatSession', 'Diagram', 'MemoryCard', 'Concept'])
                UNWIND relationships(path) AS r
                UNWIND nodes(path) AS n
                RETURN DISTINCT n, r
                LIMIT 50
            }
            RETURN n, r
            """,
            node_id=node_id,
        )
        async for record in result:
            n = record["n"]
            if n and n.element_id not in nodes_map:
                nodes_map[n.element_id] = _node_from_record(n)
            r = record["r"]
            if r and r.element_id not in edges_map:
                edges_map[r.element_id] = _edge_from_record(r)

    return GraphData(nodes=list(nodes_map.values()), edges=list(edges_map.values()))


async def seed_demo_data() -> dict:
    """Create initial seed data for the knowledge graph."""
    async with get_session() as session:
        result = await session.run(
            """
            MERGE (c1:Concept {name: "ethical boundaries"})
              SET c1.description = "Boundaries that define acceptable system behavior"
            MERGE (c2:Concept {name: "value preservation"})
              SET c2.description = "The importance of recognizing and maintaining core values"
            MERGE (c3:Concept {name: "tool augmentation"})
              SET c3.description = "Using sharper tools while respecting human cognition limits"
            MERGE (c4:Concept {name: "human cognition"})
              SET c4.description = "The nature and limits of human mental processes"
            MERGE (c5:Concept {name: "local-first architecture"})
              SET c5.description = "Systems that prioritize local processing and data sovereignty"
            MERGE (c6:Concept {name: "knowledge graphs"})
              SET c6.description = "Graph-structured representation of interconnected knowledge"
            MERGE (c7:Concept {name: "memory governance"})
              SET c7.description = "Human-controlled approval of system memory proposals"

            MERGE (c1)-[:RELATED_TO {strength: 0.9}]->(c2)
            MERGE (c3)-[:RELATED_TO {strength: 0.8}]->(c4)
            MERGE (c5)-[:RELATED_TO {strength: 0.6}]->(c6)
            MERGE (c6)-[:RELATED_TO {strength: 0.7}]->(c7)
            MERGE (c2)-[:RELATED_TO {strength: 0.5}]->(c7)

            MERGE (s1:Session {session_id: "seed-session-001"})
              SET s1.name = "Initial seed session",
                  s1.created_at = datetime()

            RETURN count(*) AS operations
            """
        )
        record = await result.single()
        return {
            "status": "seeded",
            "operations": record["operations"] if record else 0,
        }
