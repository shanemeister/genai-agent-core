from __future__ import annotations

import re
from typing import Optional

from core.graph.models import GraphData, GraphEdge, GraphNode
from core.graph.neo4j_client import get_session

# Allowed relationship types — Neo4j doesn't support parameterized rel types,
# so we must validate against a whitelist to prevent Cypher injection.
_VALID_REL_TYPES = frozenset({
    "RELATED_TO",
    "MENTIONS",
    "USED_CONTEXT",
    "DISCUSSES",
    "PRODUCED",
    "PROPOSED",
    "CURATED_FROM",
    "ABOUT",
    "CONTAINS",
})

_REL_TYPE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,49}$")


def _validate_rel_type(rel_type: str) -> str:
    """Validate and sanitize a relationship type to prevent Cypher injection."""
    cleaned = rel_type.strip().upper()
    if cleaned in _VALID_REL_TYPES:
        return cleaned
    if _REL_TYPE_PATTERN.match(cleaned):
        return cleaned
    raise ValueError(f"Invalid relationship type: {rel_type!r}")


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
    """Return all nodes and edges up to a limit.

    Fetches nodes first, then all edges between those nodes separately,
    so the node limit doesn't starve edge results.
    """
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    async with get_session() as session:
        # Step 1: Fetch nodes up to the limit
        result = await session.run(
            "MATCH (n) RETURN n LIMIT $limit",
            limit=limit,
        )
        async for record in result:
            n = record["n"]
            if n:
                nodes_map[n.element_id] = _node_from_record(n)

        if not nodes_map:
            return GraphData(nodes=[], edges=[])

        # Step 2: Fetch all edges between the returned nodes.
        # We return a, r, b so the async driver fully hydrates the relationship.
        node_ids = list(nodes_map.keys())
        result = await session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE elementId(a) IN $ids AND elementId(b) IN $ids
            RETURN elementId(r) AS eid, type(r) AS rtype,
                   elementId(a) AS src, elementId(b) AS tgt,
                   properties(r) AS rprops
            """,
            ids=node_ids,
        )
        async for record in result:
            eid = record["eid"]
            if eid and eid not in edges_map:
                edges_map[eid] = GraphEdge(
                    id=eid,
                    source=record["src"],
                    target=record["tgt"],
                    label=record["rtype"],
                    properties=_sanitize_props(dict(record["rprops"] or {})),
                )

    return GraphData(
        nodes=list(nodes_map.values()),
        edges=list(edges_map.values()),
    )


async def get_neighbors(node_id: str, depth: int = 1, limit: int = 25) -> GraphData:
    """Expand neighborhood around a node."""
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    # Neo4j doesn't support parameterized variable-length path ranges,
    # so we validate depth as an int and interpolate it safely.
    safe_depth = max(1, min(int(depth), 3))

    async with get_session() as session:
        result = await session.run(
            f"""
            MATCH (start) WHERE elementId(start) = $node_id
            CALL {{
                WITH start
                MATCH path = (start)-[*1..{safe_depth}]-(neighbor)
                UNWIND relationships(path) AS r
                UNWIND nodes(path) AS n
                RETURN DISTINCT n, r
                LIMIT $limit
            }}
            RETURN n, r
            """,
            node_id=node_id,
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
            WHERE toLower(n.name) CONTAINS toLower($search_term)
               OR toLower(coalesce(n.description, '')) CONTAINS toLower($search_term)
               OR toLower(coalesce(n.text, '')) CONTAINS toLower($search_term)
            WITH n LIMIT $limit
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m
            """,
            search_term=query,
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
    safe_rel_type = _validate_rel_type(rel_type)
    async with get_session() as session:
        result = await session.run(
            f"""
            MATCH (a) WHERE elementId(a) = $source_id
            MATCH (b) WHERE elementId(b) = $target_id
            MERGE (a)-[r:{safe_rel_type}]->(b)
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
    """Record a diagram and link it to the chat session that produced it.

    Also parses the Mermaid code to extract concept names and links them
    via Diagram -[:DISCUSSES]-> Concept edges.
    """
    async with get_session() as session:
        result = await session.run(
            """
            MERGE (d:Diagram {diagram_id: $diagram_id})
            SET d.prompt = left($prompt, 200),
                d.code = $diagram_code,
                d.model = $model,
                d.name = left($prompt, 60),
                d.created_at = coalesce(d.created_at, datetime()),
                d.updated_at = datetime()
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
        node = _node_from_record(record["d"])

    # Auto-extract concepts from Mermaid code and link via DISCUSSES
    try:
        parsed_nodes, _ = _parse_mermaid(diagram_code)
        async with get_session() as session:
            for label_text in parsed_nodes.values():
                concept_name = label_text.strip().lower()
                if len(concept_name) < 2:
                    continue
                await session.run(
                    """
                    MERGE (c:Concept {name: $concept_name})
                    SET c.updated_at = datetime()
                    WITH c
                    MATCH (d:Diagram {diagram_id: $diagram_id})
                    MERGE (d)-[:DISCUSSES]->(c)
                    """,
                    concept_name=concept_name,
                    diagram_id=diagram_id,
                )
    except Exception:
        pass  # Concept extraction from diagrams is best-effort

    return node


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


async def get_session_subgraph(session_id: str) -> GraphData:
    """Return a focused subgraph around a chat session.

    Includes the ChatSession node and its direct neighbors (1 hop):
    concepts it discusses, memory cards it used as context,
    diagrams it produced, and memory cards it proposed.
    """
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    async with get_session() as session:
        # Step 1: Get the session node itself
        result = await session.run(
            "MATCH (cs:ChatSession {session_id: $session_id}) RETURN cs",
            session_id=session_id,
        )
        record = await result.single()
        if not record:
            return GraphData(nodes=[], edges=[])
        cs_node = record["cs"]
        nodes_map[cs_node.element_id] = _node_from_record(cs_node)

        # Step 2: Get direct neighbors (1 hop)
        result = await session.run(
            """
            MATCH (cs:ChatSession {session_id: $session_id})-[r]-(n)
            RETURN DISTINCT n
            """,
            session_id=session_id,
        )
        async for record in result:
            n = record["n"]
            if n and n.element_id not in nodes_map:
                nodes_map[n.element_id] = _node_from_record(n)

        # Step 3: Get all edges between the collected nodes
        node_ids = list(nodes_map.keys())
        result = await session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE elementId(a) IN $ids AND elementId(b) IN $ids
            RETURN elementId(r) AS eid, type(r) AS rtype,
                   elementId(a) AS src, elementId(b) AS tgt,
                   properties(r) AS rprops
            """,
            ids=node_ids,
        )
        async for record in result:
            eid = record["eid"]
            if eid and eid not in edges_map:
                edges_map[eid] = GraphEdge(
                    id=eid,
                    source=record["src"],
                    target=record["tgt"],
                    label=record["rtype"],
                    properties=_sanitize_props(dict(record["rprops"] or {})),
                )

    return GraphData(
        nodes=list(nodes_map.values()),
        edges=list(edges_map.values()),
    )


async def get_scoped_graph(
    scope: str = "session",
    session_id: Optional[str] = None,
    node_id: Optional[str] = None,
    depth: int = 1,
    view_mode: str = "provenance",
) -> GraphData:
    """Return a scoped subgraph with predictable, bounded results.

    Scopes:
      session   — a ChatSession and its direct artifacts (concepts, cards, diagrams)
      question  — a single question node and what it touched (same as session for now)
      artifact  — lineage chain for a specific node (Diagram, MemoryCard, MindFileEntry)

    View modes:
      provenance — show where data came from (session → used_context → card → concept)
      lineage    — show how artifacts connect to each other (session → produced → diagram)
      full       — both provenance + lineage (union of above)

    Depth: 1-3 hops from the anchor node. Default 1.
    """
    safe_depth = max(1, min(int(depth), 3))
    nodes_map: dict[str, GraphNode] = {}
    edges_map: dict[str, GraphEdge] = {}

    if scope in ("session", "question"):
        if not session_id:
            return GraphData(nodes=[], edges=[])

        async with get_session() as sess:
            # Get the session node
            result = await sess.run(
                "MATCH (cs:ChatSession {session_id: $session_id}) RETURN cs",
                session_id=session_id,
            )
            record = await result.single()
            if not record:
                return GraphData(nodes=[], edges=[])
            cs_node = record["cs"]
            nodes_map[cs_node.element_id] = _node_from_record(cs_node)

            # Build relationship type filter based on view_mode
            if view_mode == "provenance":
                rel_filter = "USED_CONTEXT|MENTIONS|ABOUT|CURATED_FROM"
            elif view_mode == "lineage":
                rel_filter = "PRODUCED|PROPOSED|DISCUSSES"
            else:  # full
                rel_filter = "USED_CONTEXT|MENTIONS|ABOUT|CURATED_FROM|PRODUCED|PROPOSED|DISCUSSES"

            # Expand from the session node with depth and rel type filter
            result = await sess.run(
                f"""
                MATCH (cs:ChatSession {{session_id: $session_id}})
                CALL {{
                    WITH cs
                    MATCH path = (cs)-[:{rel_filter}*1..{safe_depth}]-(n)
                    UNWIND nodes(path) AS nd
                    RETURN DISTINCT nd
                    LIMIT 50
                }}
                RETURN nd
                """,
                session_id=session_id,
            )
            async for record in result:
                nd = record["nd"]
                if nd and nd.element_id not in nodes_map:
                    nodes_map[nd.element_id] = _node_from_record(nd)

    elif scope == "artifact":
        if not node_id:
            return GraphData(nodes=[], edges=[])

        async with get_session() as sess:
            # Get the anchor node
            result = await sess.run(
                "MATCH (n) WHERE elementId(n) = $node_id RETURN n",
                node_id=node_id,
            )
            record = await result.single()
            if not record:
                return GraphData(nodes=[], edges=[])
            anchor = record["n"]
            nodes_map[anchor.element_id] = _node_from_record(anchor)

            # Traverse lineage from this artifact
            result = await sess.run(
                f"""
                MATCH (start) WHERE elementId(start) = $node_id
                CALL {{
                    WITH start
                    MATCH path = (start)-[*1..{safe_depth}]-(connected)
                    WHERE any(label IN labels(connected)
                        WHERE label IN ['ChatSession', 'Diagram', 'MemoryCard', 'Concept', 'MindFileEntry'])
                    UNWIND nodes(path) AS nd
                    RETURN DISTINCT nd
                    LIMIT 50
                }}
                RETURN nd
                """,
                node_id=node_id,
            )
            async for record in result:
                nd = record["nd"]
                if nd and nd.element_id not in nodes_map:
                    nodes_map[nd.element_id] = _node_from_record(nd)
    else:
        return GraphData(nodes=[], edges=[])

    if not nodes_map:
        return GraphData(nodes=[], edges=[])

    # Fetch all edges between collected nodes (separate query for hydration)
    node_ids = list(nodes_map.keys())
    async with get_session() as sess:
        result = await sess.run(
            """
            MATCH (a)-[r]->(b)
            WHERE elementId(a) IN $ids AND elementId(b) IN $ids
            RETURN elementId(r) AS eid, type(r) AS rtype,
                   elementId(a) AS src, elementId(b) AS tgt,
                   properties(r) AS rprops
            """,
            ids=node_ids,
        )
        async for record in result:
            eid = record["eid"]
            if eid and eid not in edges_map:
                edges_map[eid] = GraphEdge(
                    id=eid,
                    source=record["src"],
                    target=record["tgt"],
                    label=record["rtype"],
                    properties=_sanitize_props(dict(record["rprops"] or {})),
                )

    return GraphData(
        nodes=list(nodes_map.values()),
        edges=list(edges_map.values()),
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


async def import_diagram_to_graph(
    diagram_code: str,
    source: str = "diagram_import",
    diagram_id: Optional[str] = None,
) -> GraphData:
    """Parse Mermaid flowchart syntax and create Concept nodes + RELATED_TO edges.

    Also creates a Diagram node linked to all extracted Concepts via DISCUSSES,
    providing full provenance for where each concept came from.

    Supports:
      - flowchart/graph TD/LR/BT/RL
      - Node shapes: A[Text], B{Text?}, C([Text]), D((Text)), E>Text]
      - Edges: -->, --->, -->|label|, -- label -->
    """
    import uuid as _uuid
    nodes_created: dict[str, GraphNode] = {}
    edges_created: list[GraphEdge] = []

    if not diagram_id:
        diagram_id = f"diag-{_uuid.uuid4().hex[:12]}"

    # Parse nodes and edges from Mermaid syntax
    parsed_nodes, parsed_edges = _parse_mermaid(diagram_code)

    # Create a Diagram node for provenance
    async with get_session() as session:
        # Build a summary from the first few node labels
        summary_parts = list(parsed_nodes.values())[:5]
        summary = ", ".join(p.strip() for p in summary_parts)
        if len(parsed_nodes) > 5:
            summary += f" (+{len(parsed_nodes) - 5} more)"

        result = await session.run(
            """
            MERGE (d:Diagram {diagram_id: $diagram_id})
            SET d.code = $code,
                d.name = $name,
                d.source = $source,
                d.node_count = $node_count,
                d.created_at = coalesce(d.created_at, datetime()),
                d.updated_at = datetime()
            RETURN d
            """,
            diagram_id=diagram_id,
            code=diagram_code,
            name=summary[:60],
            source=source,
            node_count=len(parsed_nodes),
        )
        record = await result.single()
        if record:
            nodes_created["__diagram__"] = _node_from_record(record["d"])

    # Create/merge Concept nodes for each Mermaid node
    async with get_session() as session:
        for node_id, label_text in parsed_nodes.items():
            result = await session.run(
                """
                MERGE (c:Concept {name: $name})
                SET c.description = coalesce($description, c.description),
                    c.source = coalesce(c.source, $source),
                    c.diagram_node_id = $diagram_node_id,
                    c.updated_at = datetime()
                RETURN c
                """,
                name=label_text.strip().lower(),
                description=label_text,
                source=source,
                diagram_node_id=node_id,
            )
            record = await result.single()
            if record:
                nodes_created[node_id] = _node_from_record(record["c"])

    # Link Diagram -[:DISCUSSES]-> each Concept
    async with get_session() as session:
        for node_id, label_text in parsed_nodes.items():
            result = await session.run(
                """
                MATCH (d:Diagram {diagram_id: $diagram_id})
                MATCH (c:Concept {name: $concept_name})
                MERGE (d)-[r:DISCUSSES]->(c)
                SET r.updated_at = datetime()
                RETURN r
                """,
                diagram_id=diagram_id,
                concept_name=label_text.strip().lower(),
            )
            record = await result.single()
            if record:
                edges_created.append(_edge_from_record(record["r"]))

    # Create RELATED_TO edges between concepts
    async with get_session() as session:
        for src_id, tgt_id, edge_label in parsed_edges:
            if src_id not in parsed_nodes or tgt_id not in parsed_nodes:
                continue
            src_name = parsed_nodes[src_id].strip().lower()
            tgt_name = parsed_nodes[tgt_id].strip().lower()

            result = await session.run(
                """
                MATCH (a:Concept {name: $src_name})
                MATCH (b:Concept {name: $tgt_name})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.label = $edge_label,
                    r.source = $source,
                    r.updated_at = datetime()
                RETURN r
                """,
                src_name=src_name,
                tgt_name=tgt_name,
                edge_label=edge_label or "",
                source=source,
            )
            record = await result.single()
            if record:
                edges_created.append(_edge_from_record(record["r"]))

    return GraphData(
        nodes=list(nodes_created.values()),
        edges=edges_created,
    )


def _parse_mermaid(code: str) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Extract nodes and edges from Mermaid flowchart syntax.

    Returns:
        (nodes_dict, edges_list)
        nodes_dict: {node_id: display_text}
        edges_list: [(source_id, target_id, edge_label)]
    """
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []

    lines = code.strip().split("\n")

    # Node patterns: A[Text], B{Text?}, C([Text]), D((Text)), E>Text]
    node_pattern = re.compile(
        r'([A-Za-z_]\w*)'           # node ID
        r'\s*'
        r'(?:'
        r'\[([^\]]*)\]'             # [rectangle text]
        r'|\{([^}]*)\}'             # {diamond text}
        r'|\(\[([^\]]*)\]\)'        # ([rounded text])
        r'|\(\(([^)]*)\)\)'         # ((circle text))
        r'|>([^\]]*)\]'             # >asymmetric text]
        r')'
    )

    # Edge pattern: A -->|label| B  or  A --> B  or  A -- text --> B
    edge_pattern = re.compile(
        r'([A-Za-z_]\w*)'           # source node ID
        r'\s*'
        r'(?:'
        r'-->\|([^|]*)\|\s*'        # -->|label|
        r'|--\s+([^->\n]+?)\s*-->\s*'  # -- text -->
        r'|---+>\s*'                # ---> (long arrow)
        r'|-->\s*'                  # --> (standard arrow)
        r'|---\s*'                  # --- (line without arrow)
        r')'
        r'([A-Za-z_]\w*)'           # target node ID
    )

    for line in lines:
        stripped = line.strip()

        # Skip header lines (flowchart TD, graph LR, etc.)
        if re.match(r'^(flowchart|graph)\s+(TD|TB|BT|LR|RL)', stripped, re.IGNORECASE):
            continue
        # Skip empty lines and comments
        if not stripped or stripped.startswith('%%'):
            continue

        # Extract all edges from this line
        for m in edge_pattern.finditer(stripped):
            src = m.group(1)
            label = m.group(2) or m.group(3) or ""
            tgt = m.group(4)
            edges.append((src, tgt, label.strip()))

        # Extract all node definitions from this line
        for m in node_pattern.finditer(stripped):
            nid = m.group(1)
            # Find which capture group matched
            text = m.group(2) or m.group(3) or m.group(4) or m.group(5) or m.group(6)
            if text is not None and nid not in nodes:
                nodes[nid] = text

    # Also add any node IDs referenced in edges but not explicitly defined
    for src, tgt, _ in edges:
        if src not in nodes:
            nodes[src] = src
        if tgt not in nodes:
            nodes[tgt] = tgt

    return nodes, edges


async def export_graph_to_mermaid(
    node_ids: list[str] | None = None,
    depth: int = 1,
    layout: str = "TD",
) -> str:
    """Export graph nodes and edges as Mermaid flowchart syntax.

    If node_ids is empty/None, exports the full graph (up to 100 nodes).
    Otherwise, exports the neighborhood around the specified nodes.
    """
    if node_ids is not None and len(node_ids) == 0:
        return f"flowchart {layout}\n    empty[No nodes selected]"
    if node_ids and depth == 0:
        # Export exactly these nodes and edges between them (no expansion)
        all_nodes: dict[str, GraphNode] = {}
        all_edges: dict[str, GraphEdge] = {}
        async with get_session() as session:
            # Fetch the nodes
            result = await session.run(
                """
                UNWIND $ids AS nid
                MATCH (n) WHERE elementId(n) = nid
                RETURN n
                """,
                ids=node_ids,
            )
            async for record in result:
                n = record["n"]
                if n:
                    all_nodes[n.element_id] = _node_from_record(n)

            # Fetch edges between these nodes
            result = await session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE elementId(a) IN $ids AND elementId(b) IN $ids
                RETURN elementId(r) AS eid, type(r) AS rtype,
                       elementId(a) AS src, elementId(b) AS tgt,
                       properties(r) AS rprops
                """,
                ids=node_ids,
            )
            async for record in result:
                eid = record["eid"]
                if eid and eid not in all_edges:
                    all_edges[eid] = GraphEdge(
                        id=eid,
                        source=record["src"],
                        target=record["tgt"],
                        label=record["rtype"],
                        properties=_sanitize_props(dict(record["rprops"] or {})),
                    )
    elif node_ids:
        # Get neighborhood around specified nodes
        all_nodes: dict[str, GraphNode] = {}
        all_edges: dict[str, GraphEdge] = {}
        for nid in node_ids:
            data = await get_neighbors(nid, depth=depth, limit=50)
            for n in data.nodes:
                all_nodes[n.id] = n
            for e in data.edges:
                all_edges[e.id] = e
    else:
        data = await get_full_graph(limit=100)
        all_nodes = {n.id: n for n in data.nodes}
        all_edges = {e.id: e for e in data.edges}

    if not all_nodes:
        return f"flowchart {layout}\n    empty[No nodes found]"

    # Build Mermaid syntax
    lines = [f"flowchart {layout}"]

    # Assign short IDs (Mermaid node IDs can't have special chars)
    id_map: dict[str, str] = {}
    counter = 0
    for eid in all_nodes:
        counter += 1
        id_map[eid] = f"N{counter}"

    # Node definitions — shape varies by label type
    for eid, node in all_nodes.items():
        mid = id_map[eid]
        name = node.name.replace('"', "'").replace("[", "(").replace("]", ")")
        if len(name) > 40:
            name = name[:37] + "..."

        if node.label == "Concept":
            lines.append(f"    {mid}[{name}]")
        elif node.label == "MemoryCard":
            lines.append(f"    {mid}([{name}])")
        elif node.label in ("Session", "ChatSession"):
            lines.append(f"    {mid}>{name}]")
        elif node.label == "Diagram":
            lines.append(f"    {mid}{{{name}}}")
        elif node.label == "MindFileEntry":
            lines.append(f"    {mid}(({name}))")
        else:
            lines.append(f"    {mid}[{name}]")

    # Edge definitions
    for edge in all_edges.values():
        src_mid = id_map.get(edge.source)
        tgt_mid = id_map.get(edge.target)
        if not src_mid or not tgt_mid:
            continue
        label = edge.label.replace("_", " ").lower()
        lines.append(f"    {src_mid} -->|{label}| {tgt_mid}")

    return "\n".join(lines)


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


async def sync_mindfile_entry(entry) -> GraphNode:
    """Upsert a MindFileEntry node and link to its source MemoryCard + Concepts.

    Creates:
      MindFileEntry -[:CURATED_FROM]-> MemoryCard
      MindFileEntry -[:ABOUT]-> Concept  (inherits concepts from the source card)
    """
    async with get_session() as session:
        result = await session.run(
            """
            MERGE (mf:MindFileEntry {entry_id: $entry_id})
            SET mf.text = $text,
                mf.category = $category,
                mf.name = left($text, 60),
                mf.note = $note,
                mf.updated_at = datetime(),
                mf.created_at = coalesce(mf.created_at, datetime())
            RETURN mf
            """,
            entry_id=entry.id,
            text=entry.text,
            category=entry.category.value if hasattr(entry.category, "value") else str(entry.category),
            note=entry.note or "",
        )
        record = await result.single()
        node = _node_from_record(record["mf"])

    # Link MindFileEntry -[:CURATED_FROM]-> MemoryCard
    async with get_session() as session:
        await session.run(
            """
            MATCH (mf:MindFileEntry {entry_id: $entry_id})
            MATCH (m:MemoryCard {card_id: $card_id})
            MERGE (mf)-[:CURATED_FROM]->(m)
            """,
            entry_id=entry.id,
            card_id=entry.source_memory_card_id,
        )

    # Inherit concepts: find all Concepts the source MemoryCard MENTIONS,
    # and link MindFileEntry -[:ABOUT]-> those same Concepts
    async with get_session() as session:
        await session.run(
            """
            MATCH (m:MemoryCard {card_id: $card_id})-[:MENTIONS]->(c:Concept)
            MATCH (mf:MindFileEntry {entry_id: $entry_id})
            MERGE (mf)-[:ABOUT]->(c)
            """,
            card_id=entry.source_memory_card_id,
            entry_id=entry.id,
        )

    return node


# ============================================================================
# Room 1: Timeline, Patterns, Cognitive Profile
# ============================================================================

async def get_concept_timeline(concept_name: str) -> list[dict]:
    """Return a chronological timeline of all artifacts related to a concept.

    Finds the Concept node by name, then traverses all connected nodes
    (MemoryCards, ChatSessions, Diagrams) and returns them sorted by date.
    """
    events: list[dict] = []

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (c:Concept {name: $name})
            OPTIONAL MATCH (m:MemoryCard)-[:MENTIONS]->(c)
            OPTIONAL MATCH (cs:ChatSession)-[:DISCUSSES]->(c)
            OPTIONAL MATCH (d:Diagram)-[:DISCUSSES]->(c)
            OPTIONAL MATCH (mf:MindFileEntry)-[:ABOUT]->(c)
            WITH c,
                 collect(DISTINCT m) AS cards,
                 collect(DISTINCT cs) AS sessions,
                 collect(DISTINCT d) AS diagrams,
                 collect(DISTINCT mf) AS mindfile_entries
            RETURN cards, sessions, diagrams, mindfile_entries
            """,
            name=concept_name.lower(),
        )
        record = await result.single()
        if not record:
            return []

        for m in record["cards"]:
            if m is None:
                continue
            events.append({
                "type": "memory_card",
                "id": m.get("card_id", m.element_id),
                "timestamp": str(m.get("created_at", "")),
                "summary": m.get("text", m.get("name", "")),
                "category": m.get("category", ""),
            })

        for cs in record["sessions"]:
            if cs is None:
                continue
            msg = cs.get("user_message", cs.get("name", ""))
            events.append({
                "type": "chat_session",
                "id": cs.get("session_id", cs.element_id),
                "timestamp": str(cs.get("created_at", "")),
                "summary": msg[:120] + ("..." if len(msg) > 120 else ""),
            })

        for d in record["diagrams"]:
            if d is None:
                continue
            events.append({
                "type": "diagram",
                "id": d.get("diagram_id", d.element_id),
                "timestamp": str(d.get("created_at", "")),
                "summary": d.get("prompt", d.get("name", "")),
            })

        for mf in record["mindfile_entries"]:
            if mf is None:
                continue
            events.append({
                "type": "mindfile_entry",
                "id": mf.get("entry_id", mf.element_id),
                "timestamp": str(mf.get("created_at", "")),
                "summary": mf.get("text", mf.get("name", "")),
                "category": mf.get("category", ""),
            })

    # Sort by timestamp (most recent first)
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events


async def get_top_concepts(limit: int = 20) -> list[dict]:
    """Return the most connected concepts ranked by total connections."""
    concepts: list[dict] = []

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (c:Concept)
            OPTIONAL MATCH (c)-[r]-()
            WITH c, count(r) AS connections
            ORDER BY connections DESC
            LIMIT $limit
            RETURN c.name AS name, c.description AS description, connections
            """,
            limit=limit,
        )
        async for record in result:
            concepts.append({
                "name": record["name"],
                "description": record["description"],
                "connections": record["connections"],
            })

    return concepts


async def get_concept_cooccurrences(limit: int = 20) -> list[dict]:
    """Find concepts that frequently appear together (co-occur in the same sessions/cards)."""
    pairs: list[dict] = []

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (c1:Concept)<-[:DISCUSSES|MENTIONS]-(artifact)-[:DISCUSSES|MENTIONS]->(c2:Concept)
            WHERE id(c1) < id(c2)
            WITH c1.name AS concept_a, c2.name AS concept_b, count(DISTINCT artifact) AS co_count
            ORDER BY co_count DESC
            LIMIT $limit
            RETURN concept_a, concept_b, co_count
            """,
            limit=limit,
        )
        async for record in result:
            pairs.append({
                "concept_a": record["concept_a"],
                "concept_b": record["concept_b"],
                "count": record["co_count"],
            })

    return pairs


async def get_category_trend() -> list[dict]:
    """Return monthly counts of memory cards by category."""
    trend: list[dict] = []

    async with get_session() as session:
        result = await session.run(
            """
            MATCH (m:MemoryCard)
            WHERE m.created_at IS NOT NULL
            WITH m.category AS category,
                 substring(toString(m.created_at), 0, 7) AS month
            RETURN month, category, count(*) AS cnt
            ORDER BY month
            """
        )
        async for record in result:
            trend.append({
                "month": record["month"],
                "category": record["category"],
                "count": record["cnt"],
            })

    return trend
