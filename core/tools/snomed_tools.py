"""SNOMED CT tools for LLM tool-use / MCP.

Each tool has two parts:
  1. A TOOL_DEFINITION dict (OpenAI function-calling schema) that tells
     the LLM what the tool does and what arguments it accepts.
  2. An async execute function that calls Neo4j directly and returns
     structured results the LLM can use in its answer.

These tools wrap the same Neo4j queries used by the /ontology/snomed/*
endpoints, but as direct function calls — no HTTP overhead.

Usage in the chat pipeline:
    from core.tools.snomed_tools import SNOMED_TOOLS, execute_tool

    # Pass SNOMED_TOOLS to the LLM as available tools
    # When the LLM returns a tool_call, execute it:
    result = await execute_tool(tool_name, tool_args)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.graph.neo4j_client import get_session

log = logging.getLogger("noesis.tools.snomed")


# ── Tool Definitions (OpenAI function-calling format) ─────────────────

SNOMED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "snomed_search",
            "description": (
                "Search SNOMED CT concepts by name. Use this when the user asks "
                "about a clinical concept and you need to find its SNOMED CT code "
                "(SCTID), or when you need to look up what concepts exist for a "
                "given clinical term like 'heart failure' or 'diabetes'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Clinical term to search for (e.g., 'heart failure', 'sepsis', 'diabetes mellitus')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snomed_get_concept",
            "description": (
                "Get full details for a SNOMED CT concept by its SCTID, including "
                "parent concepts (IS_A), clinical relationships (finding site, "
                "associated morphology, causative agent, etc.), child concepts, "
                "and ICD-10 cross-mappings. Use this when you already have a SCTID "
                "and need to understand the concept's clinical context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sctid": {
                        "type": "string",
                        "description": "SNOMED CT concept ID (e.g., '84114007' for heart failure)",
                    },
                },
                "required": ["sctid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snomed_get_descendants",
            "description": (
                "Get all descendant (child, grandchild, etc.) concepts of a SNOMED CT "
                "concept. Use this when the user asks 'what types of X exist?' or "
                "'what are the subtypes of X?' For example, descendants of 'heart "
                "failure' include systolic heart failure, diastolic heart failure, "
                "acute heart failure, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sctid": {
                        "type": "string",
                        "description": "SNOMED CT concept ID to get descendants for",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum hierarchy depth to traverse (1-5, default 2)",
                        "default": 2,
                    },
                },
                "required": ["sctid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snomed_get_ancestors",
            "description": (
                "Get all ancestor (parent, grandparent, etc.) concepts of a SNOMED CT "
                "concept. Use this when the user asks 'what category does X belong to?' "
                "or 'what is X a type of?' For example, ancestors of 'acute systolic "
                "heart failure' include 'heart failure', 'disorder of cardiovascular "
                "system', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sctid": {
                        "type": "string",
                        "description": "SNOMED CT concept ID to get ancestors for",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum hierarchy depth to traverse (1-5, default 2)",
                        "default": 2,
                    },
                },
                "required": ["sctid"],
            },
        },
    },
]


# ── Tool Execution Functions ──────────────────────────────────────────

async def _snomed_search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search SNOMED CT concepts using Neo4j full-text index."""
    try:
        async with get_session() as session:
            # Try full-text index first
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes(
                        'snomed_term_search', $search_term
                    ) YIELD node, score
                    RETURN node.sctid AS sctid,
                           node.fsn AS fsn,
                           node.preferred_term AS preferred_term,
                           node.semantic_tag AS semantic_tag,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    search_term=query,
                    limit=limit,
                )
                records = [r.data() async for r in result]
                if records:
                    return {"results": records, "count": len(records)}
            except Exception:
                pass

            # Fallback: case-insensitive CONTAINS
            result = await session.run(
                """
                MATCH (s:SnomedConcept)
                WHERE toLower(s.preferred_term) CONTAINS toLower($search_term)
                   OR toLower(s.fsn) CONTAINS toLower($search_term)
                RETURN s.sctid AS sctid,
                       s.fsn AS fsn,
                       s.preferred_term AS preferred_term,
                       s.semantic_tag AS semantic_tag,
                       1.0 AS score
                LIMIT $limit
                """,
                search_term=query,
                limit=limit,
            )
            records = [r.data() async for r in result]
            return {"results": records, "count": len(records)}

    except Exception as e:
        return {"error": str(e), "results": [], "count": 0}


async def _snomed_get_concept(sctid: str) -> dict[str, Any]:
    """Get full SNOMED CT concept detail from Neo4j."""
    try:
        async with get_session() as session:
            # Core concept
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})
                RETURN s.sctid AS sctid, s.fsn AS fsn,
                       s.preferred_term AS preferred_term,
                       s.semantic_tag AS semantic_tag,
                       s.definition_status AS definition_status
                """,
                sctid=sctid,
            )
            record = await result.single()
            if not record:
                return {"error": f"Concept {sctid} not found"}
            concept = record.data()

            # Parents (IS_A)
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[:IS_A]->(p:SnomedConcept)
                RETURN p.sctid AS sctid, p.preferred_term AS term, p.semantic_tag AS tag
                """,
                sctid=sctid,
            )
            concept["parents"] = [r.data() async for r in result]

            # Children
            result = await session.run(
                """
                MATCH (c:SnomedConcept)-[:IS_A]->(s:SnomedConcept {sctid: $sctid})
                RETURN c.sctid AS sctid, c.preferred_term AS term, c.semantic_tag AS tag
                LIMIT 50
                """,
                sctid=sctid,
            )
            concept["children"] = [r.data() async for r in result]

            # Relationships (non-IS_A)
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[r]->(t:SnomedConcept)
                WHERE type(r) <> 'IS_A'
                RETURN type(r) AS rel_type, t.sctid AS target_sctid,
                       t.preferred_term AS target_term,
                       r.rel_group AS rel_group
                """,
                sctid=sctid,
            )
            concept["relationships"] = [r.data() async for r in result]

            # ICD-10 mappings
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[m:MAPS_TO]->(i:ICD10Code)
                RETURN i.code AS code, m.map_group AS map_group,
                       m.priority AS priority, m.advice AS advice
                ORDER BY m.map_group, m.priority
                """,
                sctid=sctid,
            )
            concept["icd10_codes"] = [r.data() async for r in result]

            return concept

    except Exception as e:
        return {"error": str(e)}


async def _snomed_get_descendants(sctid: str, max_depth: int = 2) -> dict[str, Any]:
    """Get descendant concepts via IS_A traversal."""
    safe_depth = max(1, min(max_depth, 5))
    try:
        async with get_session() as session:
            result = await session.run(
                f"""
                MATCH path = (d:SnomedConcept)-[:IS_A*1..{safe_depth}]->(s:SnomedConcept {{sctid: $sctid}})
                RETURN d.sctid AS sctid, d.preferred_term AS term,
                       d.semantic_tag AS tag, length(path) AS distance
                ORDER BY distance, d.preferred_term
                LIMIT 200
                """,
                sctid=sctid,
            )
            descendants = [r.data() async for r in result]
            return {
                "sctid": sctid,
                "depth": safe_depth,
                "descendants": descendants,
                "count": len(descendants),
            }
    except Exception as e:
        return {"error": str(e), "descendants": [], "count": 0}


async def _snomed_get_ancestors(sctid: str, max_depth: int = 2) -> dict[str, Any]:
    """Get ancestor concepts via IS_A traversal."""
    safe_depth = max(1, min(max_depth, 5))
    try:
        async with get_session() as session:
            result = await session.run(
                f"""
                MATCH path = (s:SnomedConcept {{sctid: $sctid}})-[:IS_A*1..{safe_depth}]->(a:SnomedConcept)
                RETURN a.sctid AS sctid, a.preferred_term AS term,
                       a.semantic_tag AS tag, length(path) AS distance
                ORDER BY distance, a.preferred_term
                """,
                sctid=sctid,
            )
            ancestors = [r.data() async for r in result]
            return {
                "sctid": sctid,
                "depth": safe_depth,
                "ancestors": ancestors,
                "count": len(ancestors),
            }
    except Exception as e:
        return {"error": str(e), "ancestors": [], "count": 0}


# ── Tool Dispatch ─────────────────────────────────────────────────────

_SNOMED_REGISTRY: dict[str, Any] = {
    "snomed_search": _snomed_search,
    "snomed_get_concept": _snomed_get_concept,
    "snomed_get_descendants": _snomed_get_descendants,
    "snomed_get_ancestors": _snomed_get_ancestors,
}


async def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Execute a tool by name and return the result as a JSON string.

    Dispatches to SNOMED tools (snomed_*) or Rules tools (rules_*).
    Returns JSON so the result can be fed directly back to the LLM
    as a tool response message.
    """
    # SNOMED tools
    fn = _SNOMED_REGISTRY.get(tool_name)
    if fn:
        try:
            result = await fn(**tool_args)
            return json.dumps(result, default=str)
        except Exception as e:
            log.error("SNOMED tool execution failed (%s): %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # Rules tools
    if tool_name.startswith("rules_"):
        try:
            from core.tools.rules_tools import execute_tool as execute_rules_tool
            return await execute_rules_tool(tool_name, tool_args)
        except Exception as e:
            log.error("Rules tool execution failed (%s): %s", tool_name, e)
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def get_all_tool_definitions() -> list[dict]:
    """Return all available tool definitions for the LLM.

    Combines SNOMED tools (Phase 2) and Rules tools (Phase 3).
    Future phases add more tools here.
    """
    from core.tools.rules_tools import RULES_TOOLS
    return SNOMED_TOOLS + RULES_TOOLS
