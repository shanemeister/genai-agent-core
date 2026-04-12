"""CDI Specificity Rules tools for LLM tool-use / MCP.

Gives the LLM access to the CDI rules engine — the same 24+ specificity
rules that power the Dashboard and Coding Workbench. This lets Clinical
Reference answer questions like "What documentation does Noesis require
for heart failure?" using the actual rule definitions, not generic
medical knowledge.

Follows the same pattern as snomed_tools.py:
  1. TOOL_DEFINITION dicts (OpenAI function-calling schema)
  2. async execute function that calls the rules engine directly

Usage in the chat pipeline:
    from core.tools.rules_tools import RULES_TOOLS, execute_tool

    # Pass RULES_TOOLS to the LLM alongside SNOMED_TOOLS
    # When the LLM returns a tool_call, dispatch to execute_tool
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.validation.specificity_rules import get_rules, get_rule

log = logging.getLogger("noesis.tools.rules")


# ── Tool definitions (OpenAI function-calling format) ──────────────────

RULES_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rules_list",
            "description": (
                "List all CDI specificity rules loaded in Noesis. Returns each "
                "rule's ID, condition name, priority, layer (1=base, 3=hospital), "
                "and the number of required attributes. Use this to see what "
                "conditions Noesis checks for documentation specificity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "priority_filter": {
                        "type": "string",
                        "description": "Optional filter: 'high', 'medium', or 'low'",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rules_get_detail",
            "description": (
                "Get the full detail of a CDI specificity rule by its rule_id "
                "(e.g., 'HF-001', 'SEP-001', 'PNA-001'). Returns the condition name, "
                "SNOMED codes that trigger the rule, ICD-10 unspecified vs specific "
                "code mappings, required documentation attributes with evidence "
                "patterns, the CDI query template, priority, and layer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "The rule ID (e.g., 'HF-001')",
                    },
                },
                "required": ["rule_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rules_find_by_condition",
            "description": (
                "Find CDI specificity rules that match a clinical condition name. "
                "Uses case-insensitive substring matching. For example, searching "
                "'heart' would match 'heart failure' and 'hypertensive heart disease'. "
                "Returns matching rules with full detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "The condition name or partial name to search for",
                    },
                },
                "required": ["condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rules_find_by_snomed",
            "description": (
                "Find CDI specificity rules that cover a specific SNOMED CT code. "
                "Given a SCTID, returns any rules whose snomed_codes list includes "
                "that code. Use this to check if Noesis has a specificity rule for "
                "a particular clinical concept."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sctid": {
                        "type": "string",
                        "description": "The SNOMED CT concept ID to look up",
                    },
                },
                "required": ["sctid"],
            },
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────

def _rule_to_summary(rule) -> dict[str, Any]:
    """Compact summary for list views."""
    return {
        "rule_id": rule.rule_id,
        "condition": rule.condition,
        "priority": rule.priority,
        "layer": rule.layer,
        "layer_label": "Base" if rule.layer == 1 else "Hospital",
        "snomed_code_count": len(rule.snomed_codes),
        "required_attributes": [
            a.name for a in rule.attributes if a.required
        ],
        "icd10_unspecified": rule.icd10_unspecified,
    }


def _rule_to_detail(rule) -> dict[str, Any]:
    """Full detail for single-rule views."""
    return {
        "rule_id": rule.rule_id,
        "condition": rule.condition,
        "priority": rule.priority,
        "layer": rule.layer,
        "layer_label": "Base" if rule.layer == 1 else "Hospital",
        "snomed_codes": rule.snomed_codes,
        "icd10_unspecified": rule.icd10_unspecified,
        "icd10_specific": rule.icd10_specific,
        "attributes": [
            {
                "name": a.name,
                "description": a.description,
                "required": a.required,
                "possible_values": a.values,
            }
            for a in rule.attributes
        ],
        "query_template": rule.query_template,
    }


async def execute_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Execute a rules tool and return JSON string result.

    Returns a JSON string (not dict) because the chat pipeline
    inserts tool results as text into the LLM context.
    """
    try:
        if tool_name == "rules_list":
            rules = get_rules()
            priority_filter = args.get("priority_filter")
            if priority_filter:
                rules = [r for r in rules if r.priority == priority_filter]
            result = {
                "total_rules": len(rules),
                "rules": [_rule_to_summary(r) for r in rules],
            }
            return json.dumps(result, indent=2)

        elif tool_name == "rules_get_detail":
            rule_id = args.get("rule_id", "")
            rule = get_rule(rule_id)
            if not rule:
                # Try case-insensitive match
                for r in get_rules():
                    if r.rule_id.lower() == rule_id.lower():
                        rule = r
                        break
            if not rule:
                return json.dumps({
                    "error": f"Rule '{rule_id}' not found",
                    "available_ids": [r.rule_id for r in get_rules()],
                })
            return json.dumps(_rule_to_detail(rule), indent=2)

        elif tool_name == "rules_find_by_condition":
            condition = args.get("condition", "").lower()
            if not condition:
                return json.dumps({"error": "condition parameter is required"})
            matches = [
                r for r in get_rules()
                if condition in r.condition.lower()
            ]
            if not matches:
                return json.dumps({
                    "matches": 0,
                    "suggestion": "No rules match that condition. Available conditions: "
                    + ", ".join(r.condition for r in get_rules()),
                })
            return json.dumps({
                "matches": len(matches),
                "rules": [_rule_to_detail(r) for r in matches],
            }, indent=2)

        elif tool_name == "rules_find_by_snomed":
            sctid = args.get("sctid", "")
            if not sctid:
                return json.dumps({"error": "sctid parameter is required"})
            matches = [
                r for r in get_rules()
                if sctid in r.snomed_codes
            ]
            if not matches:
                return json.dumps({
                    "matches": 0,
                    "message": f"No rules cover SNOMED code {sctid}. "
                    "This concept may not have a CDI specificity rule yet.",
                })
            return json.dumps({
                "matches": len(matches),
                "rules": [_rule_to_detail(r) for r in matches],
            }, indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        log.error("Rules tool %s failed: %s", tool_name, e)
        return json.dumps({"error": str(e)})
