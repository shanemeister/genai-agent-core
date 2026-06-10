"""noesis-rules MCP server.

Wraps the 4 CDI specificity-rule tools in core/tools/rules_tools.py
behind the MCP wire protocol. Business logic lives in execute_tool();
this file is transport only.

Run as a stdio server for an MCP client to connect:

    PYTHONPATH=. python -m core.agents.mcp_servers.rules_server

Open the MCP Inspector against this server (browser UI for poking
at tools by hand). NOTE: `mcp dev` is the documented launcher in
the MCP SDK, but it defaults to spawning `uv run --with mcp mcp run
<file>` — which doesn't have our PYTHONPATH set and can't import
`core.tools.rules_tools`. Use the inspector directly instead:

    PYTHONPATH=. npx -y @modelcontextprotocol/inspector \\
        $(which python) -m core.agents.mcp_servers.rules_server

The same pattern (explicit interpreter + module + PYTHONPATH) is
what Claude Desktop's config file expects under `command`, `args`,
and `env`, so this is also the production-shape integration recipe.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from core.tools.rules_tools import execute_tool

mcp = FastMCP("noesis-rules")


@mcp.tool()
async def rules_list(priority_filter: str | None = None) -> str:
    """List all CDI specificity rules loaded in Noesis.

    Returns each rule's ID, condition name, priority, layer
    (1=base, 3=hospital), and the number of required attributes.
    Use this to see what conditions Noesis checks for documentation
    specificity.

    Args:
        priority_filter: Optional filter — 'high', 'medium', or 'low'.
    """
    args: dict = {}
    if priority_filter is not None:
        args["priority_filter"] = priority_filter
    return await execute_tool("rules_list", args)


@mcp.tool()
async def rules_get_detail(rule_id: str) -> str:
    """Get the full detail of a CDI specificity rule by its rule_id.

    Returns the condition name, SNOMED codes that trigger the rule,
    ICD-10 unspecified vs specific code mappings, required documentation
    attributes with evidence patterns, the CDI query template, priority,
    and layer.

    Args:
        rule_id: The rule ID (e.g., 'HF-001', 'SEP-001', 'PNA-001').
    """
    return await execute_tool("rules_get_detail", {"rule_id": rule_id})


@mcp.tool()
async def rules_find_by_condition(condition: str) -> str:
    """Find CDI specificity rules that match a clinical condition name.

    Uses case-insensitive substring matching. For example, searching
    'heart' would match 'heart failure' and 'hypertensive heart disease'.
    Returns matching rules with full detail.

    Args:
        condition: The condition name or partial name to search for.
    """
    return await execute_tool("rules_find_by_condition", {"condition": condition})


@mcp.tool()
async def rules_find_by_snomed(sctid: str) -> str:
    """Find CDI specificity rules that cover a specific SNOMED CT code.

    Given a SCTID, returns any rules whose snomed_codes list includes
    that code. Use this to check if Noesis has a specificity rule for
    a particular clinical concept.

    Args:
        sctid: The SNOMED CT concept ID to look up.
    """
    return await execute_tool("rules_find_by_snomed", {"sctid": sctid})


if __name__ == "__main__":
    mcp.run()
