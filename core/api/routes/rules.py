"""CDI Specificity Rules API.

GET    /rules              -- List all loaded rules
GET    /rules/stats        -- Rule coverage statistics
GET    /rules/{rule_id}    -- Get a single rule by ID
POST   /rules              -- Create a new hospital rule (Layer 3)
PUT    /rules/{rule_id}    -- Update a rule (creates Layer 3 override for base rules)
DELETE /rules/{rule_id}    -- Delete a hospital rule (Layer 3 only)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.validation.specificity_rules import (
    get_rules,
    get_rule,
    save_rule,
    delete_rule,
    SpecificityRule,
    SpecificityAttribute,
)

router = APIRouter(prefix="/rules", tags=["rules"])


# ── Request model (mirrors SpecificityRule but all fields explicit) ────

class AttributeRequest(BaseModel):
    name: str
    description: str
    required: bool = True
    evidence_patterns: list[str] = Field(default_factory=list)
    snomed_descendants: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)


class RuleRequest(BaseModel):
    rule_id: str
    condition: str
    snomed_codes: list[str] = Field(default_factory=list)
    icd10_unspecified: list[str] = Field(default_factory=list)
    icd10_specific: dict[str, str] = Field(default_factory=dict)
    attributes: list[AttributeRequest] = Field(default_factory=list)
    query_template: str = ""
    priority: str = "medium"


# ── Read endpoints ────────────────────────────────────────────────────

@router.get("")
async def list_rules():
    """List all CDI specificity rules with their attributes and ICD-10 mappings."""
    rules = get_rules()
    return [r.dict() for r in rules]


@router.get("/stats")
async def rule_stats():
    """Summary statistics about the loaded rule set."""
    rules = get_rules()
    layers = {}
    priorities = {}
    for r in rules:
        layers[r.layer] = layers.get(r.layer, 0) + 1
        priorities[r.priority] = priorities.get(r.priority, 0) + 1

    return {
        "total_rules": len(rules),
        "by_layer": layers,
        "by_priority": priorities,
        "conditions": [r.condition for r in rules],
    }


@router.get("/{rule_id}")
async def get_rule_detail(rule_id: str):
    """Get a single rule by its ID."""
    rule = get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return rule.dict()


# ── Write endpoints ───────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_rule(req: RuleRequest):
    """Create a new hospital-specific rule (Layer 3).

    If a rule with this ID already exists, returns 409.
    """
    existing = get_rule(req.rule_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Rule {req.rule_id} already exists. Use PUT to update.",
        )

    rule = SpecificityRule(
        rule_id=req.rule_id,
        condition=req.condition,
        snomed_codes=req.snomed_codes,
        icd10_unspecified=req.icd10_unspecified,
        icd10_specific=req.icd10_specific,
        attributes=[SpecificityAttribute(**a.dict()) for a in req.attributes],
        query_template=req.query_template,
        priority=req.priority,
        layer=3,
    )
    saved = save_rule(rule)
    return saved.dict()


@router.put("/{rule_id}")
async def update_rule(rule_id: str, req: RuleRequest):
    """Update an existing rule.

    For base rules (Layer 1), this creates a Layer 3 override.
    For hospital rules (Layer 3), this updates in place.
    """
    if req.rule_id != rule_id:
        raise HTTPException(
            status_code=400,
            detail="rule_id in body must match URL parameter",
        )

    rule = SpecificityRule(
        rule_id=req.rule_id,
        condition=req.condition,
        snomed_codes=req.snomed_codes,
        icd10_unspecified=req.icd10_unspecified,
        icd10_specific=req.icd10_specific,
        attributes=[SpecificityAttribute(**a.dict()) for a in req.attributes],
        query_template=req.query_template,
        priority=req.priority,
        layer=3,
    )
    saved = save_rule(rule)
    return saved.dict()


@router.delete("/{rule_id}")
async def remove_rule(rule_id: str):
    """Delete a hospital-specific rule (Layer 3 only).

    Base rules (Layer 1) cannot be deleted — use PUT to override instead.
    """
    existing = get_rule(rule_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")

    if existing.layer == 1:
        raise HTTPException(
            status_code=403,
            detail="Base rules (Layer 1) cannot be deleted. Use PUT to create a Layer 3 override.",
        )

    ok = delete_rule(rule_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete rule")
    return {"deleted": rule_id}
