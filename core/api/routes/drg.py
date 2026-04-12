"""DRG Financial Impact API.

Single source of truth for DRG impact calculations. Other services
(Noesis Gateway, MCP tools, future agents) call these endpoints rather
than reimplementing the logic.

Endpoints:
  POST /drg/impact       — Batch-compute impact for a list of specificity gaps
  POST /drg/impact/one   — Single-gap convenience endpoint
  GET  /drg/cmi          — Case-Mix Index for a set of analyses (by DRG numbers)
  GET  /drg/info         — CMS data loader status (row counts, last load time)
  GET  /drg/triplet/{name} — Look up a DRG triplet by base name

See memory/project_drg_financial_impact.md for the design rationale.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.db.postgres import get_pool
from core.validation.drg_impact import (
    DEFAULT_BASE_RATE,
    DrgImpact,
    compute_cmi,
    compute_impact,
)

log = logging.getLogger("noesis.drg")
router = APIRouter(prefix="/drg", tags=["drg"])


# ── Request/response models ───────────────────────────────────


class GapInput(BaseModel):
    """A specificity gap passed in for impact calculation."""
    condition: str = Field(..., description="Clinical condition name from the rule")
    current_icd10: str = Field("", description="Currently-documented ICD-10 code (typically unspecified)")
    potential_icd10_map: Optional[dict[str, str]] = Field(
        None,
        description="Dict of variant_name → ICD-10 from the rule's icd10_specific field",
    )


class ImpactBatchRequest(BaseModel):
    """Batch request: compute impact for many gaps in one call."""
    gaps: list[GapInput]
    base_rate: float = Field(DEFAULT_BASE_RATE, description="Hospital blended payment rate")


class ImpactResponse(BaseModel):
    """Response model for a single gap's impact — flat dict from DrgImpact."""
    condition: str
    current_icd10: str
    current_designation: Optional[str] = None
    target_icd10: Optional[str] = None
    target_designation: Optional[str] = None
    delta_weight: Optional[float] = None
    impact_per_case_usd: Optional[float] = None
    base_rate_used: float
    triplet_base: Optional[str] = None
    current_drg: Optional[int] = None
    target_drg: Optional[int] = None
    mode: str
    is_typical: bool
    reason: str


class CmiRequest(BaseModel):
    """CMI calculation request — accepts a list of DRG numbers."""
    drg_numbers: list[int]


class DrgInfo(BaseModel):
    """Status of the CMS reference data tables."""
    fiscal_year: int
    drg_row_count: int
    complete_triplet_count: int
    mcc_code_count: int
    cc_code_count: int
    total_cc_mcc_codes: int
    last_loaded_at: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────


@router.get("/info", response_model=DrgInfo)
async def drg_info():
    """Return CMS data loader status — row counts and last-load time.

    Callers use this to verify the cms_drg_weights table has been loaded
    before making impact calls. If drg_row_count is 0, the loader has
    not run yet (tools/cms/load_drg_data.py).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        drg_count = await conn.fetchval(
            "SELECT COUNT(*) FROM cms_drg_weights WHERE fiscal_year = 2025"
        ) or 0
        triplet_count = await conn.fetchval("""
            SELECT COUNT(DISTINCT triplet_base)
            FROM cms_drg_weights
            WHERE fiscal_year = 2025
              AND triplet_base IS NOT NULL
        """) or 0
        mcc_count = await conn.fetchval(
            "SELECT COUNT(*) FROM cms_cc_mcc_codes WHERE designation = 'MCC'"
        ) or 0
        cc_count = await conn.fetchval(
            "SELECT COUNT(*) FROM cms_cc_mcc_codes WHERE designation = 'CC'"
        ) or 0
        last_loaded = await conn.fetchval(
            "SELECT MAX(loaded_at) FROM cms_drg_weights WHERE fiscal_year = 2025"
        )

    return DrgInfo(
        fiscal_year=2025,
        drg_row_count=drg_count,
        complete_triplet_count=triplet_count,
        mcc_code_count=mcc_count,
        cc_code_count=cc_count,
        total_cc_mcc_codes=mcc_count + cc_count,
        last_loaded_at=last_loaded.isoformat() if last_loaded else None,
    )


@router.post("/impact", response_model=list[ImpactResponse])
async def compute_batch_impact(req: ImpactBatchRequest):
    """Compute DRG financial impact for a batch of specificity gaps.

    This is the main endpoint for Dashboard / Coding Workbench / any
    other consumer that needs per-case impact numbers. One HTTP call
    returns impacts for all gaps in the request.

    Returns a list in the same order as the input gaps. Each element
    is a full impact dict with null fields + a reason if the impact
    couldn't be computed (e.g., no CC/MCC improvement available).
    """
    results: list[ImpactResponse] = []
    for gap in req.gaps:
        try:
            impact: DrgImpact = await compute_impact(
                condition=gap.condition,
                current_icd10=gap.current_icd10,
                potential_icd10_map=gap.potential_icd10_map,
                base_rate=req.base_rate,
            )
            results.append(ImpactResponse(**impact.to_dict()))
        except Exception as e:
            log.error("Impact calc failed for %s: %s", gap.condition, e)
            results.append(ImpactResponse(
                condition=gap.condition,
                current_icd10=gap.current_icd10,
                base_rate_used=req.base_rate,
                mode="error",
                is_typical=False,
                reason=f"Calculation error: {e}",
            ))
    return results


@router.post("/impact/one", response_model=ImpactResponse)
async def compute_single_impact(gap: GapInput, base_rate: float = DEFAULT_BASE_RATE):
    """Single-gap convenience endpoint.

    For callers that only need one impact calculation at a time
    (e.g., the Coding Workbench when a user clicks on a specific gap).
    For Dashboard loads, use POST /drg/impact with a batch.
    """
    impact = await compute_impact(
        condition=gap.condition,
        current_icd10=gap.current_icd10,
        potential_icd10_map=gap.potential_icd10_map,
        base_rate=base_rate,
    )
    return ImpactResponse(**impact.to_dict())


@router.post("/cmi")
async def compute_case_mix_index(req: CmiRequest):
    """Compute CMI (Case-Mix Index) for a list of DRG numbers.

    CMI = average weight_capped across the provided DRGs.
    Returns None if no DRGs were found in cms_drg_weights.

    Typical usage:
      1. Gateway queries clinical_analyses for the working_drg column
         (once DRG assignment is wired into the analysis pipeline)
      2. Sends the list of DRG numbers to this endpoint
      3. Displays "Current CMI" on the Dashboard

    For "Projected CMI" (what CMI would be if gaps were resolved),
    the caller passes the TARGET DRGs instead — i.e., the DRG each
    case would land in if the specificity gap were resolved.
    """
    cmi = await compute_cmi(req.drg_numbers)
    return {
        "drg_count": len(req.drg_numbers),
        "drgs_found_in_cms": (await _count_drgs_found(req.drg_numbers))
                             if req.drg_numbers else 0,
        "cmi": cmi,
        "base_rate_reference": DEFAULT_BASE_RATE,
    }


async def _count_drgs_found(drg_numbers: list[int]) -> int:
    """Helper: how many of the requested DRGs exist in cms_drg_weights."""
    if not drg_numbers:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM cms_drg_weights WHERE drg = ANY($1::int[])",
            drg_numbers,
        )
    return count or 0


@router.get("/triplet/{triplet_name}")
async def get_triplet(triplet_name: str):
    """Look up a DRG triplet by base name (case-insensitive).

    Returns all variants (MCC, CC, NONE) of the triplet along with their
    weights. Useful for debugging and for the CDI Rules Browser when
    authors want to verify which triplet a rule maps to.

    Example:
        GET /drg/triplet/heart%20failure
        → returns DRGs 291, 292, 293
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT drg, title, cc_mcc_status, weight_capped, weight_uncapped,
                   gmlos, alos, mdc, drg_type
            FROM cms_drg_weights
            WHERE triplet_base ILIKE $1
              AND fiscal_year = 2025
            ORDER BY drg
            """,
            f"%{triplet_name}%",
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No DRG triplet matching '{triplet_name}'",
        )

    return {
        "query": triplet_name,
        "matches": len(rows),
        "drgs": [
            {
                "drg": r["drg"],
                "title": r["title"],
                "cc_mcc_status": r["cc_mcc_status"],
                "weight_capped": float(r["weight_capped"]),
                "weight_uncapped": float(r["weight_uncapped"]),
                "gmlos": float(r["gmlos"]),
                "alos": float(r["alos"]),
                "mdc": r["mdc"],
                "drg_type": r["drg_type"],
            }
            for r in rows
        ],
    }
