"""CMS Hospital Compare Benchmarking API.

Exposes peer-comparison data for the Dashboard and for future consumers
(reports, Case Reviewer agent, MCP tools). Single source of truth for
hospital-level CMI, peer-group aggregates, and revenue gap calculations.

Endpoints:
  GET  /benchmark/hospitals/search?q=...      — find hospitals by name (autocomplete)
  GET  /benchmark/hospital/{ccn}              — one hospital + national + state comparison
  GET  /benchmark/peers?state=&hospital_type= — aggregate stats for a peer group
  GET  /benchmark/info                        — dataset freshness and row counts

All CMI values use FY2025 MS-DRG v42 weights joined to per-hospital
discharge mixes from the Medicare Inpatient by Provider and Service file.

See memory/project_free_data_edges.md for the strategic rationale.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.db.postgres import get_pool
from core.validation.drg_impact import DEFAULT_BASE_RATE

log = logging.getLogger("noesis.benchmark")
router = APIRouter(prefix="/benchmark", tags=["benchmark"])


# ── Response models ──────────────────────────────────────────


class HospitalSummary(BaseModel):
    ccn: str
    name: str
    facility_type: Optional[str] = None         # 'IPPS', 'PCH', 'OTHER'
    also_known_as: Optional[list[str]] = None
    city: Optional[str] = None
    state: Optional[str] = None
    hospital_type: Optional[str] = None
    ownership: Optional[str] = None
    overall_rating: Optional[int] = None
    cmi: Optional[float] = None
    total_discharges: Optional[int] = None


class PeerGroupStats(BaseModel):
    """Aggregate CMI percentiles for a peer group."""
    group_name: str                      # e.g., "National", "Illinois", "IL Acute Care Hospitals"
    hospital_count: int
    cmi_p10: Optional[float] = None
    cmi_p25: Optional[float] = None
    cmi_median: Optional[float] = None
    cmi_p75: Optional[float] = None
    cmi_p90: Optional[float] = None
    total_discharges: Optional[int] = None


class HospitalBenchmark(BaseModel):
    """One hospital with full comparison context — the Dashboard centerpiece."""
    hospital: HospitalSummary
    national: PeerGroupStats
    state: Optional[PeerGroupStats] = None
    state_type: Optional[PeerGroupStats] = None   # same state, same hospital_type

    # Derived revenue gap calculation
    percentile_national: Optional[int] = None      # 0-100
    percentile_state: Optional[int] = None
    cmi_gap_to_state_median: Optional[float] = None
    revenue_gap_to_state_median_usd: Optional[float] = None  # gap × discharges × base rate


class BenchmarkInfo(BaseModel):
    dataset_year: int = 2023
    fiscal_year_weights: int = 2025
    hospital_count: int
    hospitals_with_cmi: int
    national_cmi_median: Optional[float] = None


# ── Helpers ──────────────────────────────────────────────────


async def _compute_peer_stats(
    conn,
    where_clause: str,
    params: list[Any],
    group_name: str,
    min_discharges: int = 100,
) -> PeerGroupStats:
    """Compute CMI percentiles for a filtered set of hospitals."""
    where_sql = f"WHERE c.cmi IS NOT NULL AND c.total_drg_discharges >= ${len(params) + 1}"
    params = params + [min_discharges]
    if where_clause:
        where_sql += f" AND ({where_clause})"

    row = await conn.fetchrow(
        f"""
        SELECT
            COUNT(*) AS cnt,
            percentile_cont(0.10) WITHIN GROUP (ORDER BY c.cmi)::float AS p10,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY c.cmi)::float AS p25,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY c.cmi)::float AS p50,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY c.cmi)::float AS p75,
            percentile_cont(0.90) WITHIN GROUP (ORDER BY c.cmi)::float AS p90,
            SUM(c.total_drg_discharges) AS total_discharges
        FROM cms_hospital_cmi c
        LEFT JOIN cms_hospitals h ON h.ccn = c.ccn
        {where_sql}
        """,
        *params,
    )

    return PeerGroupStats(
        group_name=group_name,
        hospital_count=row["cnt"] or 0,
        cmi_p10=row["p10"],
        cmi_p25=row["p25"],
        cmi_median=row["p50"],
        cmi_p75=row["p75"],
        cmi_p90=row["p90"],
        total_discharges=row["total_discharges"],
    )


async def _compute_percentile(conn, cmi: float, where_clause: str, params: list[Any]) -> int | None:
    """Compute the percentile rank of a given CMI within a filtered set."""
    where_sql = "WHERE c.cmi IS NOT NULL AND c.total_drg_discharges >= 100"
    if where_clause:
        where_sql += f" AND ({where_clause})"

    row = await conn.fetchrow(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE c.cmi <= $1) AS at_or_below,
            COUNT(*) AS total
        FROM cms_hospital_cmi c
        LEFT JOIN cms_hospitals h ON h.ccn = c.ccn
        {where_sql}
        """,
        cmi,
        *params,
    )
    if not row or row["total"] == 0:
        return None
    return round((row["at_or_below"] / row["total"]) * 100)


# ── Endpoints ────────────────────────────────────────────────


@router.get("/info", response_model=BenchmarkInfo)
async def benchmark_info():
    """Return dataset loader status — row counts and national CMI median."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        hospital_count = await conn.fetchval("SELECT COUNT(*) FROM cms_hospitals") or 0
        cmi_count = await conn.fetchval("SELECT COUNT(*) FROM cms_hospital_cmi") or 0
        national_median = await conn.fetchval("""
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY cmi)::float
            FROM cms_hospital_cmi
            WHERE cmi IS NOT NULL AND total_drg_discharges >= 100
        """)
    return BenchmarkInfo(
        hospital_count=hospital_count,
        hospitals_with_cmi=cmi_count,
        national_cmi_median=national_median,
    )


@router.get("/hospitals/search", response_model=list[HospitalSummary])
async def hospital_search(
    q: str = Query(..., min_length=2, description="Name, alias, or CCN substring"),
    state: Optional[str] = Query(None, description="Two-letter state code"),
    limit: int = Query(20, ge=1, le=100),
):
    """Find hospitals by name, alias, or CCN for the hospital picker.

    Search rules:
      - Exact CCN match → always ranks first
      - Name prefix match → second
      - Alias match → second (aliases are curated brand/common names
        like "Heart Hospital of Austin" → St David's Medical Center,
        "MD Anderson" → University of Texas MD Anderson Cancer Center)
      - Name substring match → third
      - Within each rank, sorted by total discharges descending so
        larger teaching hospitals surface above small clinics

    Returns facility_type so the UI can render PPS-Exempt Cancer
    Hospitals with a special state (no CMI, alternative metrics).
    """
    pool = await get_pool()
    search_upper = q.upper().strip()

    # Build the search predicate. We match against:
    #   1. Hospital name (case-insensitive substring)
    #   2. CCN (exact)
    #   3. any entry in also_known_as (case-insensitive substring,
    #      evaluated via jsonb_array_elements_text + UPPER)
    #
    # The alias check is a correlated subquery: for each hospital row,
    # check whether the JSONB array contains any element that matches
    # the search term. This is fast on ~5,400 rows because PostgreSQL
    # short-circuits the WHERE clause.
    where_parts = [
        """(
            UPPER(h.name) LIKE $1
            OR h.ccn = $2
            OR EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(h.also_known_as) AS alias
                WHERE UPPER(alias) LIKE $1
            )
        )"""
    ]
    params: list[Any] = [f"%{search_upper}%", search_upper]

    if state:
        where_parts.append(f"h.state = ${len(params) + 1}")
        params.append(state.upper())

    where_sql = " AND ".join(where_parts)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                h.ccn, h.name, h.facility_type,
                COALESCE(
                    (SELECT jsonb_agg(a) FROM jsonb_array_elements_text(h.also_known_as) AS a),
                    '[]'::jsonb
                ) AS also_known_as_json,
                h.city, h.state,
                h.hospital_type, h.ownership, h.overall_rating,
                c.cmi::float AS cmi,
                c.total_drg_discharges AS total_discharges,
                CASE
                    WHEN h.ccn = $2 THEN 0
                    WHEN UPPER(h.name) LIKE $2 || '%' THEN 1
                    WHEN EXISTS (
                        SELECT 1 FROM jsonb_array_elements_text(h.also_known_as) AS alias
                        WHERE UPPER(alias) LIKE $2 || '%'
                    ) THEN 1
                    ELSE 2
                END AS match_rank
            FROM cms_hospitals h
            LEFT JOIN cms_hospital_cmi c ON c.ccn = h.ccn
            WHERE {where_sql}
            ORDER BY match_rank, c.total_drg_discharges DESC NULLS LAST, h.name
            LIMIT ${len(params) + 1}
            """,
            *params,
            limit,
        )

    results = []
    for r in rows:
        d = dict(r)
        # Flatten jsonb to a list
        aliases = d.pop("also_known_as_json", None)
        if aliases and isinstance(aliases, list):
            d["also_known_as"] = aliases
        elif aliases:
            import json as _json
            try:
                d["also_known_as"] = _json.loads(aliases) if isinstance(aliases, str) else list(aliases)
            except Exception:
                d["also_known_as"] = None
        else:
            d["also_known_as"] = None
        d.pop("match_rank", None)
        results.append(HospitalSummary(**d))
    return results


@router.get("/hospital/{ccn}", response_model=HospitalBenchmark)
async def hospital_benchmark(
    ccn: str,
    base_rate: float = Query(DEFAULT_BASE_RATE, description="Hospital base payment rate"),
):
    """Return one hospital with national + state + state/type peer comparisons.

    This is the Dashboard centerpiece: shows the hospital's CMI, the
    percentile it sits at nationally and within its state, the gap to
    the state median, and the revenue gap in dollars.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        hospital_row = await conn.fetchrow(
            """
            SELECT
                h.ccn, h.name, h.facility_type,
                COALESCE(
                    (SELECT jsonb_agg(a) FROM jsonb_array_elements_text(h.also_known_as) AS a),
                    '[]'::jsonb
                ) AS also_known_as_json,
                h.city, h.state,
                h.hospital_type, h.ownership, h.overall_rating,
                c.cmi::float AS cmi,
                c.total_drg_discharges AS total_discharges
            FROM cms_hospitals h
            LEFT JOIN cms_hospital_cmi c ON c.ccn = h.ccn
            WHERE h.ccn = $1
            """,
            ccn,
        )
        if not hospital_row:
            raise HTTPException(status_code=404, detail=f"Hospital {ccn} not found")

        # Coerce JSONB aliases to plain list for the HospitalSummary model.
        # asyncpg returns JSONB as a str by default, so we parse it.
        import json as _json
        h_dict = dict(hospital_row)
        aliases_raw = h_dict.pop("also_known_as_json", None)
        if aliases_raw:
            if isinstance(aliases_raw, list):
                h_dict["also_known_as"] = aliases_raw
            elif isinstance(aliases_raw, str):
                try:
                    parsed = _json.loads(aliases_raw)
                    h_dict["also_known_as"] = parsed if isinstance(parsed, list) else None
                except (ValueError, TypeError):
                    h_dict["also_known_as"] = None
            else:
                h_dict["also_known_as"] = None
        else:
            h_dict["also_known_as"] = None
        hospital = HospitalSummary(**h_dict)

        # National peer group (all hospitals with CMI)
        national = await _compute_peer_stats(conn, "", [], "National")

        # State peer group
        state_stats = None
        if hospital.state:
            state_stats = await _compute_peer_stats(
                conn,
                "h.state = $1",
                [hospital.state],
                f"{hospital.state} (all acute care hospitals)",
            )

        # State + hospital type peer group (tightest peer comparison)
        state_type_stats = None
        if hospital.state and hospital.hospital_type:
            state_type_stats = await _compute_peer_stats(
                conn,
                "h.state = $1 AND h.hospital_type = $2",
                [hospital.state, hospital.hospital_type],
                f"{hospital.state} {hospital.hospital_type}",
            )

        # Percentile calculations
        percentile_national = None
        percentile_state = None
        if hospital.cmi is not None:
            percentile_national = await _compute_percentile(conn, hospital.cmi, "", [])
            if hospital.state:
                percentile_state = await _compute_percentile(
                    conn, hospital.cmi, "h.state = $2", [hospital.state],
                )

    # Revenue gap calculation:
    #   gap_weight = state_median_cmi - hospital_cmi (if negative, hospital is above)
    #   annual revenue gap = gap_weight × hospital discharges × base rate
    # For Medicare base rate: use DEFAULT_BASE_RATE unless overridden.
    cmi_gap = None
    revenue_gap = None
    if (
        hospital.cmi is not None
        and state_stats is not None
        and state_stats.cmi_median is not None
        and hospital.total_discharges
    ):
        cmi_gap = round(state_stats.cmi_median - hospital.cmi, 4)
        # Only positive gap represents "money left on the table"; negative
        # means the hospital is above the state median — still show the
        # number, but the narrative flips to "you're ahead of your peers".
        revenue_gap = round(cmi_gap * hospital.total_discharges * base_rate, 2)

    return HospitalBenchmark(
        hospital=hospital,
        national=national,
        state=state_stats,
        state_type=state_type_stats,
        percentile_national=percentile_national,
        percentile_state=percentile_state,
        cmi_gap_to_state_median=cmi_gap,
        revenue_gap_to_state_median_usd=revenue_gap,
    )


@router.get("/peers", response_model=PeerGroupStats)
async def peer_stats(
    state: Optional[str] = None,
    hospital_type: Optional[str] = None,
    ownership: Optional[str] = None,
    min_discharges: int = 100,
):
    """Return CMI percentiles for an arbitrary peer group filter.

    For drilling into a specific comparison — e.g., "all non-profit
    acute care hospitals in Illinois". Used by the Dashboard's peer
    group selector and by the future Benchmark tab.
    """
    pool = await get_pool()

    parts = []
    params: list[Any] = []
    label_parts = []

    if state:
        parts.append(f"h.state = ${len(params) + 1}")
        params.append(state.upper())
        label_parts.append(state.upper())
    if hospital_type:
        parts.append(f"h.hospital_type = ${len(params) + 1}")
        params.append(hospital_type)
        label_parts.append(hospital_type)
    if ownership:
        parts.append(f"h.ownership = ${len(params) + 1}")
        params.append(ownership)
        label_parts.append(ownership)

    where_clause = " AND ".join(parts)
    label = " ".join(label_parts) if label_parts else "National"

    async with pool.acquire() as conn:
        return await _compute_peer_stats(conn, where_clause, params, label, min_discharges)
