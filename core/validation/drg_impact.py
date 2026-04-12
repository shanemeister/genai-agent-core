"""DRG Financial Impact Calculator.

For each CDI specificity gap, compute the estimated revenue impact of
improving documentation specificity. This is the data that powers the
"Est. Impact/Case" and "Total Opportunity" columns on the Dashboard.

This is a CC/MCC impact model, NOT a full MS-DRG grouper. The model
works in two modes depending on the rule type:

1. PRINCIPAL DIAGNOSIS RULES (heart failure, pneumonia, sepsis, etc.)
   - The condition IS the principal diagnosis
   - Has a named DRG triplet in cms_drg_weights (e.g., "HEART FAILURE AND SHOCK")
   - Exact impact = (best variant weight - baseline weight) × base_rate
   - Example: HF unspecified (DRG 293, 0.5490) → HF acute systolic (DRG 291, 1.3049)
     = 0.7559 × $7,200 = $5,442/case

2. COMORBIDITY RULES (malnutrition, anemia, encephalopathy, etc.)
   - The condition is a SECONDARY diagnosis that shifts whatever
     principal DRG the patient has
   - The improvement is a CC/MCC designation shift (NONE → CC, CC → MCC)
   - Typical impact uses empirical averages across all 222 CMS triplets:
     * None → CC:  avg 0.5038 weight delta = $3,627/case
     * CC → MCC:   avg 1.3188 weight delta = $9,495/case
     * None → MCC: avg 1.8111 weight delta = $13,040/case
   - Returns the typical impact with a "(typical)" label

See memory/project_drg_financial_impact.md for the full rationale.

Design principles:
  1. Never raises — failures return a DrgImpact with null fields and a reason
  2. Uses CMS FY2025 data loaded via tools/cms/load_drg_data.py
  3. Default base_rate is CMS national average ($7,200); hospital overrides later
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.db.postgres import get_pool

log = logging.getLogger("noesis.drg_impact")

# CMS national average IPPS base rate for FY2025.
# Hospitals override with their facility-specific blended rate.
DEFAULT_BASE_RATE = 7200.0

# Empirical averages from CMS FY2025 — computed across all 222 complete
# CC/MCC triplets in cms_drg_weights. Used for comorbidity-rule impact
# when the condition doesn't map to its own DRG triplet.
# Source query: see tools/cms/README.md or project_drg_financial_impact.md
EMPIRICAL_DELTAS = {
    ("NONE", "CC"): 0.5038,    # $3,627 at default base rate
    ("NONE", "MCC"): 1.8111,   # $13,040
    ("CC", "MCC"): 1.3188,     # $9,495
    # Degraded cases (documentation loss) — negative delta
    ("MCC", "CC"): -1.3188,
    ("MCC", "NONE"): -1.8111,
    ("CC", "NONE"): -0.5038,
}

# Condition name → DRG triplet_base aliases.
# CDI rules use clinical condition names ("heart failure") but CMS uses
# DRG titles ("HEART FAILURE AND SHOCK"). This map bridges the gap for
# principal-diagnosis rules.
#
# Rules NOT in this map fall back to comorbidity mode (empirical delta).
CONDITION_TO_TRIPLET: dict[str, str] = {
    "heart failure":            "HEART FAILURE AND SHOCK",
    "pneumonia":                "SIMPLE PNEUMONIA AND PLEURISY",
    "sepsis":                   "SEPTICEMIA OR SEVERE SEPSIS WITHOUT MV >96 HOURS",
    "acute kidney injury":      "RENAL FAILURE",
    "acute kidney failure":     "RENAL FAILURE",
    "chronic kidney disease":   "RENAL FAILURE",
    "respiratory failure":      "RESPIRATORY SYSTEM DIAGNOSIS WITH VENTILATOR SUPPORT",
    "cerebrovascular accident": "INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION",
    "copd":                     "CHRONIC OBSTRUCTIVE PULMONARY DISEASE",
    "atrial fibrillation":      "CARDIAC ARRHYTHMIA AND CONDUCTION DISORDERS",
    "pulmonary embolism":       "PULMONARY EMBOLISM",
    "deep vein thrombosis":     "PERIPHERAL VASCULAR DISORDERS",
    "hypertensive heart disease": "HYPERTENSIVE HEART DISEASE",
    # Psychiatric rules — use mental disorder DRG family
    "major depressive disorder": "DEPRESSIVE NEUROSES",
    # Everything else (anemia, coagulopathy, encephalopathy, malnutrition,
    # obesity, pressure injury, hyponatremia, hypokalemia, diabetes,
    # alcohol dependence, opioid dependence) → comorbidity mode
}


# ── Data model ────────────────────────────────────────────────


@dataclass
class DrgImpact:
    """Financial impact of improving documentation specificity for one gap."""
    # Clinical identifiers
    condition: str
    current_icd10: str
    current_designation: Optional[str]     # 'MCC', 'CC', or 'NONE'

    # Target (best-case improvement)
    target_icd10: Optional[str]
    target_designation: Optional[str]      # 'MCC', 'CC', or 'NONE'

    # Computed impact
    delta_weight: Optional[float]
    impact_per_case_usd: Optional[float]
    base_rate_used: float

    # DRG triplet context (only populated in principal-diagnosis mode)
    triplet_base: Optional[str] = None
    current_drg: Optional[int] = None
    target_drg: Optional[int] = None

    # Metadata
    mode: str = "unknown"                  # 'principal', 'comorbidity', or 'unknown'
    is_typical: bool = False               # True if using empirical average
    reason: str = ""                       # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "current_icd10": self.current_icd10,
            "current_designation": self.current_designation,
            "target_icd10": self.target_icd10,
            "target_designation": self.target_designation,
            "delta_weight": self.delta_weight,
            "impact_per_case_usd": self.impact_per_case_usd,
            "base_rate_used": self.base_rate_used,
            "triplet_base": self.triplet_base,
            "current_drg": self.current_drg,
            "target_drg": self.target_drg,
            "mode": self.mode,
            "is_typical": self.is_typical,
            "reason": self.reason,
        }


# ── Lookup helpers ────────────────────────────────────────────


async def get_code_designation(icd10_code: str) -> Optional[str]:
    """Return 'MCC', 'CC', or None (meaning neither)."""
    if not icd10_code:
        return None
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT designation FROM cms_cc_mcc_codes WHERE icd10_code = $1",
            icd10_code,
        )
    return row["designation"] if row else None


async def get_triplet_variant(triplet_base: str, cc_mcc_status: str) -> Optional[dict]:
    """Get a specific variant (MCC/CC/NONE) from a DRG triplet."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT drg, title, weight_capped, weight_uncapped, gmlos, alos
            FROM cms_drg_weights
            WHERE triplet_base = $1 AND cc_mcc_status = $2
            ORDER BY drg
            LIMIT 1
            """,
            triplet_base,
            cc_mcc_status,
        )
    return dict(row) if row else None


def _normalize_designation(d: Optional[str]) -> str:
    """Normalize None → 'NONE' so it's hashable for EMPIRICAL_DELTAS lookup."""
    return d if d in ("MCC", "CC") else "NONE"


# ── Principal-diagnosis mode ──────────────────────────────────


async def _compute_principal_impact(
    condition: str,
    current_icd10: str,
    potential_icd10_map: Optional[dict[str, str]],
    base_rate: float,
) -> Optional[DrgImpact]:
    """Compute exact impact for conditions with a named DRG triplet."""
    triplet_base = CONDITION_TO_TRIPLET.get(condition.lower())
    if not triplet_base:
        return None

    # Get all variants. A triplet can have any subset of these depending on
    # how CMS defines the DRG pair. Pulmonary embolism for example only has
    # NOT_MCC (DRG 176) and MCC (DRG 175) — no CC tier, no NONE tier.
    none_variant = await get_triplet_variant(triplet_base, "NONE")
    not_mcc_variant = await get_triplet_variant(triplet_base, "NOT_MCC")
    mcc_variant = await get_triplet_variant(triplet_base, "MCC")
    cc_variant = await get_triplet_variant(triplet_base, "CC")

    # Baseline: NONE (no qualifying comorbidity) → NOT_MCC (below-MCC pair)
    #           → CC (for triplets that only have MCC+CC). In that order of
    #           preference, since NONE is the "cleanest" baseline and
    #           NOT_MCC is semantically equivalent ("below the MCC tier").
    baseline = none_variant or not_mcc_variant or cc_variant
    # Target: prefer the highest-value variant. MCC > CC > nothing.
    target = mcc_variant or cc_variant

    if not baseline or not target or baseline["drg"] == target["drg"]:
        return None

    current_designation = await get_code_designation(current_icd10)
    target_designation = "MCC" if mcc_variant else "CC"

    # Pick the best target ICD-10 code from the rule's specific map
    target_code = None
    if potential_icd10_map:
        for code in potential_icd10_map.values():
            desg = await get_code_designation(code)
            if desg == target_designation:
                target_code = code
                break
        if not target_code:
            # No exact match — use the first specific code
            target_code = next(iter(potential_icd10_map.values()), None)

    delta_weight = float(target["weight_capped"]) - float(baseline["weight_capped"])
    impact = delta_weight * base_rate

    return DrgImpact(
        condition=condition,
        current_icd10=current_icd10,
        current_designation=current_designation,
        target_icd10=target_code,
        target_designation=target_designation,
        delta_weight=round(delta_weight, 4),
        impact_per_case_usd=round(impact, 2),
        base_rate_used=base_rate,
        triplet_base=triplet_base,
        current_drg=baseline["drg"],
        target_drg=target["drg"],
        mode="principal",
        is_typical=False,
        reason=(
            f"Principal diagnosis: improving from '{current_designation or 'non-CC/MCC'}' "
            f"to {target_designation} shifts DRG {baseline['drg']} → {target['drg']}, "
            f"delta {delta_weight:+.4f} = ${impact:,.2f}/case"
        ),
    )


# ── Comorbidity mode ──────────────────────────────────────────


async def _compute_comorbidity_impact(
    condition: str,
    current_icd10: str,
    potential_icd10_map: Optional[dict[str, str]],
    base_rate: float,
) -> DrgImpact:
    """Compute typical impact for comorbidity-only conditions.

    The condition doesn't have its own DRG triplet — it's a secondary
    diagnosis that shifts whatever principal diagnosis the patient has.
    Use empirical averages from CMS data.
    """
    current_designation = await get_code_designation(current_icd10)
    current_key = _normalize_designation(current_designation)

    # Find the best target designation from the specific codes
    target_code = None
    target_designation = None
    best_rank = -1
    designation_rank = {"NONE": 0, "CC": 1, "MCC": 2}

    if potential_icd10_map:
        for code in potential_icd10_map.values():
            desg = await get_code_designation(code)
            desg_key = _normalize_designation(desg)
            rank = designation_rank.get(desg_key, 0)
            if rank > best_rank:
                best_rank = rank
                target_code = code
                target_designation = desg_key

    target_key = _normalize_designation(target_designation)

    # No improvement possible (target is same or worse than current)
    if designation_rank.get(target_key, 0) <= designation_rank.get(current_key, 0):
        return DrgImpact(
            condition=condition,
            current_icd10=current_icd10,
            current_designation=current_designation,
            target_icd10=target_code,
            target_designation=target_designation,
            delta_weight=None,
            impact_per_case_usd=None,
            base_rate_used=base_rate,
            mode="comorbidity",
            is_typical=True,
            reason=(
                f"Comorbidity: no designation improvement available "
                f"({current_key} → {target_key})"
            ),
        )

    delta_weight = EMPIRICAL_DELTAS.get((current_key, target_key))
    if delta_weight is None:
        return DrgImpact(
            condition=condition,
            current_icd10=current_icd10,
            current_designation=current_designation,
            target_icd10=target_code,
            target_designation=target_designation,
            delta_weight=None,
            impact_per_case_usd=None,
            base_rate_used=base_rate,
            mode="comorbidity",
            is_typical=True,
            reason=f"Comorbidity: unrecognized designation shift {current_key} → {target_key}",
        )

    impact = delta_weight * base_rate
    return DrgImpact(
        condition=condition,
        current_icd10=current_icd10,
        current_designation=current_designation,
        target_icd10=target_code,
        target_designation=target_designation,
        delta_weight=round(delta_weight, 4),
        impact_per_case_usd=round(impact, 2),
        base_rate_used=base_rate,
        mode="comorbidity",
        is_typical=True,
        reason=(
            f"Comorbidity (typical): improving {current_key} → {target_key} "
            f"shifts principal DRG by avg {delta_weight:+.4f} = ${impact:,.2f}/case. "
            "Actual impact depends on patient's principal diagnosis."
        ),
    )


# ── Main entry point ──────────────────────────────────────────


async def compute_impact(
    condition: str,
    current_icd10: str,
    potential_icd10_map: Optional[dict[str, str]] = None,
    base_rate: float = DEFAULT_BASE_RATE,
) -> DrgImpact:
    """Compute the dollar impact of improving documentation specificity.

    Dispatches to principal-diagnosis or comorbidity mode based on whether
    the condition has a named DRG triplet.

    Args:
        condition: Clinical condition name (e.g., "heart failure")
        current_icd10: Currently-documented ICD-10 code (typically unspecified)
        potential_icd10_map: Dict of variant_name → ICD-10 code from the
                             rule's icd10_specific field. Used to pick the
                             best target code.
        base_rate: Hospital blended payment rate (defaults to CMS national avg)

    Returns:
        A DrgImpact object. Check impact_per_case_usd for the computed value,
        mode for ('principal' vs 'comorbidity'), and is_typical for whether
        the value is exact (principal) or empirical (comorbidity).
    """
    # Try principal-diagnosis mode first
    result = await _compute_principal_impact(
        condition, current_icd10, potential_icd10_map, base_rate,
    )
    if result is not None:
        return result

    # Fall back to comorbidity mode
    return await _compute_comorbidity_impact(
        condition, current_icd10, potential_icd10_map, base_rate,
    )


# ── CMI calculator ────────────────────────────────────────────


async def compute_cmi(drg_numbers: list[int]) -> Optional[float]:
    """Case-Mix Index = average capped weight across a set of DRGs.

    Args:
        drg_numbers: List of DRG numbers (one per analyzed case)

    Returns:
        Mean capped weight, or None if the list is empty or no DRGs were
        found in cms_drg_weights.
    """
    if not drg_numbers:
        return None

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT weight_capped FROM cms_drg_weights WHERE drg = ANY($1::int[])",
            drg_numbers,
        )

    if not rows:
        return None

    total = sum(float(r["weight_capped"]) for r in rows)
    return round(total / len(rows), 4)
