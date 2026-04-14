"""Load CMS Hospital Compare + Medicare Inpatient provider data into PostgreSQL.

Three source files from CMS (all free and public, updated quarterly/annually):

  1. Hospital_General_Information.csv
     https://data.cms.gov/provider-data/dataset/xubh-q36u
     One row per Medicare-registered hospital (~5,427 rows) with demographics,
     hospital type, ownership, and overall rating.

  2. MUP_INP_RY25_P04_V10_DY23_Prv.CSV
     https://data.cms.gov/data-api/v1/dataset/ee6fb1a5-39b9-46b3-a980-a7284551a732
     "Medicare Inpatient Hospitals by Provider" (2023 data year, 2025 release)
     One row per hospital with Medicare-only aggregates (discharges, payments,
     avg risk score, chronic condition percentages).

  3. MUP_INP_RY25_P03_V10_DY23_PrvSvc.CSV
     https://data.cms.gov/data-api/v1/dataset/690ddc6c-2767-4618-b277-420ffb2bf27c
     "Medicare Inpatient Hospitals by Provider and Service"
     One row per hospital × DRG (~146,000 rows). Joining this to
     cms_drg_weights lets us compute true CMI as a weighted average.

After loading, this script also populates the derived cms_hospital_cmi table
by joining cms_hospital_drg_mix to cms_drg_weights.

Usage:
    python tools/cms/load_hospital_compare.py [--force]

Default skips download if files already exist in /tmp/cms_hospital_compare.
--force re-downloads all three from CMS.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import settings
from core.db.postgres import get_pool, init_database

log = logging.getLogger("noesis.cms_hospital_loader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Data sources ──────────────────────────────────────────────

HOSPITAL_GENERAL_URL = (
    "https://data.cms.gov/provider-data/sites/default/files/resources/"
    "893c372430d9d71a1c52737d01239d47_1770163599/Hospital_General_Information.csv"
)
INPATIENT_BY_PROVIDER_URL = (
    "https://data.cms.gov/sites/default/files/2025-05/"
    "10e4b7e9-40c5-437b-b4d6-61801b6681f2/MUP_INP_RY25_P04_V10_DY23_Prv.CSV"
)
INPATIENT_BY_PROVIDER_SERVICE_URL = (
    "https://data.cms.gov/sites/default/files/2025-05/"
    "ca1c9013-8c7c-4560-a4a1-28cf7e43ccc8/MUP_INP_RY25_P03_V10_DY23_PrvSvc.CSV"
)

DOWNLOAD_DIR = Path("/tmp/cms_hospital_compare")


# ── Step 1: Download ──────────────────────────────────────────


def download_files(force: bool = False) -> None:
    """Download the three CMS files if not already cached."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    downloads = [
        (HOSPITAL_GENERAL_URL, "hospital_general.csv"),
        (INPATIENT_BY_PROVIDER_URL, "inpatient_by_provider.csv"),
        (INPATIENT_BY_PROVIDER_SERVICE_URL, "inpatient_by_prv_svc.csv"),
    ]

    for url, filename in downloads:
        path = DOWNLOAD_DIR / filename
        if path.exists() and not force:
            log.info("Already have %s (use --force to re-download)", filename)
            continue
        log.info("Downloading %s ...", filename)
        subprocess.run(["curl", "-sL", "-o", str(path), url], check=True)
        size_mb = path.stat().st_size / 1024 / 1024
        log.info("  → %s (%.1f MB)", filename, size_mb)


# ── Helpers ──────────────────────────────────────────────────


def _int(val: str | None) -> int | None:
    if val is None or val == "" or val == "Not Available":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _float(val: str | None) -> float | None:
    if val is None or val == "" or val == "Not Available":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _bool(val: str | None) -> bool | None:
    if val is None or val == "":
        return None
    v = val.strip().lower()
    if v in ("yes", "y", "true", "1"):
        return True
    if v in ("no", "n", "false", "0"):
        return False
    return None


def _str(val: str | None) -> str | None:
    if val is None:
        return None
    v = val.strip()
    return v or None


# ── Step 2: Parse Hospital General Information ───────────────


def parse_hospital_general(path: Path) -> list[dict]:
    """Parse Hospital_General_Information.csv into cms_hospitals rows."""
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ccn = _str(r.get("Facility ID"))
            if not ccn:
                continue
            rows.append({
                "ccn": ccn,
                "name": _str(r.get("Facility Name")) or "Unknown",
                "address": _str(r.get("Address")),
                "city": _str(r.get("City/Town")),
                "state": _str(r.get("State")),
                "zip": _str(r.get("ZIP Code")),
                "county": _str(r.get("County/Parish")),
                "phone": _str(r.get("Telephone Number")),
                "hospital_type": _str(r.get("Hospital Type")),
                "ownership": _str(r.get("Hospital Ownership")),
                "emergency_services": _bool(r.get("Emergency Services")),
                "birthing_friendly": _bool(r.get("Meets criteria for birthing friendly designation")),
                "overall_rating": _int(r.get("Hospital overall rating")),
                "mort_measures_count": _int(r.get("Count of Facility MORT Measures")),
                "mort_better": _int(r.get("Count of MORT Measures Better")),
                "mort_worse": _int(r.get("Count of MORT Measures Worse")),
                "safety_measures_count": _int(r.get("Count of Facility Safety Measures")),
                "safety_better": _int(r.get("Count of Safety Measures Better")),
                "safety_worse": _int(r.get("Count of Safety Measures Worse")),
                "readm_measures_count": _int(r.get("Count of Facility READM Measures")),
                "readm_better": _int(r.get("Count of READM Measures Better")),
                "readm_worse": _int(r.get("Count of READM Measures Worse")),
                "ptexp_measures_count": _int(r.get("Count of Facility Pt Exp Measures")),
                "te_measures_count": _int(r.get("Count of Facility TE Measures")),
            })
    log.info("Parsed %d hospitals from Hospital General Information", len(rows))
    return rows


# ── Step 3: Parse Inpatient by Provider (aggregate) ──────────


def parse_inpatient_by_provider(path: Path) -> list[dict]:
    """Parse the provider-level Medicare inpatient aggregate file.

    These Medicare Inpatient files use Latin-1 encoding, not UTF-8.
    """
    rows = []
    with path.open(encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ccn = _str(r.get("Rndrng_Prvdr_CCN"))
            if not ccn:
                continue
            rows.append({
                "ccn": ccn,
                "total_beneficiaries": _int(r.get("Tot_Benes")),
                "total_discharges": _int(r.get("Tot_Dschrgs")),
                "total_covered_charges": _float(r.get("Tot_Submtd_Cvrd_Chrg")),
                "total_payment": _float(r.get("Tot_Pymt_Amt")),
                "total_medicare_payment": _float(r.get("Tot_Mdcr_Pymt_Amt")),
                "total_covered_days": _int(r.get("Tot_Cvrd_Days")),
                "total_days": _int(r.get("Tot_Days")),
                "avg_beneficiary_age": _float(r.get("Bene_Avg_Age")),
                "avg_risk_score": _float(r.get("Bene_Avg_Risk_Scre")),
                "pct_heart_failure": _float(r.get("Bene_CC_PH_HF_NonIHD_V2_Pct")),
                "pct_diabetes": _float(r.get("Bene_CC_PH_Diabetes_V2_Pct")),
                "pct_ckd": _float(r.get("Bene_CC_PH_CKD_V2_Pct")),
                "pct_copd": _float(r.get("Bene_CC_PH_COPD_V2_Pct")),
                "pct_depression": _float(r.get("Bene_CC_BH_Depress_V1_Pct")),
                "pct_afib": _float(r.get("Bene_CC_PH_Afib_V2_Pct")),
                "pct_stroke": _float(r.get("Bene_CC_PH_Stroke_TIA_V2_Pct")),
            })
    log.info("Parsed %d provider-level aggregate rows", len(rows))
    return rows


# ── Step 4: Parse Inpatient by Provider and Service ──────────


def parse_inpatient_by_prv_svc(path: Path) -> list[dict]:
    """Parse the per-hospital per-DRG file (~146K rows).

    Latin-1 encoding like the other Medicare Inpatient file.
    """
    rows = []
    with path.open(encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ccn = _str(r.get("Rndrng_Prvdr_CCN"))
            drg_str = _str(r.get("DRG_Cd"))
            if not ccn or not drg_str:
                continue
            try:
                drg = int(drg_str)
            except ValueError:
                continue
            rows.append({
                "ccn": ccn,
                "drg": drg,
                "drg_description": _str(r.get("DRG_Desc")),
                "discharges": _int(r.get("Tot_Dschrgs")) or 0,
                "avg_covered_charges": _float(r.get("Avg_Submtd_Cvrd_Chrg")),
                "avg_total_payment": _float(r.get("Avg_Tot_Pymt_Amt")),
                "avg_medicare_payment": _float(r.get("Avg_Mdcr_Pymt_Amt")),
            })
    log.info("Parsed %d hospital × DRG rows", len(rows))
    return rows


# ── Step 5: Load into PostgreSQL ─────────────────────────────


async def load_hospitals(pool: asyncpg.Pool, rows: list[dict]) -> None:
    """Load cms_hospitals. Bulk upsert with ON CONFLICT."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM cms_hospitals")
        await conn.executemany(
            """
            INSERT INTO cms_hospitals (
                ccn, name, address, city, state, zip, county, phone,
                hospital_type, ownership, emergency_services, birthing_friendly,
                overall_rating, mort_measures_count, mort_better, mort_worse,
                safety_measures_count, safety_better, safety_worse,
                readm_measures_count, readm_better, readm_worse,
                ptexp_measures_count, te_measures_count
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
            )
            """,
            [
                (
                    r["ccn"], r["name"], r["address"], r["city"], r["state"],
                    r["zip"], r["county"], r["phone"],
                    r["hospital_type"], r["ownership"],
                    r["emergency_services"], r["birthing_friendly"],
                    r["overall_rating"],
                    r["mort_measures_count"], r["mort_better"], r["mort_worse"],
                    r["safety_measures_count"], r["safety_better"], r["safety_worse"],
                    r["readm_measures_count"], r["readm_better"], r["readm_worse"],
                    r["ptexp_measures_count"], r["te_measures_count"],
                )
                for r in rows
            ],
        )
    log.info("Loaded %d rows into cms_hospitals", len(rows))


async def load_inpatient_agg(pool: asyncpg.Pool, rows: list[dict]) -> int:
    """Load cms_hospital_inpatient_agg. Skip rows with unknown CCN."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM cms_hospital_inpatient_agg")

        # Filter to only CCNs that exist in cms_hospitals to satisfy FK
        known_ccns = await conn.fetch("SELECT ccn FROM cms_hospitals")
        known = {r["ccn"] for r in known_ccns}
        filtered = [r for r in rows if r["ccn"] in known]
        skipped = len(rows) - len(filtered)

        await conn.executemany(
            """
            INSERT INTO cms_hospital_inpatient_agg (
                ccn, total_beneficiaries, total_discharges,
                total_covered_charges, total_payment, total_medicare_payment,
                total_covered_days, total_days,
                avg_beneficiary_age, avg_risk_score,
                pct_heart_failure, pct_diabetes, pct_ckd, pct_copd,
                pct_depression, pct_afib, pct_stroke
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17
            )
            """,
            [
                (
                    r["ccn"], r["total_beneficiaries"], r["total_discharges"],
                    r["total_covered_charges"], r["total_payment"], r["total_medicare_payment"],
                    r["total_covered_days"], r["total_days"],
                    r["avg_beneficiary_age"], r["avg_risk_score"],
                    r["pct_heart_failure"], r["pct_diabetes"], r["pct_ckd"], r["pct_copd"],
                    r["pct_depression"], r["pct_afib"], r["pct_stroke"],
                )
                for r in filtered
            ],
        )
    log.info(
        "Loaded %d aggregate rows (%d skipped — CCN not in hospital list)",
        len(filtered), skipped,
    )
    return skipped


async def load_drg_mix(pool: asyncpg.Pool, rows: list[dict]) -> int:
    """Load cms_hospital_drg_mix. Skip rows with unknown CCN."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM cms_hospital_drg_mix")

        known_ccns = await conn.fetch("SELECT ccn FROM cms_hospitals")
        known = {r["ccn"] for r in known_ccns}
        filtered = [r for r in rows if r["ccn"] in known]
        skipped = len(rows) - len(filtered)

        # Insert in batches of 5000 to keep memory reasonable
        batch_size = 5000
        for i in range(0, len(filtered), batch_size):
            batch = filtered[i:i + batch_size]
            await conn.executemany(
                """
                INSERT INTO cms_hospital_drg_mix (
                    ccn, drg, drg_description, discharges,
                    avg_covered_charges, avg_total_payment, avg_medicare_payment
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (ccn, drg, data_year) DO UPDATE SET
                    drg_description = EXCLUDED.drg_description,
                    discharges = EXCLUDED.discharges,
                    avg_covered_charges = EXCLUDED.avg_covered_charges,
                    avg_total_payment = EXCLUDED.avg_total_payment,
                    avg_medicare_payment = EXCLUDED.avg_medicare_payment,
                    loaded_at = NOW()
                """,
                [
                    (
                        r["ccn"], r["drg"], r["drg_description"], r["discharges"],
                        r["avg_covered_charges"], r["avg_total_payment"],
                        r["avg_medicare_payment"],
                    )
                    for r in batch
                ],
            )
    log.info(
        "Loaded %d hospital × DRG rows (%d skipped — CCN not in hospital list)",
        len(filtered), skipped,
    )
    return skipped


async def compute_cmi(pool: asyncpg.Pool) -> None:
    """Compute CMI per hospital by joining drg_mix to drg_weights.

    CMI = SUM(weight × discharges) / SUM(discharges)
    using cms_drg_weights.weight_capped for the current fiscal year.
    """
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM cms_hospital_cmi")
        await conn.execute("""
            INSERT INTO cms_hospital_cmi (ccn, cmi, total_drg_discharges, drg_count)
            SELECT
                m.ccn,
                SUM(d.weight_capped * m.discharges) / NULLIF(SUM(m.discharges), 0) AS cmi,
                SUM(m.discharges) AS total_drg_discharges,
                COUNT(DISTINCT m.drg) AS drg_count
            FROM cms_hospital_drg_mix m
            JOIN cms_drg_weights d ON d.drg = m.drg AND d.fiscal_year = 2025
            GROUP BY m.ccn
        """)
        count = await conn.fetchval("SELECT COUNT(*) FROM cms_hospital_cmi")
    log.info("Computed CMI for %d hospitals", count)


# ── Verification ─────────────────────────────────────────────


async def verify_load(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        hospital_count = await conn.fetchval("SELECT COUNT(*) FROM cms_hospitals")
        agg_count = await conn.fetchval("SELECT COUNT(*) FROM cms_hospital_inpatient_agg")
        drg_count = await conn.fetchval("SELECT COUNT(*) FROM cms_hospital_drg_mix")
        cmi_count = await conn.fetchval("SELECT COUNT(*) FROM cms_hospital_cmi")

        # National median CMI
        national_cmi = await conn.fetchval("""
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY cmi)
            FROM cms_hospital_cmi
            WHERE cmi IS NOT NULL AND total_drg_discharges >= 100
        """)
        national_cmi_p25 = await conn.fetchval("""
            SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY cmi)
            FROM cms_hospital_cmi
            WHERE cmi IS NOT NULL AND total_drg_discharges >= 100
        """)
        national_cmi_p75 = await conn.fetchval("""
            SELECT percentile_cont(0.75) WITHIN GROUP (ORDER BY cmi)
            FROM cms_hospital_cmi
            WHERE cmi IS NOT NULL AND total_drg_discharges >= 100
        """)

        # Example: UChicago + Northwestern + a community hospital
        examples = await conn.fetch("""
            SELECT h.ccn, h.name, h.state, h.hospital_type,
                   c.cmi, c.total_drg_discharges,
                   a.total_medicare_payment / NULLIF(a.total_discharges, 0) AS avg_pmt_per_discharge
            FROM cms_hospitals h
            LEFT JOIN cms_hospital_cmi c ON c.ccn = h.ccn
            LEFT JOIN cms_hospital_inpatient_agg a ON a.ccn = h.ccn
            WHERE h.name ILIKE ANY (ARRAY[
                '%UNIVERSITY OF CHICAGO%',
                '%NORTHWESTERN%',
                '%MAYO%ROCHESTER%',
                '%MASSACHUSETTS GENERAL%'
            ])
            ORDER BY h.name
            LIMIT 10
        """)

    log.info("━" * 70)
    log.info("Load verification:")
    log.info("  cms_hospitals:                  %d rows", hospital_count)
    log.info("  cms_hospital_inpatient_agg:     %d rows", agg_count)
    log.info("  cms_hospital_drg_mix:           %d rows", drg_count)
    log.info("  cms_hospital_cmi (computed):    %d hospitals", cmi_count)
    log.info("━" * 70)
    log.info("National CMI (hospitals with ≥100 DRG discharges):")
    log.info("  25th percentile: %.4f", float(national_cmi_p25 or 0))
    log.info("  Median:          %.4f", float(national_cmi or 0))
    log.info("  75th percentile: %.4f", float(national_cmi_p75 or 0))
    log.info("━" * 70)
    log.info("Example major academic medical centers:")
    for r in examples:
        cmi = float(r["cmi"]) if r["cmi"] else None
        pmt = float(r["avg_pmt_per_discharge"]) if r["avg_pmt_per_discharge"] else None
        log.info(
            "  %s (%s, %s)",
            r["name"][:55], r["state"], r["hospital_type"][:30] if r["hospital_type"] else "",
        )
        log.info(
            "    CMI=%s, discharges=%s, avg_pmt=%s",
            f"{cmi:.4f}" if cmi else "n/a",
            f"{r['total_drg_discharges']:,}" if r["total_drg_discharges"] else "n/a",
            f"${pmt:,.0f}" if pmt else "n/a",
        )


# ── Main ─────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download files")
    args = parser.parse_args()

    download_files(force=args.force)

    log.info("Parsing CSV files ...")
    hospital_rows = parse_hospital_general(DOWNLOAD_DIR / "hospital_general.csv")
    agg_rows = parse_inpatient_by_provider(DOWNLOAD_DIR / "inpatient_by_provider.csv")
    drg_rows = parse_inpatient_by_prv_svc(DOWNLOAD_DIR / "inpatient_by_prv_svc.csv")

    await init_database()
    pool = await get_pool()

    log.info("Loading cms_hospitals ...")
    await load_hospitals(pool, hospital_rows)

    log.info("Loading cms_hospital_inpatient_agg ...")
    await load_inpatient_agg(pool, agg_rows)

    log.info("Loading cms_hospital_drg_mix (batched) ...")
    await load_drg_mix(pool, drg_rows)

    log.info("Computing CMI per hospital ...")
    await compute_cmi(pool)

    await verify_load(pool)


if __name__ == "__main__":
    asyncio.run(main())
