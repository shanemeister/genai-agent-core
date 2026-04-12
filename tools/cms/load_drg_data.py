"""Load CMS MS-DRG v42 (FY2025) data into PostgreSQL.

Downloads the FY 2025 IPPS Final Rule tables from CMS.gov and loads
them into the cms_drg_weights and cms_cc_mcc_codes tables.

Source files (free, public, from CMS.gov):
  - Table 5: MS-DRG relative weights, GMLOS, ALOS (~773 rows)
  - Table 6I: Complete MCC ICD-10 code list (~3,336 codes)
  - Table 6J: Complete CC ICD-10 code list (~14,973 codes)

The triplet_base column groups DRGs that differ only by CC/MCC status
(e.g., DRGs 291/292/293 all share triplet "HEART FAILURE AND SHOCK").
This is the key to computing specificity gap financial impact.

Usage:
    python tools/cms/load_drg_data.py [--force]

By default, skips download if files already exist in /tmp/cms_drg.
--force re-downloads from CMS.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import subprocess
import sys
from pathlib import Path

import asyncpg

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import settings
from core.db.postgres import get_pool, init_database

log = logging.getLogger("noesis.cms_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Data source URLs ──────────────────────────────────────────

CMS_BASE = "https://www.cms.gov/files/zip"
TABLE_5_URL = f"{CMS_BASE}/fy-2025-ipps-final-rule-table-5.zip"
TABLES_6_URL = f"{CMS_BASE}/fy-2025-ipps-final-rule-tables-6a-6k-and-tables-6p1a-6p4d.zip"

DOWNLOAD_DIR = Path("/tmp/cms_drg")


# ── Step 1: Download and extract ──────────────────────────────

def download_and_extract(force: bool = False) -> None:
    """Download CMS files and unzip them in-place."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    table5_zip = DOWNLOAD_DIR / "table5.zip"
    tables6_zip = DOWNLOAD_DIR / "tables6.zip"

    for url, target in [(TABLE_5_URL, table5_zip), (TABLES_6_URL, tables6_zip)]:
        if target.exists() and not force:
            log.info("Already have %s (use --force to re-download)", target.name)
            continue
        log.info("Downloading %s ...", url)
        subprocess.run(["curl", "-sL", "-o", str(target), url], check=True)
        size = target.stat().st_size
        log.info("  → %s (%.1f KB)", target.name, size / 1024)

    # Extract table 5
    log.info("Extracting Table 5 ...")
    subprocess.run(
        ["unzip", "-o", "-q", str(table5_zip), "-d", str(DOWNLOAD_DIR)],
        check=True,
    )

    # Extract tables 6 (two-level unzip: outer archive contains inner archive)
    log.info("Extracting Tables 6A-6K ...")
    subprocess.run(
        ["unzip", "-o", "-q", str(tables6_zip), "-d", str(DOWNLOAD_DIR)],
        check=True,
    )
    inner_zip = DOWNLOAD_DIR / "CMS-1808-F Tables 6A-6K v420.zip"
    tables6k_dir = DOWNLOAD_DIR / "tables6k"
    tables6k_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["unzip", "-o", "-q", str(inner_zip), "-d", str(tables6k_dir)],
        check=True,
    )


# ── Step 2: Parse Table 5 (DRG weights) ──────────────────────

# CC/MCC marker detection in DRG titles.
#
# Most titles end with one of these suffixes:
#   WITH MCC, WITH CC, WITH CC/MCC, WITHOUT CC/MCC, WITHOUT MCC, WITHOUT CC
#
# A meaningful minority (15 DRGs in FY2025) have "WITH MCC OR <something>"
# or "WITH CC OR <something>" where the OR introduces an additional
# qualifying factor (e.g., "WITH MCC OR ACUTE COR PULMONALE"). These
# still belong to the MCC/CC tier — the OR clause just gives a second
# path into that tier.
#
# So we search for the CC/MCC marker anywhere in the title, not just
# anchored to the end. The triplet_base is whatever comes BEFORE the
# WITH/WITHOUT clause.
CC_MCC_PATTERNS = [
    # Check longest/most-specific patterns first
    (re.compile(r"\bWITHOUT\s+CC/MCC\b", re.IGNORECASE), "NONE"),
    (re.compile(r"\bWITHOUT\s+MCC\b", re.IGNORECASE), "NOT_MCC"),
    (re.compile(r"\bWITHOUT\s+CC\b", re.IGNORECASE), "NOT_CC"),
    (re.compile(r"\bWITH\s+CC/MCC\b", re.IGNORECASE), "CC_OR_MCC"),
    (re.compile(r"\bWITH\s+MCC\b", re.IGNORECASE), "MCC"),
    (re.compile(r"\bWITH\s+CC\b", re.IGNORECASE), "CC"),
]


def parse_cc_mcc(title: str) -> tuple[str, str | None]:
    """Split a DRG title into (triplet_base, cc_mcc_status).

    The triplet_base is the portion of the title BEFORE the first
    WITH/WITHOUT marker. The cc_mcc_status is the designation that
    marker indicates.

    Examples:
      "HEART FAILURE AND SHOCK WITH MCC"
        → ("HEART FAILURE AND SHOCK", "MCC")
      "PULMONARY EMBOLISM WITH MCC OR ACUTE COR PULMONALE"
        → ("PULMONARY EMBOLISM", "MCC")   # OR clause stripped
      "SIMPLE PNEUMONIA AND PLEURISY WITHOUT CC/MCC"
        → ("SIMPLE PNEUMONIA AND PLEURISY", "NONE")

    Returns (title, None) if no WITH/WITHOUT marker is present.
    """
    for pattern, status in CC_MCC_PATTERNS:
        match = pattern.search(title)
        if match:
            # Keep everything before the marker; discard the marker and
            # everything after it (including any "OR ..." qualifier).
            base = title[: match.start()].strip()
            return base, status
    return title, None


def parse_table_5(path: Path) -> list[dict]:
    """Parse the FY2025 Table 5 tab-separated file.

    Columns (from CMS):
      MS-DRG, Post-Acute, Special Pay, MDC, TYPE, Title,
      Weights Before Cap, Weights 10% Cap Applied, GMLOS, ALOS
    """
    rows = []
    with path.open(encoding="latin-1") as f:
        # Skip the three header lines (two title lines + column header)
        for _ in range(3):
            next(f)
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith('"'):
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                continue
            try:
                drg = int(parts[0].strip())
            except ValueError:
                continue
            title = parts[5].strip()
            triplet_base, cc_mcc = parse_cc_mcc(title)
            rows.append({
                "drg": drg,
                "post_acute": parts[1].strip().lower() == "yes",
                "special_pay": parts[2].strip().lower() == "yes",
                "mdc": parts[3].strip() or None,
                "drg_type": parts[4].strip() or None,
                "title": title,
                "triplet_base": triplet_base,
                "cc_mcc_status": cc_mcc,
                "weight_uncapped": float(parts[6].strip() or 0),
                "weight_capped": float(parts[7].strip() or 0),
                "gmlos": float(parts[8].strip() or 0),
                "alos": float(parts[9].strip() or 0),
            })
    log.info("Parsed %d DRGs from Table 5", len(rows))
    return rows


# ── Step 3: Parse Tables 6I and 6J (CC/MCC lists) ───────────

def parse_cc_mcc_table(path: Path, designation: str) -> list[dict]:
    """Parse a CC or MCC list (Table 6I or 6J).

    Format:
      Line 1: "TABLE 6I - COMPLETE MCC LIST" (or 6J)
      Line 2: "Diagnosis Code\tDescription"
      Remaining: ICD-10 code, tab, description
    """
    rows = []
    with path.open(encoding="latin-1") as f:
        next(f)  # skip title
        next(f)  # skip column header
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            code = parts[0].strip()
            description = parts[1].strip()
            if not code:
                continue
            rows.append({
                "icd10_code": code,
                "description": description,
                "designation": designation,
            })
    log.info("Parsed %d %s codes from %s", len(rows), designation, path.name)
    return rows


# ── Step 4: Load into PostgreSQL ──────────────────────────────

async def load_drg_weights(pool: asyncpg.Pool, rows: list[dict]) -> None:
    async with pool.acquire() as conn:
        # Clear existing FY2025 data so we can re-run cleanly
        await conn.execute("DELETE FROM cms_drg_weights WHERE fiscal_year = 2025")

        # Bulk insert
        await conn.executemany(
            """
            INSERT INTO cms_drg_weights
                (drg, mdc, drg_type, title, triplet_base, cc_mcc_status,
                 weight_uncapped, weight_capped, gmlos, alos,
                 post_acute, special_pay, fiscal_year)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 2025)
            ON CONFLICT (drg) DO UPDATE SET
                mdc = EXCLUDED.mdc,
                drg_type = EXCLUDED.drg_type,
                title = EXCLUDED.title,
                triplet_base = EXCLUDED.triplet_base,
                cc_mcc_status = EXCLUDED.cc_mcc_status,
                weight_uncapped = EXCLUDED.weight_uncapped,
                weight_capped = EXCLUDED.weight_capped,
                gmlos = EXCLUDED.gmlos,
                alos = EXCLUDED.alos,
                post_acute = EXCLUDED.post_acute,
                special_pay = EXCLUDED.special_pay,
                loaded_at = NOW()
            """,
            [
                (
                    r["drg"], r["mdc"], r["drg_type"], r["title"],
                    r["triplet_base"], r["cc_mcc_status"],
                    r["weight_uncapped"], r["weight_capped"],
                    r["gmlos"], r["alos"],
                    r["post_acute"], r["special_pay"],
                )
                for r in rows
            ],
        )
    log.info("Loaded %d DRG rows into cms_drg_weights", len(rows))


async def load_cc_mcc_codes(pool: asyncpg.Pool, mcc_rows: list[dict], cc_rows: list[dict]) -> None:
    """Load CC/MCC codes. If a code is in both lists, MCC wins."""
    # Merge: MCC first so it takes precedence
    merged: dict[str, dict] = {r["icd10_code"]: r for r in mcc_rows}
    for r in cc_rows:
        if r["icd10_code"] not in merged:
            merged[r["icd10_code"]] = r

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM cms_cc_mcc_codes WHERE fiscal_year = 2025")
        await conn.executemany(
            """
            INSERT INTO cms_cc_mcc_codes
                (icd10_code, description, designation, fiscal_year)
            VALUES ($1, $2, $3, 2025)
            ON CONFLICT (icd10_code) DO UPDATE SET
                description = EXCLUDED.description,
                designation = EXCLUDED.designation,
                loaded_at = NOW()
            """,
            [(r["icd10_code"], r["description"], r["designation"]) for r in merged.values()],
        )
    log.info("Loaded %d CC/MCC codes (MCC: %d, CC: %d)",
             len(merged),
             sum(1 for r in merged.values() if r["designation"] == "MCC"),
             sum(1 for r in merged.values() if r["designation"] == "CC"))


# ── Step 5: Sanity-check queries ──────────────────────────────

async def verify_load(pool: asyncpg.Pool) -> None:
    """Run a few sanity queries against the loaded data."""
    async with pool.acquire() as conn:
        total_drg = await conn.fetchval("SELECT COUNT(*) FROM cms_drg_weights WHERE fiscal_year=2025")
        total_triplets = await conn.fetchval("""
            SELECT COUNT(DISTINCT triplet_base)
            FROM cms_drg_weights
            WHERE fiscal_year=2025 AND cc_mcc_status IN ('MCC','CC','NONE')
        """)
        total_mcc = await conn.fetchval("SELECT COUNT(*) FROM cms_cc_mcc_codes WHERE designation='MCC'")
        total_cc = await conn.fetchval("SELECT COUNT(*) FROM cms_cc_mcc_codes WHERE designation='CC'")

        # Heart failure triplet example
        hf_triplet = await conn.fetch("""
            SELECT drg, title, cc_mcc_status, weight_capped
            FROM cms_drg_weights
            WHERE triplet_base ILIKE '%HEART FAILURE%'
            ORDER BY drg
            LIMIT 5
        """)

        # Pneumonia triplet example
        pna_triplet = await conn.fetch("""
            SELECT drg, title, cc_mcc_status, weight_capped
            FROM cms_drg_weights
            WHERE triplet_base ILIKE '%PNEUMONIA%'
            ORDER BY drg
            LIMIT 5
        """)

    log.info("━" * 60)
    log.info("Load verification:")
    log.info("  Total DRGs loaded: %d", total_drg)
    log.info("  Complete CC/MCC triplets: %d", total_triplets)
    log.info("  MCC codes: %d", total_mcc)
    log.info("  CC codes: %d", total_cc)
    log.info("━" * 60)
    log.info("Heart failure triplet:")
    for r in hf_triplet:
        log.info("  DRG %d: %s (%s) = %.4f",
                 r["drg"], r["title"], r["cc_mcc_status"] or "—",
                 float(r["weight_capped"]))
    log.info("━" * 60)
    log.info("Pneumonia triplet:")
    for r in pna_triplet:
        log.info("  DRG %d: %s (%s) = %.4f",
                 r["drg"], r["title"], r["cc_mcc_status"] or "—",
                 float(r["weight_capped"]))


# ── Main ──────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Load CMS MS-DRG FY2025 data into PostgreSQL")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files from CMS even if already cached")
    args = parser.parse_args()

    # Step 1: Download
    download_and_extract(force=args.force)

    # Step 2/3: Parse files
    table5_path = DOWNLOAD_DIR / "FY2025 IPPS Final Rule Table 5.txt"
    mcc_path = DOWNLOAD_DIR / "tables6k" / "CMS-1808-F Table 6I - Complete MCC List - FY 2025.txt"
    cc_path = DOWNLOAD_DIR / "tables6k" / "CMS-1808-F Table 6J - Complete CC List - FY 2025.txt"

    drg_rows = parse_table_5(table5_path)
    mcc_rows = parse_cc_mcc_table(mcc_path, "MCC")
    cc_rows = parse_cc_mcc_table(cc_path, "CC")

    # Step 4: Load
    await init_database()
    pool = await get_pool()
    await load_drg_weights(pool, drg_rows)
    await load_cc_mcc_codes(pool, mcc_rows, cc_rows)

    # Step 5: Verify
    await verify_load(pool)


if __name__ == "__main__":
    asyncio.run(main())
