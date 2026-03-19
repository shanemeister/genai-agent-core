"""Parser for RxNorm RRF (Rich Release Format) pipe-delimited files.

Reads RXNCONSO.RRF and RXNREL.RRF and yields structured records.
Generator-based — never loads entire files into memory.

RxNorm RRF specification:
  https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Iterator

# RRF files can have very large fields
csv.field_size_limit(sys.maxsize)

log = logging.getLogger("noesis.ontology")

# ── RxNorm term type (TTY) constants ──────────────────────────────────────

# Clinical term types to include in the concept load
CLINICAL_TTYS = frozenset({
    "IN",    # Ingredient
    "PIN",   # Precise Ingredient
    "MIN",   # Multiple Ingredients
    "BN",    # Brand Name
    "SCD",   # Semantic Clinical Drug
    "SBD",   # Semantic Branded Drug
    "SCDC",  # Semantic Clinical Drug Component
    "SBDC",  # Semantic Branded Drug Component
    "SCDF",  # Semantic Clinical Drug Form
    "SBDF",  # Semantic Branded Drug Form
    "SY",    # Synonym
    "TMSY",  # Tall Man Lettering Synonym
    "PSN",   # Prescribable Name
})

# TTYs to skip during embedding (not useful for NLP text matching)
SKIP_EMBED_TTYS = frozenset({"DF", "DFG", "ET", "SCDGP", "SBDFP", "SBDG", "SCDG"})

# Preferred TTYs for selecting the "best" term per concept (ordered by priority)
_PREFERRED_TTY_ORDER = ["PSN", "SCD", "SBD", "IN", "PIN", "MIN", "BN"]

# Relationship types to load from RXNREL
RXNORM_RELA_TYPES = frozenset({
    "has_ingredient",
    "ingredient_of",
    "has_tradename",
    "tradename_of",
    "has_dose_form",
    "dose_form_of",
    "consists_of",
    "constitutes",
    "has_precise_ingredient",
    "precise_ingredient_of",
    "isa",
    "inverse_isa",
    "has_quantified_form",
    "quantified_form_of",
    "has_form",
    "form_of",
    "contains",
    "contained_in",
})

# RXNCONSO column indices (0-based, pipe-delimited, no header)
_COL_RXCUI = 0
_COL_SAB = 11
_COL_TTY = 12
_COL_CODE = 13
_COL_STR = 14
_COL_SUPPRESS = 16

# RXNREL column indices
_REL_RXCUI1 = 0
_REL_RXCUI2 = 4
_REL_RELA = 7
_REL_SAB = 10
_REL_SUPPRESS = 14


# ── File discovery ────────────────────────────────────────────────────────

def _find_rrf_file(rrf_dir: Path, filename: str) -> Path:
    """Find an RRF file in the directory."""
    path = rrf_dir / filename
    if path.exists():
        return path
    # Try in rrf/ subdirectory
    rrf_sub = rrf_dir / "rrf" / filename
    if rrf_sub.exists():
        return rrf_sub
    raise FileNotFoundError(
        f"'{filename}' not found in {rrf_dir} or {rrf_dir / 'rrf'}"
    )


# ── Parsers ───────────────────────────────────────────────────────────────

def parse_rxn_concepts(rrf_dir: str | Path) -> Iterator[dict]:
    """Parse RXNCONSO.RRF — yield active RxNorm-sourced concept atoms.

    Filters to SAB='RXNORM', SUPPRESS='N', and clinical TTYs.

    Yields:
        {rxcui, term, tty}
    """
    path = _find_rrf_file(Path(rrf_dir), "RXNCONSO.RRF")
    log.info("Parsing RxNorm concepts from %s", path.name)

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 17:
                continue
            sab = row[_COL_SAB]
            suppress = row[_COL_SUPPRESS]
            tty = row[_COL_TTY]

            if sab != "RXNORM" or suppress != "N":
                continue
            if tty not in CLINICAL_TTYS:
                continue

            count += 1
            yield {
                "rxcui": row[_COL_RXCUI],
                "term": row[_COL_STR],
                "tty": tty,
            }

    log.info("Parsed %d active RxNorm concept atoms", count)


def parse_rxn_relationships(rrf_dir: str | Path) -> Iterator[dict]:
    """Parse RXNREL.RRF — yield active RxNorm relationships.

    Filters to SAB='RXNORM', SUPPRESS='N', and known RELA types.

    Yields:
        {rxcui1, rxcui2, rela}
    """
    path = _find_rrf_file(Path(rrf_dir), "RXNREL.RRF")
    log.info("Parsing RxNorm relationships from %s", path.name)

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 15:
                continue
            sab = row[_REL_SAB]
            suppress = row[_REL_SUPPRESS]
            rela = row[_REL_RELA]

            if sab != "RXNORM" or suppress != "N":
                continue
            if not rela or rela not in RXNORM_RELA_TYPES:
                continue

            rxcui1 = row[_REL_RXCUI1]
            rxcui2 = row[_REL_RXCUI2]
            if not rxcui1 or not rxcui2:
                continue

            count += 1
            yield {
                "rxcui1": rxcui1,
                "rxcui2": rxcui2,
                "rela": rela,
            }

    log.info("Parsed %d active RxNorm relationships", count)


def parse_snomed_crosswalk(rrf_dir: str | Path) -> Iterator[dict]:
    """Parse RXNCONSO.RRF for SNOMED CT crosswalk entries.

    SNOMED CT concepts appear in RXNCONSO with SAB='SNOMEDCT_US'.
    They share the same RXCUI as the corresponding RXNORM entries,
    giving us a free SNOMED↔RxNorm mapping.

    Yields:
        {rxcui, snomed_sctid}
    """
    path = _find_rrf_file(Path(rrf_dir), "RXNCONSO.RRF")
    log.info("Parsing SNOMED CT crosswalk from %s", path.name)

    count = 0
    seen = set()  # Deduplicate rxcui→sctid pairs

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 17:
                continue
            if row[_COL_SAB] != "SNOMEDCT_US":
                continue
            if row[_COL_SUPPRESS] != "N":
                continue

            rxcui = row[_COL_RXCUI]
            snomed_sctid = row[_COL_CODE]
            if not rxcui or not snomed_sctid:
                continue

            pair = (rxcui, snomed_sctid)
            if pair in seen:
                continue
            seen.add(pair)

            count += 1
            yield {
                "rxcui": rxcui,
                "snomed_sctid": snomed_sctid,
            }

    log.info("Parsed %d unique SNOMED CT crosswalk entries", count)


# ── Concept aggregation helper ────────────────────────────────────────────

def build_rxnorm_lookup(rrf_dir: str | Path) -> dict[str, dict]:
    """Build in-memory lookup: rxcui → {preferred_term, synonyms, tty}.

    Reads RXNCONSO.RRF and groups by RXCUI. Selects the best preferred
    term based on TTY priority, collects all other terms as synonyms.

    Returns:
        dict mapping rxcui → {preferred_term, synonyms, tty}
    """
    lookup: dict[str, dict] = {}

    for atom in parse_rxn_concepts(rrf_dir):
        rxcui = atom["rxcui"]
        term = atom["term"]
        tty = atom["tty"]

        if rxcui not in lookup:
            lookup[rxcui] = {
                "preferred_term": "",
                "synonyms": [],
                "tty": "",
                "_tty_priority": 999,
            }

        entry = lookup[rxcui]

        # Determine priority for this TTY
        try:
            priority = _PREFERRED_TTY_ORDER.index(tty)
        except ValueError:
            priority = len(_PREFERRED_TTY_ORDER)

        # Update preferred term if this TTY has higher priority
        if priority < entry["_tty_priority"]:
            # Demote current preferred term to synonyms
            if entry["preferred_term"]:
                entry["synonyms"].append(entry["preferred_term"])
            entry["preferred_term"] = term
            entry["tty"] = tty
            entry["_tty_priority"] = priority
        else:
            # Add as synonym (avoid duplicates)
            if term != entry.get("preferred_term") and term not in entry["synonyms"]:
                entry["synonyms"].append(term)

    # Clean up internal fields
    for rxcui, info in lookup.items():
        del info["_tty_priority"]

    total_synonyms = sum(len(info["synonyms"]) for info in lookup.values())
    log.info(
        "Built RxNorm lookup with %d concepts, %d total synonyms (avg %.1f/concept)",
        len(lookup), total_synonyms, total_synonyms / len(lookup) if lookup else 0,
    )
    return lookup
