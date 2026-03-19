"""Parser for SNOMED CT RF2 (Release Format 2) tab-delimited files.

Reads Snapshot files and yields structured records. Generator-based —
never loads entire files into memory.

RF2 specification: https://confluence.ihtsdotools.org/display/DOCRELFMT
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

# RF2 files can have very large fields (OWL expressions, long descriptions)
csv.field_size_limit(sys.maxsize)

log = logging.getLogger("noesis.ontology")

# ── SNOMED CT type ID constants ──────────────────────────────────────────

FSN_TYPE_ID = "900000000000003001"       # Fully Specified Name
SYNONYM_TYPE_ID = "900000000000013009"   # Synonym

IS_A_TYPE_ID = "116680003"

# Top ~25 relationship type IDs → human-readable names
RELATIONSHIP_TYPE_NAMES: dict[str, str] = {
    "116680003": "IS_A",
    "363698007": "FINDING_SITE",
    "116676008": "ASSOCIATED_MORPHOLOGY",
    "260686004": "METHOD",
    "363714003": "INTERPRETS",
    "405813007": "PROCEDURE_SITE_DIRECT",
    "370135005": "PATHOLOGICAL_PROCESS",
    "272741003": "LATERALITY",
    "246454002": "OCCURRENCE",
    "127489000": "HAS_ACTIVE_INGREDIENT",
    "42752001": "DUE_TO",
    "363713009": "HAS_INTERPRETATION",
    "246075003": "CAUSATIVE_AGENT",
    "411116001": "HAS_DOSE_FORM",
    "762949000": "HAS_PRECISE_ACTIVE_INGREDIENT",
    "732943007": "HAS_BASIS_OF_STRENGTH_SUBSTANCE",
    "363700003": "DIRECT_MORPHOLOGY",
    "405815000": "PROCEDURE_SITE_INDIRECT",
    "363704007": "PROCEDURE_SITE",
    "260870009": "PRIORITY",
    "363710007": "INDIRECT_MORPHOLOGY",
    "704319004": "INHERES_IN",
    "704318007": "PROPERTY_TYPE",
    "704327008": "DIRECT_DEVICE",
    "371881003": "DURING",
    "704326004": "PRECONDITION",
    "246112005": "SEVERITY",
    "263502005": "CLINICAL_COURSE",
    "47429007": "ASSOCIATED_WITH",
    "255234002": "AFTER",
    "363705008": "HAS_DEFINITIONAL_MANIFESTATION",
    "370130000": "PROPERTY",
    "370132008": "SCALE_TYPE",
    "370134009": "TIME_ASPECT",
    "246093002": "COMPONENT",
    "704321009": "CHARACTERIZES",
    "718497002": "INHERENT_LOCATION",
}

# Semantic tags to skip during embedding (not clinically useful)
SKIP_SEMANTIC_TAGS = frozenset({
    "namespace concept",
    "metadata",
    "foundation metadata concept",
    "core metadata concept",
    "linkage concept",
    "OWL metadata concept",
    "inactive concept",
    "navigational concept",
    "environment / location",
})

# Definition status ID → human-readable
DEFINITION_STATUS = {
    "900000000000074008": "Primitive",
    "900000000000073002": "Fully defined",
}

# ── Semantic tag extraction ──────────────────────────────────────────────

_SEMANTIC_TAG_RE = re.compile(r"\(([^)]+)\)\s*$")


def extract_semantic_tag(fsn: str) -> str:
    """Extract the semantic tag from a Fully Specified Name.

    Example: "Appendicitis (disorder)" → "disorder"
    """
    m = _SEMANTIC_TAG_RE.search(fsn)
    return m.group(1) if m else "unknown"


# ── File discovery ───────────────────────────────────────────────────────

def _find_file(snapshot_dir: Path, prefix: str, suffix: str = ".txt") -> Path:
    """Find a file in the Snapshot directory matching a prefix pattern."""
    # Check Terminology/ and Refset/Map/ subdirectories
    for subdir in ["Terminology", "Refset/Map", "Refset/Content", "Refset/Language"]:
        search_dir = snapshot_dir / subdir
        if not search_dir.exists():
            continue
        for f in search_dir.iterdir():
            if f.name.startswith(prefix) and f.name.endswith(suffix):
                return f
    raise FileNotFoundError(
        f"No file matching '{prefix}*{suffix}' in {snapshot_dir}"
    )


# ── Parsers ──────────────────────────────────────────────────────────────

def parse_concepts(snapshot_dir: str | Path) -> Iterator[dict]:
    """Parse sct2_Concept_Snapshot — yields active concepts.

    Yields:
        {sctid, module_id, definition_status}
    """
    path = _find_file(Path(snapshot_dir), "sct2_Concept_Snapshot")
    log.info("Parsing concepts from %s", path.name)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row in reader:
            if row["active"] != "1":
                continue
            count += 1
            yield {
                "sctid": row["id"],
                "module_id": row["moduleId"],
                "definition_status": DEFINITION_STATUS.get(
                    row["definitionStatusId"], row["definitionStatusId"]
                ),
            }
        log.info("Parsed %d active concepts", count)


def parse_descriptions(snapshot_dir: str | Path) -> Iterator[dict]:
    """Parse sct2_Description_Snapshot-en — yields active descriptions.

    Yields:
        {desc_id, concept_id, type_id, term, is_fsn, case_significance}
    """
    path = _find_file(Path(snapshot_dir), "sct2_Description_Snapshot-en")
    log.info("Parsing descriptions from %s", path.name)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row in reader:
            if row["active"] != "1":
                continue
            count += 1
            yield {
                "desc_id": row["id"],
                "concept_id": row["conceptId"],
                "type_id": row["typeId"],
                "term": row["term"],
                "is_fsn": row["typeId"] == FSN_TYPE_ID,
                "case_significance": row["caseSignificanceId"],
            }
        log.info("Parsed %d active descriptions", count)


def parse_relationships(snapshot_dir: str | Path) -> Iterator[dict]:
    """Parse sct2_Relationship_Snapshot (inferred) — yields active relationships.

    Yields:
        {rel_id, source_id, dest_id, type_id, type_name, rel_group}
    """
    path = _find_file(Path(snapshot_dir), "sct2_Relationship_Snapshot")
    log.info("Parsing inferred relationships from %s", path.name)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row in reader:
            if row["active"] != "1":
                continue
            type_id = row["typeId"]
            count += 1
            yield {
                "rel_id": row["id"],
                "source_id": row["sourceId"],
                "dest_id": row["destinationId"],
                "type_id": type_id,
                "type_name": RELATIONSHIP_TYPE_NAMES.get(type_id, f"REL_{type_id}"),
                "rel_group": int(row["relationshipGroup"]),
            }
        log.info("Parsed %d active inferred relationships", count)


def parse_icd10_map(snapshot_dir: str | Path) -> Iterator[dict]:
    """Parse der2_iisssccRefset_ExtendedMapSnapshot — yields active ICD-10 mappings.

    Yields:
        {snomed_id, icd10_code, map_group, map_priority, map_rule,
         map_advice, map_category_id}
    """
    path = _find_file(Path(snapshot_dir), "der2_iisssccRefset_ExtendedMap")
    log.info("Parsing ICD-10 extended map from %s", path.name)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row in reader:
            if row["active"] != "1":
                continue
            icd10_code = row.get("mapTarget", "").strip()
            if not icd10_code:
                continue
            count += 1
            yield {
                "snomed_id": row["referencedComponentId"],
                "icd10_code": icd10_code,
                "map_group": int(row["mapGroup"]),
                "map_priority": int(row["mapPriority"]),
                "map_rule": row.get("mapRule", ""),
                "map_advice": row.get("mapAdvice", ""),
                "map_category_id": row.get("mapCategoryId", ""),
            }
        log.info("Parsed %d active ICD-10 mappings", count)


# ── Description aggregation helper ───────────────────────────────────────

def build_concept_lookup(snapshot_dir: str | Path) -> dict[str, dict]:
    """Build an in-memory lookup: concept_id → {fsn, preferred_term, semantic_tag, synonyms}.

    Reads the full descriptions file and groups by concept_id.
    For each concept, captures the FSN, the first synonym (as preferred_term),
    and all additional synonyms.

    Returns:
        dict mapping sctid → {fsn, preferred_term, semantic_tag, synonyms}
    """
    lookup: dict[str, dict] = {}

    for desc in parse_descriptions(snapshot_dir):
        cid = desc["concept_id"]

        if cid not in lookup:
            lookup[cid] = {"fsn": "", "preferred_term": "", "semantic_tag": "", "synonyms": []}

        if desc["is_fsn"]:
            lookup[cid]["fsn"] = desc["term"]
            lookup[cid]["semantic_tag"] = extract_semantic_tag(desc["term"])
        else:
            if not lookup[cid]["preferred_term"]:
                # Use the first synonym as the preferred term
                lookup[cid]["preferred_term"] = desc["term"]
            # Collect all synonyms (including the preferred term)
            lookup[cid]["synonyms"].append(desc["term"])

    # For concepts with FSN but no synonym, use FSN minus the semantic tag
    for cid, info in lookup.items():
        if info["fsn"] and not info["preferred_term"]:
            # Strip "(semantic tag)" from FSN to get a clean preferred term
            info["preferred_term"] = _SEMANTIC_TAG_RE.sub("", info["fsn"]).strip()

    total_synonyms = sum(len(info["synonyms"]) for info in lookup.values())
    log.info(
        "Built concept lookup with %d entries, %d total synonyms (avg %.1f/concept)",
        len(lookup), total_synonyms, total_synonyms / len(lookup) if lookup else 0,
    )
    return lookup
