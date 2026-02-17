# Design: Ontology-Backed Response Validation

> **Status:** Design document — guiding all hardening and new feature decisions.
> **Author:** River AI / relufox.ai
> **Date:** 2026-02-16

---

## The Problem

LLMs produce fluent, confident text that can be wrong in subtle ways:
- **Missing concepts** — a financial analysis omits regulatory requirements
- **Incorrect relationships** — symptoms mapped to the wrong diagnosis
- **Shallow coverage** — an architecture design skips critical components

Current validation (Room 1) checks claims against RAG evidence and does simple
concept-presence checks in the graph. It answers: "Is there evidence for this
claim?" It does NOT answer: "What's missing from this response?"

---

## The Vision

**Use domain ontologies as structural guardrails.**

An ontology isn't just a list of terms — it's a graph of relationships that
encode how concepts connect in a domain. When an LLM produces a response about
a topic, we can:

1. **Identify the relevant ontology region** (what part of the graph applies?)
2. **Map response concepts to ontology nodes** (what did the LLM mention?)
3. **Find the gaps** (what's in the ontology but missing from the response?)
4. **Score structural alignment** (do the relationships match?)

### Example: Medical Diagnosis

```
Patient symptoms: fever, nausea, right lower quadrant pain

LLM response: "This could be appendicitis"

Ontology check (SNOMED CT):
  Appendicitis
    ├── finding_site → Appendix → part_of → Large_intestine
    ├── associated_symptom → RLQ_pain ✓ (present in symptoms)
    ├── associated_symptom → Fever ✓ (present)
    ├── associated_symptom → Nausea ✓ (present)
    ├── associated_symptom → Rebound_tenderness ✗ (NOT checked)
    └── differential → Mesenteric_lymphadenitis (NOT mentioned)

Coverage: 3/4 symptoms confirmed, 1 unchecked, 1 differential omitted
Result: "Appendicitis is plausible but rebound tenderness should be
         assessed. Consider mesenteric lymphadenitis as differential."
```

### Example: Architecture Review

```
User request: "Design an AWS data mesh architecture"

LLM response: Mentions S3, Glue, Lake Formation, Athena

Ontology check (AWS reference architecture):
  Data_Mesh
    ├── requires → Data_Governance ✓ (Lake Formation)
    ├── requires → Data_Catalog ✓ (Glue)
    ├── requires → Batch_Ingestion ✓ (Glue ETL)
    ├── requires → Stream_Ingestion ✗ (MISSING — no Kinesis/MSK)
    ├── requires → Data_Storage ✓ (S3)
    ├── requires → Query_Engine ✓ (Athena)
    └── requires → Access_Control ✓ (Lake Formation)

Coverage: 6/7 components present, 1 critical gap
Result: "Architecture covers 6/7 required components. Missing:
         streaming ingestion (Kinesis Data Streams or MSK)."
```

---

## Three-Level Implementation

### Level 1: Personal Knowledge Graph (current — partially done)

**What:** Extract concepts from every artifact (memory cards, chat sessions,
diagrams, documents) and store as nodes with relationships in Neo4j.

**Purpose:** Build the user's own domain knowledge over time. Each approved
memory card, each chat session, each imported document adds to a growing
personal graph that reflects what the user knows and cares about.

**Status:** Mostly implemented. `sync_memory_card()`, `record_chat_session()`,
`record_diagram_lineage()` all create Concept nodes with MENTIONS/DISCUSSES
edges. LLM-based concept extraction works. The personal KG grows with usage.

**Gap:** Document chunks are not yet synced to the graph (Phase 2b remaining
item). Concept extraction quality depends on LLM availability.

### Level 2: Ontology Coverage Check (next milestone)

**What:** Given a response about topic X, find the ontology subgraph around X,
compare against concepts in the response, return missing nodes.

**Implementation path:**

1. **Ontology loading** — Import selected ontology subgraphs into the `noesis`
   Neo4j database with a distinct label (e.g., `:OntologyNode`) to distinguish
   from user-created nodes. This avoids cross-database queries.

2. **Ontology region selection** — When a response needs validation:
   - Extract key concepts from the response (existing pipeline)
   - Use embedding similarity to find the nearest ontology region (embed
     ontology node names, store in pgvector with `source_type='ontology'`)
   - Retrieve the subgraph around matched nodes (existing `get_neighbors()`
     with depth 2-3)

3. **Coverage comparison** — Compare response concepts against ontology subgraph:
   - Direct match: response concept = ontology node name
   - Semantic match: response concept embedding is close to ontology node embedding
   - Relationship match: response implies A→B, ontology confirms or contradicts

4. **Gap report** — Return structured result:
   - `matched_concepts`: ontology nodes present in response
   - `missing_concepts`: ontology nodes absent from response (ranked by importance)
   - `novel_concepts`: response concepts not in ontology (may be valid additions)
   - `coverage_score`: matched / (matched + missing)
   - `relationship_alignment`: do the connections match?

**Effort:** 2-3 focused sessions once the foundation is hardened.

### Level 3: Multi-Ontology + Auto-Detection (future)

**What:** Detect the domain of a response, select the appropriate ontology,
run coverage check automatically.

**Implementation path:**

1. **Domain classifier** — Embed the user query, match against ontology
   descriptions to select which ontology applies (could be multiple).

2. **Ontology registry** — Metadata about available ontologies:
   ```
   ontology_id | name      | domain     | node_count | description
   fibo        | FIBO      | finance    | 80K+       | Financial Industry Business Ontology
   snomed      | SNOMED CT | medical    | varies     | Clinical terminology
   aws-ref     | AWS Ref   | cloud-arch | varies     | AWS reference architectures
   user-kg     | Personal  | mixed      | grows      | User's personal knowledge graph
   ```

3. **Composite validation** — Run coverage against personal KG AND domain
   ontology. Report both: "Your personal knowledge covers X, domain
   ontology shows Y is also relevant."

4. **Feedback loop** — When ontology reveals gaps, propose memory cards for
   the missing concepts. User approval enriches the personal KG.

**Effort:** Significant — multiple sessions. Build toward this incrementally.

---

## Data Model Decisions

### Ontology Nodes vs. User Nodes

Ontology nodes imported into `noesis` need to be distinguishable from
user-created nodes. Two options:

**Option A: Separate label (recommended)**
```
(:Concept {name: "data governance", source: "user"})
(:OntologyNode {name: "Data_Governance", ontology: "fibo", class_id: "..."})
(:OntologyNode)-[:EQUIVALENT_TO]->(:Concept)
```
- Pro: Clear separation, can query each independently
- Pro: Ontology imports don't pollute user's concept graph
- Pro: Can link user concepts to ontology nodes explicitly
- Con: More relationship types to manage

**Option B: Property-based distinction**
```
(:Concept {name: "data governance", source: "user", is_ontology: false})
(:Concept {name: "Data_Governance", source: "fibo", is_ontology: true})
```
- Pro: Simpler schema, same queries work for both
- Con: Easy to accidentally mix user and ontology concepts
- Con: Harder to bulk-update ontology imports

**Decision: Option A.** Clean separation is worth the schema complexity.
Ontologies are reference data; user concepts are personal data. Mixing them
creates confusion about what the user actually knows vs. what the system
imported.

### New Node Labels

| Label | Purpose |
|-------|---------|
| `OntologyNode` | Imported node from a domain ontology (FIBO, SNOMED, etc.) |
| `Ontology` | Metadata node for an ontology itself (name, domain, version) |

### New Relationship Types

| Type | Between | Purpose |
|------|---------|---------|
| `EQUIVALENT_TO` | Concept ↔ OntologyNode | User concept matches ontology node |
| `SUBCLASS_OF` | OntologyNode → OntologyNode | Ontology hierarchy |
| `RELATED_ONTOLOGY` | OntologyNode → OntologyNode | Domain relationship (e.g., requires, associated_symptom) |
| `BELONGS_TO` | OntologyNode → Ontology | Which ontology this node came from |
| `COVERS` | ChatSession/MemoryCard → OntologyNode | Validation result: this artifact covers this ontology concept |
| `MISSES` | ValidationResult → OntologyNode | Validation result: this concept was missing |

### Extended Validation Models

```python
class OntologyCoverage(BaseModel):
    ontology_id: str                    # Which ontology was checked
    ontology_name: str                  # Human-readable name
    matched_concepts: list[str]         # Ontology nodes found in response
    missing_concepts: list[str]         # Ontology nodes missing from response
    novel_concepts: list[str]           # Response concepts not in ontology
    coverage_score: float               # matched / (matched + missing)
    critical_gaps: list[str]            # Missing concepts marked as critical in ontology

class ValidatedClaim(BaseModel):       # Extended from current model
    text: str
    status: ClaimStatus
    confidence: float
    evidence: list[ClaimEvidence]
    graph_concepts_found: list[str]     # Existing
    graph_concepts_missing: list[str]   # Existing
    ontology_coverage: OntologyCoverage | None  # NEW
```

### pgvector Extension for Ontology Matching

Ontology node names/descriptions get embedded and stored in the `embeddings`
table with `source_type='ontology'`:

```
doc_id: "ontology:fibo:Data_Governance"
embedding: [768d vector]
text: "Data Governance - the exercise of authority and control..."
metadata: {"ontology": "fibo", "class_id": "...", "source_type": "ontology"}
source_type: "ontology"
```

This allows semantic matching: when a response mentions "data management
framework," the embedding search finds the nearest ontology node
("Data_Governance") even though the exact term doesn't match.

---

## FIBO Ontology — First Target

FIBO is already loaded in Neo4j (`fibo` database, 80K+ nodes). Strategy:

1. **Don't import all 80K nodes.** Most are deep hierarchy nodes that aren't
   useful for response validation.

2. **Import the useful layers:**
   - Top-level domain classes (Business Entities, Financial Instruments,
     Derivatives, Securities, etc.)
   - Key regulatory concepts (KYC, AML, Basel III, MiFID II)
   - Commonly referenced relationships

3. **Selective import script:**
   ```cypher
   -- Query fibo database for top-level classes
   -- Filter by depth, connection count, or curated list
   -- Insert into noesis database as :OntologyNode
   ```

4. **Embed imported nodes** in pgvector for semantic matching.

5. **Test with a finance question:** Ask LLM about a financial topic, run
   coverage check against FIBO subgraph, verify gap detection works.

---

## How This Shapes the Hardening Steps

Every hardening decision should support this design:

### Config (Step 1)
- Add `ONTOLOGY_REGISTRY` settings (list of available ontologies)
- Add `ONTOLOGY_IMPORT_BATCH_SIZE` for bulk loading
- Add `COVERAGE_THRESHOLD` (minimum score to consider "well-covered")

### Logging (Step 2)
- Graph sync failures must be logged — ontology validation depends on graph health
- Embedding failures must be logged — semantic matching depends on pgvector

### Router Split (Step 3)
- `routes/graph.py` — existing graph CRUD endpoints
- `routes/validation.py` — existing claim validation + future coverage checking
- `routes/ontology.py` — ontology import, registry, management
- Keep them separate so ontology features can evolve independently

### Health Checks (Step 4)
- Include Neo4j health (ontology validation is dead without it)
- Include pgvector embedding count (semantic matching needs populated vectors)
- Include ontology status (which ontologies are loaded, how many nodes)

### Tests (Step 5)
- Test concept extraction pipeline (foundation for coverage matching)
- Test graph node creation/retrieval (foundation for ontology node management)
- Test vector search with source_type filtering (foundation for ontology matching)
- Test validation pipeline end-to-end (will be extended for coverage checks)

---

## What Success Looks Like

### Near-term (Level 2 — after hardening)
A user asks: "What are the key components of a KYC compliance program?"

The LLM responds with a list of components. User clicks "Validate."

Validation result shows:
```
Claim Validation: 4/5 claims supported (existing)

Ontology Coverage (FIBO — Financial Regulation):
  Matched: Customer Due Diligence, Risk Assessment, Transaction Monitoring,
           Suspicious Activity Reporting
  Missing: Enhanced Due Diligence (for high-risk customers),
           Ongoing Monitoring (periodic re-assessment)
  Coverage: 4/6 (67%)

  Suggestion: "Response covers core KYC components but omits Enhanced Due
  Diligence and Ongoing Monitoring, both required for regulatory compliance."
```

### Long-term (Level 3)
The system detects the question is about finance, auto-selects FIBO, and also
checks the user's personal KG for relevant past conversations. The validation
combines domain knowledge (FIBO says EDD is required) with personal knowledge
(user previously discussed EDD in a session about HSBC compliance) to produce
a richer, more personalized gap analysis.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ontology import too large | Slow queries, memory issues | Selective import (curated subgraphs, not full 80K) |
| Semantic matching inaccurate | False positives/negatives in coverage | Combine embedding match with graph traversal (confirm via relationships) |
| LLM concept extraction unreliable | Coverage check misses real concepts | Fallback keyword extraction + user can manually add concepts |
| Neo4j licensing | Legal risk for commercial distribution | Use Apache 2.0 driver for queries only; evaluate Apache AGE migration if distributing |
| User overwhelmed by gap reports | Feature feels noisy instead of helpful | Rank gaps by importance (ontology node centrality); show top 3 by default |
| Ontology maintenance burden | Stale ontologies give wrong results | Version ontologies; import from authoritative sources; expose update mechanism |

---

## Non-Goals (explicitly excluded)

- **Ontology editing** — Users don't edit FIBO/SNOMED. They're reference data.
- **Automatic gap filling** — System reports gaps. It does NOT auto-generate content to fill them. Human judgment decides what matters.
- **Real-time validation** — Coverage check runs on-demand (like current validation), not during streaming. It's too expensive for real-time.
- **Ontology creation from scratch** — We import existing ontologies. Building new ones is out of scope.

---

## Appendix: Ontology Sources

| Ontology | Domain | License | Nodes | Notes |
|----------|--------|---------|-------|-------|
| FIBO | Finance | MIT (EDM Council) | 80K+ | Already loaded in Neo4j `fibo` database |
| SNOMED CT | Medical | National license required | 350K+ | Free for many countries; US via NLM |
| ICD-10 | Medical (diagnosis) | WHO copyright, free use | 70K+ | Diagnosis codes, widely available |
| LOINC | Medical (lab/clinical) | Free, requires license agreement | 98K+ | Lab observations and clinical measures |
| HL7 FHIR | Healthcare IT | CC0 (public domain) | Varies | Healthcare data exchange standard |
| AWS Well-Architected | Cloud architecture | Public | Curated | Would need to be built/curated manually |
| TOGAF | Enterprise architecture | Open Group, proprietary | Curated | Reference framework, would need curation |

---

## Summary

The graph database earns its place when it stops being a visualization tool
and starts being a **structural reasoning engine**. The path:

1. **Personal KG** (Level 1) — largely done. Keep growing it.
2. **Ontology coverage** (Level 2) — import domain ontologies as reference
   nodes, use embedding + graph traversal to check response completeness.
3. **Multi-ontology intelligence** (Level 3) — auto-detect domain, composite
   validation, feedback loop to enrich personal KG.

Every hardening step we take now is shaped by this destination. The config
system supports ontology settings. The routers separate graph, validation,
and ontology concerns. The tests cover the pipelines that ontology validation
will extend. The logging ensures we can debug the complex chain of embedding
→ matching → traversal → gap detection when it inevitably breaks.

The effort is worth it because **this is the product differentiator.** Any AI
can chat. Noesis validates against structural domain knowledge — and that's
what privacy-conscious professionals will pay for.
