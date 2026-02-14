# GenAI Workshop (V0)

A local-first, visual GenAI workspace designed to help think, explore, and build
with modern AI systems — without turning conversation into the product.

This project prioritizes **artifacts over chat**, **human judgment over automation**,
and **visual reasoning over text-only interaction**.

---

## Why This Exists

Most GenAI tools optimize for conversation.
This project optimizes for **thinking**.

The goal is not to replace human reasoning, but to **externalize it**:
- diagrams instead of vague descriptions
- graphs instead of mental webs
- explicit memory instead of silent accumulation

The system produces durable artifacts that can be revisited, refined,
and reasoned over — locally, privately, and intentionally.

### Ethical Position

This project treats AI as an extension of human judgment, not a replacement for it.
While tools are sharper than ever, judgment remains human. The system may preserve
explicitly approved values and cognitive framing, but it may never claim continuity
of self, infer personal identity, or operate without ongoing consent. A system that
cannot recognize values worth preserving cannot be trusted to preserve meaning.

---

## V0 Scope: One Well-Designed Room

Version 0 deliberately ships with **three artifacts only**.

### 1. Diagram Canvas
- Render and edit PlantUML / Mermaid diagrams
- Diagrams are generated or modified by the system
- Text source is always visible and editable
- Diagrams are first-class artifacts, not screenshots

### 2. Graph Explorer
- Entity + edge graph extracted from text, diagrams, or memory
- Nodes expand, cluster, and animate as relationships are discovered
- Graphs support exploration, not manual drawing
- Graphs can be converted into diagrams and vice versa

### 3. Memory Deck
- Explicit, human-approved memory cards
- Each card proposes something the system believes it learned
- User can approve, edit, or reject every memory
- Memory is visible, searchable, and revocable

**The Mind File is NOT part of V0.**

V0 exists to make the Mind File possible — safely and deliberately.

---

## Core Interaction Loop

All functionality in V0 supports this loop:

Ask → Retrieve → Reason → Produce Artifact → Propose Memory

If a feature does not strengthen this loop, it does not belong in V0.

---

## Architecture Overview

### Visual Layer (Tauri - migrated from Electron)
- Cross-platform desktop application (macOS, Linux, Windows)
- Built with React + TypeScript + Vite
- Responsible for:
  - rendering artifacts (Mermaid diagrams, Neo4j graphs, memory cards)
  - animations and transitions
  - direct manipulation (drag, expand, pin, zoom)
- Acts as the system's *visual cortex*
- **Current deployment:** Mac client (192.168.4.22:1420) ↔ Linux workstation (192.168.4.25)

### Intelligence Layer (Python FastAPI)
- Local FastAPI service on workstation
- Responsible for:
  - LLM interaction (Llama-3-8B-Local)
  - RAG and retrieval (embeddings + vector search)
  - artifact generation (diagrams, graphs, memories)
  - memory proposal logic
  - agent orchestration
- **Ports:** 8080 (LLM service), 8008 (Memory backend)

### Data Layer
- **SQLite:** Lightweight artifact metadata, development
- **PostgreSQL (port 5433):** Structured data, pgvector embeddings, sessions
- **Neo4j (port 7687):** Knowledge graph, artifact lineage, concept relationships

### GenAI Components (Current & Planned)

| Component | Role | Status | Use Case |
|---------|------|--------|----------|
| Local LLMs | Core reasoning | ✅ Implemented | Llama-3-8B for chat, diagram generation |
| RAG + Vector DB | Evidence-based responses | 🔨 Backend ready | Retrieve relevant memories + context |
| Embeddings | Similarity & recall | 📋 Planned | Semantic search across artifacts |
| **LangChain** | Chains & prompt templates | 📋 Planned | Structured reasoning pipelines |
| **LangGraph** | Multi-step workflows | 📋 Planned | Visible agent reasoning graphs |
| **Agents** | Autonomous task execution | 📋 Planned | Research, diagram refinement, memory curation |
| **MCP** | Tool protocol & connectors | 📋 Planned | Unified interface for external tools |
| Multimodal input | Image/diagram interpretation | 📋 Future | Convert sketches to diagrams |

Each component must produce a **felt capability**, not just exist for coverage.

**Legend:** ✅ Done | 🔨 In Progress | 📋 Planned | 🔮 Future

---

## Artifacts as First-Class Objects

Every output in the system is an **Artifact** with:
- type (diagram, graph, memory card, session)
- provenance (sources, tools, model)
- lineage (what it was derived from)
- lifecycle (draft → accepted → locked)

Chat messages are transient.
Artifacts are durable.

---

## Roadmap: Rooms, Not Features

- **Room 0 (V0):** Core Workspace (diagrams, graphs, memory cards)
- **Room 1:** Mind File (personal pattern memory)
- **Room 2:** External Knowledge (documents, repos, web)
- **Room 3:** Agents & Automation
- **Room 4:** Visual Simulations / Game-like Modes

Each room is an addition — not a rewrite.

---

## Non-Goals (Important)

- No cloud dependency
- No silent memory accumulation
- No “chatbot as the product”
- No premature automation

---

## Status

**Current Implementation (V0 - Phase 1):**
- ✅ Tauri desktop app with React frontend
- ✅ Chat interface with LLM (Llama-3-8B-Local)
- ✅ Diagram Canvas with Mermaid rendering
- ✅ Memory Deck with approval workflow
- ✅ SQLite persistence for memory cards
- ✅ Persistent chat history across tabs

**Missing Critical V0 Components:**
- ⚠️ Graph Explorer (Neo4j integration) - **PRIORITY**
- ⚠️ Full RAG loop (backend exists, frontend not wired)
- ⚠️ Automatic memory proposals from chat
- ⚠️ Artifact lineage tracking
- ⚠️ Diagram ↔ Graph bidirectional conversion

---

## Enhanced Capabilities Roadmap

### Phase 1: Complete V0 Core (Priority)

#### 1. Graph Explorer with Neo4j Integration
**Why:** The most important missing artifact in V0. Transforms app from "chat with memory" to "visual thinking workspace."

**Implementation:**
- Visual node/edge canvas using React Flow or Cytoscape.js
- Neo4j backend for knowledge graph storage
- Click to expand related concepts, cluster by category
- Export graph views as artifacts with provenance

**Use Case:**
```
User: "Show me authentication patterns"
→ Graph displays authentication node connected to:
  - Session management
  - CSRF protection
  - Memory cards: "Always validate tokens"
  - Previous diagrams: "API security layers"
→ User explores connections, refines graph
→ Exports refined graph as new diagram
```

#### 2. Full RAG Retrieval Loop
**Current:** Backend `/chat` endpoint has RAG, but frontend uses `/ask` (no retrieval)

**Fix:**
- Wire ChatPanel to use `/chat` endpoint
- Retrieve context from approved memories + existing diagrams
- Display retrieved context in expandable UI section
- Show provenance: "Retrieved 3 related memories, 1 diagram"

**Impact:** Responses become contextually aware of accumulated knowledge

#### 3. Automatic Memory Proposals
**Current:** Manual memory creation only

**Enhancement:**
- After each AI response, analyze for proposable insights
- AI suggests: values, heuristics, framing, or contextual facts
- Toast notification: "💭 Proposed 1 memory card"
- User reviews/approves in Memory Deck later

**Completes:** The "Propose Memory" step in core loop

#### 4. Artifact Lineage in Neo4j
**Foundation for everything else**

```cypher
// Track what created what
CREATE (chat:ChatMessage)-[:PRODUCED]->(diagram:Diagram)
CREATE (diagram)-[:PROPOSED]->(memory:MemoryCard)
CREATE (memory)-[:APPROVED_BY {timestamp: ...}]->(user:User)

// Query reasoning chains
MATCH path = (origin)-[:PRODUCED*]->(artifact)
RETURN path
```

**Enables:**
- "Show me where this idea came from"
- "What diagrams reference this memory?"
- Trust through transparency
- Prepares for Mind File (Room 1)

---

### Phase 2: Semantic Layer & Tool Integration

#### 5. PostgreSQL + pgvector for Semantic Search
**Current:** Substring search only (`q.lower() in text`)

**Enhancement:**
- Store embeddings for all artifacts (diagrams, memories, chat)
- Semantic similarity search: "authentication" finds "login security"
- Enable RAG retrieval from approved memories
- Support "show me similar cards" feature

**Benefits:**
- Find related content even with different wording
- Smarter retrieval for RAG
- Cluster similar concepts in Graph Explorer

#### 6. Diagram ↔ Graph Bidirectional Conversion
**V0 Promise:** "Graphs can be converted into diagrams and vice versa"

**Implementation:**
```python
# Mermaid → Neo4j
POST /artifact/diagram-to-graph
# Parse syntax, create nodes/edges in Neo4j

# Neo4j → Mermaid
POST /artifact/graph-to-diagram
# Query subgraph, generate flowchart with layout
```

**Workflow:**
1. AI generates Mermaid diagram from chat
2. User clicks "Explore as Graph"
3. Graph shows connections to other concepts (from Neo4j)
4. User refines in Graph Explorer
5. Exports back to Mermaid for presentation

#### 7. Session Persistence & Export
**Current:** Chat clears on app restart

**Enhancement:**
- Store chat sessions in PostgreSQL
- Track artifacts created per session
- Export session as markdown with embedded diagrams
- "Resume previous session" feature

---

### Phase 3: LangChain, LangGraph & Agentic AI

#### 8. LangChain Integration
**Purpose:** Structured reasoning pipelines with composable chains

**Valid Use Cases:**

**A. Multi-Step Diagram Refinement Chain**
```python
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate

# Chain 1: Generate initial diagram
# Chain 2: Critique diagram for clarity
# Chain 3: Refine diagram based on critique
# Chain 4: Extract memory proposals from final diagram

diagram_refinement_chain = SequentialChain(
    chains=[generate, critique, refine, extract_memories],
    input_variables=["user_prompt"],
    output_variables=["final_diagram", "proposed_memories"]
)
```

**Demo Value:** Shows structured AI reasoning, not just one-shot generation

**B. Memory Card Curation Chain**
```python
# Chain: Analyze chat → Extract insights → Categorize → Propose card
memory_proposal_chain = (
    PromptTemplate.from_template("Extract insights: {chat}")
    | llm
    | StrOutputParser()
    | categorize_insight
    | format_as_memory_card
)
```

**Monetization Angle:** "Automatic knowledge capture with human oversight"

**C. RAG Chain with Memory Retrieval**
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Neo4jVector

# Custom retriever that queries:
# 1. Approved memory cards (high priority)
# 2. Existing diagrams (medium priority)
# 3. Chat history (low priority)

qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    retriever=hybrid_retriever,  # Memory + Diagrams + History
    return_source_documents=True  # Show provenance
)
```

**Demo Value:** Transparent retrieval with source attribution

#### 9. LangGraph for Visible Reasoning
**Purpose:** Multi-step agent workflows as visual graphs (aligns with V0 philosophy!)

**Valid Use Cases:**

**A. Research Agent Graph**
```python
from langgraph.graph import StateGraph, END

# Define research workflow as graph
workflow = StateGraph()

workflow.add_node("plan", plan_research)
workflow.add_node("search_memory", search_memory_cards)
workflow.add_node("search_diagrams", search_existing_diagrams)
workflow.add_node("synthesize", synthesize_findings)
workflow.add_node("propose_diagram", generate_diagram_from_synthesis)
workflow.add_node("propose_memory", extract_learnings)

# Define edges (workflow)
workflow.add_edge("plan", "search_memory")
workflow.add_edge("plan", "search_diagrams")
workflow.add_edge("search_memory", "synthesize")
workflow.add_edge("search_diagrams", "synthesize")
workflow.add_edge("synthesize", "propose_diagram")
workflow.add_edge("synthesize", "propose_memory")
workflow.add_edge("propose_diagram", END)

# Compile and run
research_agent = workflow.compile()
```

**UI Integration:** Display LangGraph workflow as visual graph in Graph Explorer!
- Show agent reasoning steps in real-time
- User sees: "Planning → Searching Memory → Synthesizing → Generating Diagram"
- **Perfect fit for V0 philosophy:** "Visible reasoning workflows"

**Demo Value:**
- Transparent agent reasoning (no black box)
- Visual representation of thought process
- Aligns with "artifacts over chat" principle

**B. Diagram Improvement Agent**
```python
# Agent that iteratively improves diagrams
workflow = StateGraph()
workflow.add_node("analyze", analyze_diagram)
workflow.add_node("identify_gaps", find_missing_concepts)
workflow.add_node("search_related", search_knowledge_graph)
workflow.add_node("suggest_improvements", propose_additions)
workflow.add_node("human_review", wait_for_approval)  # Human in loop!

workflow.add_conditional_edges(
    "human_review",
    lambda state: "refine" if state["approved"] else "reconsider"
)
```

**Monetization Angle:** "AI-assisted diagram refinement with human oversight"

**C. Memory Curation Agent**
```python
# Agent that maintains memory card quality
workflow = StateGraph()
workflow.add_node("scan_pending", get_pending_memories)
workflow.add_node("find_duplicates", semantic_similarity_check)
workflow.add_node("find_contradictions", logical_consistency_check)
workflow.add_node("suggest_merges", propose_card_merges)
workflow.add_node("wait_approval", human_review)
```

**Demo Value:** Proactive knowledge base maintenance

#### 10. Autonomous Agents (Agentic AI)
**Purpose:** Multi-step task execution with tool use

**Valid Use Cases:**

**A. Research Agent**
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

tools = [
    Tool(
        name="SearchMemory",
        func=search_memory_cards,
        description="Search approved memory cards for relevant knowledge"
    ),
    Tool(
        name="SearchDiagrams",
        func=search_diagrams,
        description="Find existing diagrams related to topic"
    ),
    Tool(
        name="QueryGraph",
        func=query_neo4j,
        description="Explore knowledge graph for concept relationships"
    ),
    Tool(
        name="GenerateDiagram",
        func=create_mermaid_diagram,
        description="Create new Mermaid diagram from findings"
    ),
    Tool(
        name="ProposeMemory",
        func=propose_memory_card,
        description="Propose new memory card from learnings"
    )
]

research_agent = create_react_agent(
    llm=local_llm,
    tools=tools,
    prompt=research_prompt
)

executor = AgentExecutor(agent=research_agent, tools=tools, verbose=True)
```

**User Interaction:**
```
User: "Research authentication best practices and create a diagram"

Agent (visible steps):
1. 🔍 Searching memory cards... Found 3 relevant cards
2. 📊 Searching existing diagrams... Found 1 related diagram
3. 🕸️ Querying knowledge graph... Found connections to OAuth, JWT, Sessions
4. ✍️ Generating diagram from findings...
5. 💭 Proposing 2 new memory cards...

Result:
- New diagram: "Authentication Flow Best Practices"
- 2 proposed memories for review
- Artifact lineage tracked in Neo4j
```

**Demo Value:**
- Transparent agent reasoning (ReAct pattern shows thoughts)
- Human approval gates for memory proposals
- Aligns with "human judgment over automation"

**B. Diagram Refinement Agent**
```python
tools = [
    Tool(name="AnalyzeDiagram", func=parse_mermaid_syntax),
    Tool(name="FindRelatedConcepts", func=query_neo4j_for_related),
    Tool(name="SuggestAdditions", func=identify_missing_nodes),
    Tool(name="CheckConsistency", func=validate_with_memories),
]

# Agent iteratively improves diagram with user feedback loop
```

**Monetization Angle:** "AI-powered diagram assistant"

**C. Memory Maintenance Agent**
```python
tools = [
    Tool(name="FindDuplicates", func=semantic_similarity_search),
    Tool(name="FindContradictions", func=logical_consistency_check),
    Tool(name="SuggestMerges", func=propose_card_combinations),
    Tool(name="ArchiveOutdated", func=identify_stale_cards),
]

# Runs periodically, proposes maintenance actions for human review
```

**Demo Value:** Proactive knowledge base health

#### 11. MCP (Model Context Protocol)
**Purpose:** Unified tool interface for external systems

**Valid Use Cases:**

**A. MCP Server for GenAI Workshop**
```python
# Expose GenAI Workshop capabilities as MCP tools
from mcp.server import MCPServer

mcp_server = MCPServer()

@mcp_server.tool()
async def search_memories(query: str, category: Optional[str] = None):
    """Search approved memory cards by semantic similarity"""
    # Returns memory cards with provenance

@mcp_server.tool()
async def create_diagram(prompt: str, style: str = "flowchart"):
    """Generate Mermaid diagram from natural language"""
    # Returns diagram code + artifact ID

@mcp_server.tool()
async def query_knowledge_graph(concept: str, depth: int = 2):
    """Explore Neo4j graph for related concepts"""
    # Returns subgraph as JSON

@mcp_server.tool()
async def propose_memory_card(text: str, category: str, reason: str):
    """Propose new memory card for human review"""
    # Returns pending card ID
```

**Use Case:** Other AI systems can access your knowledge base through MCP protocol

**Monetization Angle:** "API for personal/team knowledge graphs"

**B. MCP Clients for External Tools**
```python
# Connect to external data sources via MCP
from mcp.client import MCPClient

# Notion MCP client
notion = MCPClient("notion://workspace")
await notion.call_tool("search_pages", {"query": "authentication"})

# GitHub MCP client
github = MCPClient("github://repo")
await github.call_tool("search_code", {"query": "OAuth implementation"})

# Confluence MCP client
confluence = MCPClient("confluence://space")
await confluence.call_tool("get_page", {"id": "123"})
```

**Use Case:**
- Import external knowledge into GenAI Workshop
- Agent can search Notion/GitHub/Confluence while generating diagrams
- Unified interface for multiple data sources

**Demo Value:** Demonstrates modern AI tooling patterns

**Monetization Angle:** "Connect your knowledge sources" (Room 2: External Knowledge)

**C. MCP Tool Registry**
```python
# Central registry of available tools for agents
tool_registry = MCPToolRegistry()

# Register internal tools
tool_registry.register("memory", memory_mcp_server)
tool_registry.register("diagrams", diagram_mcp_server)
tool_registry.register("graph", neo4j_mcp_server)

# Register external tools
tool_registry.register("notion", notion_mcp_client)
tool_registry.register("github", github_mcp_client)

# Agents can discover and use all tools dynamically
agent = create_agent_with_mcp(tool_registry)
```

**Demo Value:** Extensible architecture for future tools

---

### Phase 4: Advanced Features (Room 1+)

#### 12. Mind File System (Room 1)
**Foundation built in Phase 1-3:**
- Artifact lineage tracking (Neo4j)
- Semantic search (pgvector)
- Memory curation (agents)
- Visible reasoning (LangGraph)

**New capabilities:**
- Pattern recognition across artifacts
- Personal cognitive style analysis
- Predictive memory suggestions
- "Show me how I think about X over time"

#### 13. External Knowledge Integration (Room 2)
**Using MCP:**
- Connect Notion, Obsidian, Roam Research
- Import GitHub repos for analysis
- Web search with source tracking
- Document ingestion with provenance

#### 14. Advanced Agents (Room 3)
**With LangChain/LangGraph:**
- Research agents that span multiple sessions
- Diagram evolution agents (track changes over time)
- Cross-project insight discovery
- Collaborative agents (team knowledge)

---

## Monetization Strategy (For River AI / relufox.ai)

### Target Markets

**1. Individual Knowledge Workers**
- Researchers, engineers, writers, consultants
- Pricing: $15-30/month (local-first, no API costs)
- Value prop: "Your thinking workspace with memory"

**2. Small Teams (3-10 people)**
- Shared knowledge graphs
- Team memory cards (approved by team leads)
- Collaborative diagram refinement
- Pricing: $50-100/month
- Value prop: "Externalize team knowledge"

**3. Enterprise (Custom deployment)**
- On-premise deployment
- Custom integrations via MCP
- Compliance-ready (all data local)
- Pricing: Custom (5-6 figures annual)
- Value prop: "Institutional memory that doesn't leave your network"

### Competitive Advantages

**vs. Notion/Obsidian:**
- AI-native, not note-taking with AI bolted on
- Visual reasoning (graphs, diagrams) as first-class
- Automatic knowledge capture with human oversight

**vs. ChatGPT/Claude:**
- Artifacts > conversation
- Local-first (privacy, compliance)
- Persistent, queryable knowledge base
- Visible agent reasoning (LangGraph)

**vs. Roam Research/Logseq:**
- AI-powered knowledge synthesis
- Automatic graph construction (vs manual linking)
- Diagram generation from concepts

### Demo/Portfolio Value

**For River AI showcase:**
1. ✅ **LangChain:** Structured reasoning chains (diagram refinement)
2. ✅ **LangGraph:** Visible agent workflows in Graph Explorer
3. ✅ **Agentic AI:** Research agents with tool use (ReAct pattern)
4. ✅ **MCP:** Unified tool protocol for extensibility
5. ✅ **Neo4j:** Knowledge graph for concept relationships
6. ✅ **RAG:** Semantic retrieval from personal knowledge base
7. ✅ **Local LLMs:** Privacy-first AI (Llama-3)
8. ✅ **Multimodal:** (Future) Image → Diagram conversion

**Stand-out features:**
- Ethical AI: Human-in-the-loop for memory, explicit consent
- Transparent reasoning: LangGraph shows agent thought process
- Artifact-centric: Durable outputs, not transient chat
- Visual-first: Diagrams and graphs, not just text

---

## NEXT STEPS: Implementation Tracker

### ✅ COMPLETED (V0 Phase 1A)
- [x] Migrate from Electron to Tauri
- [x] Chat interface with Llama-3-8B-Local
- [x] Diagram Canvas with Mermaid rendering
- [x] Memory Deck with approval workflow
- [x] SQLite persistence for memory cards
- [x] Persistent chat history across tabs
- [x] Mermaid error suppression in Chat tab

### 🔨 IN PROGRESS (V0 Phase 1B - Current Sprint)
- [ ] **Graph Explorer with Neo4j** (CRITICAL - Priority #1)
  - [ ] Set up Neo4j driver in Python backend
  - [ ] Create graph data models (nodes, relationships)
  - [ ] Build Neo4j API endpoints (query, create, update)
  - [ ] Frontend: React Flow canvas for visualization
  - [ ] Click to expand nodes, cluster by category
  - [ ] Export graph as diagram

- [ ] **Wire full RAG loop**
  - [ ] Change ChatPanel to use `/chat` instead of `/ask`
  - [ ] Display retrieved context in UI
  - [ ] Show provenance sources

- [ ] **Artifact lineage tracking in Neo4j**
  - [ ] Chat → Diagram → Memory relationships
  - [ ] Query: "Show me where this came from"
  - [ ] UI: Lineage visualization in Graph Explorer

### 📋 PLANNED (V0 Phase 1C - Next Sprint)
- [ ] **Automatic memory proposals**
  - [ ] Post-chat analysis for insights
  - [ ] Toast notifications for proposed memories
  - [ ] Batch review in Memory Deck

- [ ] **Diagram ↔ Graph conversion**
  - [ ] Mermaid parser → Neo4j nodes/edges
  - [ ] Neo4j subgraph → Mermaid generator
  - [ ] UI: "Explore as Graph" button on diagrams

- [ ] **Session persistence**
  - [ ] PostgreSQL schema for chat sessions
  - [ ] Save/load session functionality
  - [ ] Export session as markdown

### 📋 BACKLOG (V0 Phase 2 - Semantic Layer)
- [ ] PostgreSQL + pgvector setup
- [ ] Embedding generation for all artifacts
- [ ] Semantic similarity search
- [ ] "Show similar cards" feature
- [ ] Enhanced RAG with semantic retrieval

### 🔮 FUTURE (Phase 3+ - Agentic AI)
- [ ] LangChain integration (chains for refinement)
- [ ] LangGraph workflows (visible reasoning graphs)
- [ ] Research agents with tool use
- [ ] Diagram refinement agents
- [ ] Memory curation agents
- [ ] MCP server (expose GenAI capabilities)
- [ ] MCP clients (Notion, GitHub, Confluence)
- [ ] Multimodal input (image → diagram)

### 🎯 CURRENT FOCUS (This Week)
**Goal:** Complete Graph Explorer (Neo4j) - the missing V0 artifact

**Tasks:**
1. Set up Neo4j Python driver (`neo4j` package)
2. Create data models: `ConceptNode`, `RelationshipEdge`, `ArtifactNode`
3. Build backend API:
   - `POST /graph/create-node`
   - `POST /graph/create-relationship`
   - `GET /graph/query` (Cypher queries)
   - `POST /graph/from-memory-cards` (auto-generate from existing memories)
4. Frontend: React Flow canvas with zoom/pan/expand
5. Integration: "Explore in Graph" button on Memory Deck + Diagram Canvas

**Success Criteria:**
- User can visualize memory cards as graph nodes
- Click to expand related concepts
- Graph is stored in Neo4j with full provenance
- Can export graph view as Mermaid diagram

### 📊 Progress Metrics
- **V0 Completion:** 55% (3/3 artifacts in UI, 1/3 fully functional with backend)
  - Diagram Canvas: 90% (missing graph integration)
  - Memory Deck: 85% (missing auto-proposals, semantic search)
  - Graph Explorer: 10% (UI exists in code but not implemented)

- **Core Loop Completion:** 60%
  - Ask: ✅ 100%
  - Retrieve: 🔨 40% (backend ready, frontend not wired)
  - Reason: ✅ 100%
  - Produce Artifact: ✅ 80% (diagrams + memories working, graphs missing)
  - Propose Memory: ⚠️ 20% (manual only, no auto-proposal)

- **LangChain/LangGraph/MCP:** 0% (Phase 3)
- **Monetization readiness:** 40% (working demo, missing killer features)

---

## Development Environment

**Mac Client:**
- IP: 192.168.4.22
- Vite dev server: port 1420
- HMR: port 1421

**Linux Workstation:**
- IP: 192.168.4.25
- LLM service: port 8080 (Llama-3-8B-Local)
- Memory backend: port 8008 (FastAPI)
- PostgreSQL: port 5433
- Neo4j: port 7687 (Cypher), 7474 (Browser)

**Tech Stack:**
- Frontend: Tauri + React + TypeScript + Vite
- Backend: Python 3.11+ + FastAPI + Uvicorn
- LLM: Llama-3-8B-Local (Ollama or llama.cpp)
- Databases: SQLite, PostgreSQL (pgvector), Neo4j
- AI Frameworks: LangChain, LangGraph (planned)

---

## Contributing

This project is currently in active development for For River AI (https://relufox.ai).

---

## License

TBD (considering Apache 2.0 or MIT for open-source release)