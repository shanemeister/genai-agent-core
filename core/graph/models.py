from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A node as returned to the frontend."""
    id: str
    label: str          # Node type: Concept, MemoryCard, Diagram, Session
    name: str           # Primary display text
    properties: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """An edge as returned to the frontend."""
    id: str
    source: str
    target: str
    label: str          # Relationship type: RELATED_TO, MENTIONS, etc.
    properties: dict = Field(default_factory=dict)


class GraphData(BaseModel):
    """Complete subgraph for frontend rendering."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class ConceptCreate(BaseModel):
    """Request to create a Concept node."""
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    source: Optional[str] = None


class RelationshipCreate(BaseModel):
    """Request to create a relationship between two nodes."""
    source_id: str
    target_id: str
    rel_type: str = "RELATED_TO"
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class NeighborRequest(BaseModel):
    """Request to expand a node's neighborhood."""
    node_id: str
    depth: int = Field(default=1, ge=1, le=3)
    limit: int = Field(default=25, ge=1, le=100)


class ChatSessionCreate(BaseModel):
    """Record a chat interaction for lineage tracking."""
    session_id: str
    user_message: str
    assistant_response: str
    model: Optional[str] = None
    processing_time: Optional[float] = None
    retrieved_doc_ids: List[str] = []


class DiagramLineage(BaseModel):
    """Link a diagram to the chat session that produced it."""
    diagram_id: str
    source_session_id: str
    prompt: str
    diagram_code: str
    model: Optional[str] = None


class DiagramToGraphRequest(BaseModel):
    """Request to import a Mermaid diagram into the knowledge graph."""
    diagram_code: str
    source: Optional[str] = "diagram_import"


class GraphToDiagramRequest(BaseModel):
    """Request to export a graph subgraph as Mermaid syntax."""
    node_ids: List[str] = Field(default_factory=list, description="If empty, exports full graph")
    depth: int = Field(default=1, ge=0, le=3)
    layout: str = Field(default="TD", description="flowchart direction: TD, LR, BT, RL")
