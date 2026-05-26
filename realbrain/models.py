from __future__ import annotations

from datetime import datetime

from .compat import UTC
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


Sensitivity = Literal["public", "personal", "private", "secret"]
ProcessedStatus = Literal["new", "extracted", "consolidated", "archived", "rejected"]
EventType = Literal[
    "conversation",
    "tool_result",
    "obsidian_edit",
    "gbrain_sync",
    "calendar_event",
    "finance_snapshot",
    "health_metric",
    "repo_event",
    "web_research",
    "reflection",
    "dream",
    "user_feedback",
    "contradiction",
    "curiosity",
    "decision",
]
NeuronType = Literal[
    "concept",
    "person",
    "company",
    "project",
    "goal",
    "decision",
    "claim",
    "episode",
    "skill",
    "source",
    "question",
    "hypothesis",
    "procedure",
    "metric",
    "agent",
    "tool",
]
NeuronState = Literal["active", "dormant", "archived", "superseded"]
RelationType = Literal[
    "supports",
    "contradicts",
    "caused_by",
    "part_of",
    "related_to",
    "depends_on",
    "similar_to",
    "learned_from",
    "used_for",
    "blocks",
    "enables",
    "owned_by",
    "observed_with",
    "predicts",
    "next_step",
]
SynapseStatus = Literal["active", "weak", "disputed", "archived"]
BeliefStatus = Literal["fact", "inference", "hypothesis", "disproven", "stale"]
DreamMode = Literal[
    "nrem_consolidation",
    "rem_generation",
    "future_simulation",
    "contradiction_scan",
    "idea_synthesis",
]
Actionability = Literal["none", "suggest", "ask_user", "run_safe_tool", "needs_approval"]


class RealBrainModel(BaseModel):
    """Base model with strict fields and JSON-friendly defaults."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BrainEvent(RealBrainModel):
    id: str = Field(default_factory=lambda: _id("evt"))
    event_type: EventType
    source: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)
    user_id: str | None = None
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    sensitivity: Sensitivity = "personal"
    evidence_refs: list[str] = Field(default_factory=list)
    processed_status: ProcessedStatus = "new"


class Neuron(RealBrainModel):
    id: str = Field(default_factory=lambda: _id("nrn"))
    type: NeuronType
    title: str = Field(min_length=1)
    canonical_path: str | None = None
    summary: str = ""
    state: NeuronState = "active"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: int = Field(default=5, ge=1, le=10)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    last_activated_at: datetime | None = None
    activation_count: int = Field(default=0, ge=0)
    created_from_event: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Synapse(RealBrainModel):
    id: str = Field(default_factory=lambda: _id("syn"))
    source_neuron_id: str = Field(min_length=1)
    target_neuron_id: str = Field(min_length=1)
    relation_type: RelationType
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    last_reinforced_at: datetime | None = None
    reinforcement_count: int = Field(default=0, ge=0)
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    status: SynapseStatus = "active"

    @field_validator("target_neuron_id")
    @classmethod
    def no_self_edge(cls, value: str, info: Any) -> str:
        source = info.data.get("source_neuron_id") if hasattr(info, "data") else None
        if source and value == source:
            raise ValueError("synapse cannot connect a neuron to itself")
        return value


class Belief(RealBrainModel):
    id: str = Field(default_factory=lambda: _id("bel"))
    claim_text: str = Field(min_length=1)
    status: BeliefStatus = "inference"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    owner_domain: str | None = None
    review_after: datetime | None = None
    created_from_event: str | None = None


class DreamRun(RealBrainModel):
    id: str = Field(default_factory=lambda: _id("drm"))
    mode: DreamMode
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    input_neuron_ids: list[str] = Field(default_factory=list)
    generated_hypotheses: list[str] = Field(default_factory=list)
    proposed_synapses: list[str] = Field(default_factory=list)
    accepted_changes: list[str] = Field(default_factory=list)
    rejected_changes: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    summary_path: str | None = None


class GlobalWorkspaceItem(RealBrainModel):
    item_id: str = Field(default_factory=lambda: _id("gws"))
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    domain: str = "general"
    summary: str = Field(min_length=1)
    why_active: str = ""
    related_neurons: list[str] = Field(default_factory=list)
    recommended_agent: str | None = None
    expires_at: datetime | None = None
    actionability: Actionability = "none"
    created_at: datetime = Field(default_factory=utc_now)
