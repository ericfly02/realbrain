from __future__ import annotations

import os
from datetime import datetime

from realbrain.compat import UTC
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from realbrain.cognitive_loop import ContradictionReview, CuriosityQueue, EventExtractor, NightlyConsolidator, SynapseHygiene
from realbrain.dream_engine import DreamEngine
from realbrain.gbrain_adapter import GBrainAdapter
from realbrain.global_workspace import GlobalWorkspace
from realbrain.models import BrainEvent, Neuron, Synapse
from realbrain.obsidian_adapter import ObsidianAdapter
from realbrain.store import RealBrainStore


def _as_of() -> str:
    return datetime.now(UTC).isoformat()


def response(result: dict, *, warnings: list[str] | None = None, evidence_refs: list[str] | None = None) -> dict:
    return {
        "success": True,
        "source": "realbrain",
        "as_of": _as_of(),
        "result": result,
        "warnings": warnings or [],
        "evidence_refs": evidence_refs or [],
    }


class RealBrainToolContext:
    """Runtime configuration for exposing RealBrain as host-agent tools.

    OpenClaw, MCP servers, FastAPI bridges, Discord bots, and local CLIs can all
    call these functions. Keep the context explicit so secrets and personal data
    do not leak into package defaults.
    """

    def __init__(
        self,
        brain_root: str | Path | None = None,
        db_path: str | Path | None = None,
        gbrain_binary: str | None = None,
        gbrain_workdir: str | Path | None = None,
        bun_binary: str | None = None,
        gbrain_home: str | Path | None = None,
    ):
        self.brain_root = Path(brain_root or os.environ.get("REALBRAIN_ROOT", "./realbrain_vault")).resolve()
        self.db_path = Path(db_path or os.environ.get("REALBRAIN_DB", self.brain_root / "ops" / "brain" / "realbrain.sqlite")).resolve()
        self.gbrain_binary = gbrain_binary or os.environ.get("GBRAIN_BINARY")
        self.gbrain_workdir = Path(gbrain_workdir).resolve() if gbrain_workdir else os.environ.get("GBRAIN_WORKDIR")
        self.bun_binary = bun_binary or os.environ.get("BUN_BINARY")
        self.gbrain_home = Path(gbrain_home).resolve() if gbrain_home else os.environ.get("GBRAIN_HOME")

    def store(self) -> RealBrainStore:
        return RealBrainStore(self.db_path)

    def obsidian(self) -> ObsidianAdapter:
        return ObsidianAdapter(self.brain_root)

    def gbrain(self) -> GBrainAdapter:
        return GBrainAdapter(
            self.brain_root,
            gbrain_binary=self.gbrain_binary,
            workdir=self.gbrain_workdir,
            bun_binary=self.bun_binary,
            gbrain_home=self.gbrain_home,
        )


DEFAULT_CONTEXT = RealBrainToolContext()


def record_event(event: dict[str, Any], *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    parsed = BrainEvent(**event)
    saved = ctx.store().record_event(parsed)
    return response({"event": saved.model_dump(mode="json"), "db_path": str(ctx.db_path)}, evidence_refs=saved.evidence_refs)


def add_neuron(candidate: dict[str, Any], *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    neuron = Neuron(**candidate)
    saved = ctx.store().add_neuron(neuron)
    return response({"neuron": saved.model_dump(mode="json"), "db_path": str(ctx.db_path)}, evidence_refs=saved.evidence_refs)


def add_synapse(candidate: dict[str, Any], *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    try:
        synapse = Synapse(**candidate)
    except ValidationError as exc:
        return response(
            {"error": "invalid_synapse_candidate", "details": exc.errors()},
            warnings=["Synapse was rejected by schema validation."],
        )
    saved = ctx.store().add_synapse(synapse)
    return response({"synapse": saved.model_dump(mode="json"), "db_path": str(ctx.db_path)}, evidence_refs=saved.evidence_refs)


def reinforce_synapse(edge_id: str, reason: str, delta: float = 0.05, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    updated = ctx.store().reinforce_synapse(edge_id, delta=delta, reason=reason)
    return response({"synapse": updated.model_dump(mode="json") if updated else None}, warnings=[] if updated else ["synapse not found"])


def search_memory(query: str, filters: dict[str, Any] | None = None, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    filters = filters or {}
    store = ctx.store()
    local_neurons = store.find_neurons(query=query, type=filters.get("type"), limit=int(filters.get("limit", 20)))
    external = ctx.gbrain().search(query, limit=int(filters.get("limit", 10)))
    return response(
        {
            "query": query,
            "neurons": [n.model_dump(mode="json") for n in local_neurons],
            "gbrain_output": external["gbrain_output"],
            "markdown_hits": external["obsidian_hits"],
        },
        warnings=external["warnings"],
    )


def resolve_canonical_path(title_or_path: str, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    resolved = ctx.gbrain().resolve_canonical_path(title_or_path)
    return response(resolved, warnings=resolved.get("warnings", []))


def write_markdown_update(path: str, content: str, overwrite: bool = False, evidence_refs: list[str] | None = None, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    written = ctx.obsidian().write_markdown(path, content, overwrite=overwrite, evidence_refs=evidence_refs)
    event = BrainEvent(
        event_type="obsidian_edit",
        source="realbrain",
        content=f"Wrote markdown update to {written['relative_path']}",
        metadata={"path": written["relative_path"], "bytes": written["bytes"]},
        sensitivity="personal",
        evidence_refs=evidence_refs or [],
        processed_status="consolidated",
    )
    ctx.store().record_event(event)
    return response({"write": written, "event_id": event.id}, evidence_refs=evidence_refs or [])


def activate(node_or_query: str, depth: int = 1, budget: int = 5, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    result = GlobalWorkspace(ctx.store(), ctx.obsidian()).activate(node_or_query, depth=depth, budget=budget)
    return response(result, evidence_refs=[result.get("workspace_path", "")])


def get_global_workspace(*, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    current = GlobalWorkspace(ctx.store(), ctx.obsidian()).read_current()
    return response({"workspace": current}, evidence_refs=[current.get("relative_path", "")])


def dream(mode: str = "rem_generation", budget: int = 5, focus_area: str | None = None, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    result = DreamEngine(ctx.store(), ctx.obsidian()).run(mode=mode, budget=budget, focus_area=focus_area)
    return response(
        result,
        warnings=["Dream output is hypothesis/suggestion only, not fact.", "No high-impact actions were executed or authorized."],
        evidence_refs=[result["summary_path"]],
    )


def extract_events(limit: int = 20, dry_run: bool = False, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    result = EventExtractor(ctx.store()).run(limit=limit, dry_run=dry_run)
    return response(result, warnings=["Extracted claims are inference-level beliefs; they are not promoted to facts automatically."])


def synapse_hygiene(stale_after_days: int = 30, max_decay: float = 0.05, dry_run: bool = False, limit: int = 100, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    result = SynapseHygiene(ctx.store()).run(stale_after_days=stale_after_days, max_decay=max_decay, dry_run=dry_run, limit=limit)
    return response(result, warnings=["Synapse hygiene only adjusts internal graph weights/status; it does not change durable facts."])


def review_contradictions(limit: int = 50, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    result = ContradictionReview(ctx.store(), ctx.obsidian()).run(limit=limit)
    return response(result, warnings=["Contradiction review is a queue for evidence review, not automatic truth resolution."], evidence_refs=[result["path"]])


def curiosity_queue(limit: int = 20, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    result = CuriosityQueue(ctx.store(), ctx.obsidian()).run(limit=limit)
    return response(result, warnings=["Curiosity items are questions/suggestions only."], evidence_refs=[result["path"]])


def nightly_consolidation(budget: int = 20, focus_area: str | None = None, dry_run: bool = False, *, ctx: RealBrainToolContext = DEFAULT_CONTEXT) -> dict:
    result = NightlyConsolidator(ctx.store(), ctx.obsidian()).run(budget=budget, focus_area=focus_area, dry_run=dry_run)
    return response(result, warnings=["Nightly consolidation is internal memory hygiene only."], evidence_refs=[result["summary_path"]])
