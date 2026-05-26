from __future__ import annotations

from datetime import datetime

from .compat import UTC
from itertools import combinations
from pathlib import Path

from .global_workspace import GlobalWorkspace
from .models import BrainEvent, DreamRun
from .obsidian_adapter import ObsidianAdapter
from .store import RealBrainStore


class DreamEngine:
    """Bounded offline cognition for RealBrain.

    The first implementation is deliberately deterministic and safe: it reviews
    existing events/neurons/synapses, writes a report, and stores generated ideas
    as hypotheses only. It cannot execute tools, promote facts, or take external
    action.
    """

    def __init__(self, store: RealBrainStore, obsidian: ObsidianAdapter):
        self.store = store
        self.obsidian = obsidian
        self.workspace = GlobalWorkspace(store, obsidian)

    def run(self, *, mode: str = "rem_generation", budget: int = 5, focus_area: str | None = None) -> dict:
        budget = max(1, min(int(budget), 20))
        dream = DreamRun(mode=mode)  # validates mode
        events = self.store.list_events(limit=budget)
        neurons = self.store.find_neurons(query=focus_area, limit=budget) if focus_area else self.store.find_neurons(limit=budget)
        synapses = self.store.list_synapses(limit=budget)
        dream.input_neuron_ids = [n.id for n in neurons]
        dream.safety_flags = [
            "hypotheses_not_facts",
            "no_external_actions",
            "no_money_messages_calendar_health_or_destructive_authority",
            "human_review_required_for_high_impact_items",
        ]

        hypotheses = self._generate_hypotheses(mode=mode, neurons=neurons, synapses=synapses, events=events, focus_area=focus_area)
        dream.generated_hypotheses = hypotheses[:budget]
        dream.proposed_synapses = [s.id for s in synapses[:budget] if s.status == "active"]
        # Guardrail: dreams propose only. Accepted changes remain empty until a
        # separate validated consolidation/review step approves them.
        dream.accepted_changes = []
        dream.rejected_changes = []
        dream.completed_at = datetime.now(UTC)
        relative_path = self._write_report(dream, events=events, neurons=neurons, synapses=synapses, focus_area=focus_area)
        dream.summary_path = relative_path
        self.store.add_dream_run(dream)
        event = BrainEvent(
            event_type="dream",
            source="real_brain.dream_engine",
            content=f"RealBrain dream run {dream.id} completed in {mode}; hypotheses require validation before use as facts.",
            metadata={"dream_id": dream.id, "mode": mode, "summary_path": relative_path, "focus_area": focus_area},
            sensitivity="personal",
            evidence_refs=[relative_path],
            processed_status="archived",
        )
        self.store.record_event(event)
        workspace = self.workspace.activate(focus_area or "RealBrain dream", depth=1, budget=min(budget, 5))
        return {
            "dream_run": dream.model_dump(mode="json"),
            "event": event.model_dump(mode="json"),
            "summary_path": relative_path,
            "workspace": workspace,
        }

    def _generate_hypotheses(self, *, mode: str, neurons: list, synapses: list, events: list, focus_area: str | None) -> list[str]:
        names = [n.title for n in neurons]
        hypotheses: list[str] = []
        if mode == "nrem_consolidation":
            if events:
                hypotheses.append(
                    f"Consolidation candidate: review {len(events)} recent event(s) for durable facts, corrections, or decisions before promotion."
                )
            if synapses:
                hypotheses.append(
                    f"Synapse hygiene candidate: inspect {len(synapses)} active edge(s) for reinforcement, contradiction, or decay."
                )
        elif mode == "contradiction_scan":
            disputed = [s for s in synapses if s.relation_type == "contradicts" or s.status == "disputed"]
            hypotheses.append(
                f"Contradiction scan hypothesis: {len(disputed)} disputed/contradictory edge(s) need human or operator review."
            )
        else:
            for a, b in combinations(names[:5], 2):
                hypotheses.append(
                    f"Hypothesis: connecting '{a}' with '{b}' may reveal a useful cross-domain opportunity; validate with evidence before promotion."
                )
            if focus_area and not hypotheses:
                hypotheses.append(
                    f"Hypothesis: '{focus_area}' may need more linked neurons or evidence; add a curiosity item if it blocks an active goal."
                )
        if not hypotheses:
            hypotheses.append("Hypothesis: RealBrain needs more recorded events/neurons before meaningful dream synthesis is possible.")
        return hypotheses

    def _write_report(self, dream: DreamRun, *, events: list, neurons: list, synapses: list, focus_area: str | None) -> str:
        day = datetime.now(UTC).date().isoformat()
        folder = "brain/sleep-reports" if dream.mode == "nrem_consolidation" else "brain/dreams"
        path = f"{folder}/{day}.md"
        existing = ""
        try:
            existing = self.obsidian.read_markdown(path)["content"].rstrip() + "\n\n"
        except FileNotFoundError:
            pass
        lines = [
            f"## {dream.mode} — {dream.id}",
            "",
            f"- started_at: {dream.started_at.isoformat()}",
            f"- completed_at: {dream.completed_at.isoformat() if dream.completed_at else ''}",
            f"- focus_area: {focus_area or 'general'}",
            "- authority: hypothesis/suggestion only; no external or high-impact actions",
            "- fact_status: dream output is **not fact** until validated with evidence",
            f"- input_neurons: {', '.join(dream.input_neuron_ids) if dream.input_neuron_ids else 'none'}",
            f"- event_count_reviewed: {len(events)}",
            f"- synapse_count_reviewed: {len(synapses)}",
            "",
            "### Generated hypotheses / consolidation candidates",
        ]
        for hyp in dream.generated_hypotheses:
            lines.append(f"- {hyp}")
        lines.extend(["", "### Safety flags"])
        for flag in dream.safety_flags:
            lines.append(f"- {flag}")
        lines.extend(["", "### Next validation step", "- Search GBrain/Obsidian and gather source-backed evidence before promoting any hypothesis to a belief/fact."])
        markdown = existing + "\n".join(lines) + "\n"
        written = self.obsidian.write_markdown(path, markdown, overwrite=True)
        return written["relative_path"]
