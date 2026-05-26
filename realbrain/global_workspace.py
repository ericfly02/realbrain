from __future__ import annotations

from datetime import datetime, timedelta

from .compat import UTC
from pathlib import Path

from .models import GlobalWorkspaceItem
from .obsidian_adapter import ObsidianAdapter
from .store import RealBrainStore


class GlobalWorkspace:
    """Small Obsidian-backed active attention board.

    This is intentionally conservative: it surfaces salience and related nodes,
    but does not execute actions. Later phases can add richer activation
    propagation and agent routing on top of this file/API contract.
    """

    def __init__(self, store: RealBrainStore, obsidian: ObsidianAdapter):
        self.store = store
        self.obsidian = obsidian

    @staticmethod
    def _synapse_key(synapse) -> tuple:
        if isinstance(synapse, dict):
            return (
                synapse.get("id"),
                synapse.get("source_neuron_id"),
                synapse.get("target_neuron_id"),
                synapse.get("relation_type"),
            )
        return (
            getattr(synapse, "id", None),
            getattr(synapse, "source_neuron_id", None),
            getattr(synapse, "target_neuron_id", None),
            getattr(synapse, "relation_type", None),
        )

    @classmethod
    def _dedupe_synapses(cls, synapses: list) -> list:
        unique = []
        seen = set()
        for synapse in synapses:
            key = cls._synapse_key(synapse)
            if key in seen:
                continue
            seen.add(key)
            unique.append(synapse)
        return unique

    def activate(self, node_or_query: str, *, depth: int = 1, budget: int = 5) -> dict:
        budget = max(1, min(budget, 20))
        depth = max(0, min(depth, 3))
        matches = self.store.find_neurons(query=node_or_query, limit=budget)
        if not matches:
            matches = self.store.find_neurons(limit=budget)
        activated = []
        synapses = []
        for neuron in matches[:budget]:
            updated = self.store.activate_neuron(neuron.id)
            if updated:
                activated.append(updated)
            if depth > 0:
                synapses.extend(self.store.list_synapses(neuron_id=neuron.id, limit=budget))
        synapses = self._dedupe_synapses(synapses)
        item = GlobalWorkspaceItem(
            salience=0.75 if activated else 0.35,
            domain="realbrain",
            summary=f"Activation request: {node_or_query}",
            why_active="Manual/API activation; related neurons and synapses surfaced for agent attention.",
            related_neurons=[n.id for n in activated],
            recommended_agent="orchestrator",
            expires_at=datetime.now(UTC) + timedelta(hours=6),
            actionability="suggest",
        )
        markdown = self.render([item], synapses=[s.model_dump(mode="json") for s in synapses])
        written = self.obsidian.write_markdown("brain/global-workspace/current.md", markdown, overwrite=True)
        return {
            "item": item.model_dump(mode="json"),
            "activated_neurons": [n.model_dump(mode="json") for n in activated],
            "synapses": [s.model_dump(mode="json") for s in synapses],
            "workspace_path": written["relative_path"],
        }

    def render(self, items: list[GlobalWorkspaceItem], *, synapses: list[dict] | None = None) -> str:
        synapses = self._dedupe_synapses(synapses or [])
        lines = [
            "# RealBrain Global Workspace — Current",
            "",
            f"- updated: {datetime.now(UTC).isoformat()}",
            "- authority: attention/suggestion only; no high-impact action authority",
            "",
            "## Active items",
        ]
        if not items:
            lines.append("- No active RealBrain workspace items.")
        for item in items:
            lines.extend(
                [
                    f"- **{item.summary}**",
                    f"  - salience: {item.salience:.2f}",
                    f"  - domain: {item.domain}",
                    f"  - why active: {item.why_active}",
                    f"  - actionability: {item.actionability}",
                    f"  - related neurons: {', '.join(item.related_neurons) if item.related_neurons else 'none'}",
                ]
            )
        lines.extend(["", "## Related synapses"])
        if not synapses:
            lines.append("- none surfaced")
        for synapse in synapses[:10]:
            lines.append(
                f"- {synapse.get('source_neuron_id')} --{synapse.get('relation_type')}:{synapse.get('weight')}--> {synapse.get('target_neuron_id')}"
            )
        return "\n".join(lines) + "\n"

    def read_current(self) -> dict:
        try:
            return self.obsidian.read_markdown("brain/global-workspace/current.md")
        except FileNotFoundError:
            markdown = self.render([])
            written = self.obsidian.write_markdown("brain/global-workspace/current.md", markdown, overwrite=True)
            return {"path": written["path"], "relative_path": written["relative_path"], "content": markdown, "truncated": False}
