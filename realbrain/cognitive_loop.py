from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .compat import UTC
from pathlib import Path
from typing import Any

from .dream_engine import DreamEngine
from .models import Belief, BrainEvent, Neuron, Synapse
from .obsidian_adapter import ObsidianAdapter
from .store import RealBrainStore

_KEYWORD_NEURONS: dict[str, tuple[str, str]] = {
    "realbrain": ("concept", "RealBrain"),
    "real brain": ("concept", "RealBrain"),
    "gbrain": ("tool", "GBrain"),
    "obsidian": ("tool", "Obsidian"),
    "synapse": ("concept", "Synapse"),
    "synapses": ("concept", "Synapse"),
    "dream": ("concept", "Dream Engine"),
    "dreaming": ("concept", "Dream Engine"),
    "contradiction": ("concept", "Contradiction Review"),
    "curiosity": ("concept", "Curiosity Queue"),
    "finance": ("concept", "Finance"),
    "health": ("concept", "Health"),
    "training": ("concept", "Training"),
}

_CLAIM_MARKERS = (
    " should ",
    " must ",
    " needs ",
    " need ",
    " is ",
    " are ",
    " works ",
    " wants ",
    " prefer ",
    " prefers ",
)


def _clip(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _contains_claim(text: str) -> bool:
    padded = f" {text.lower()} "
    return any(marker in padded for marker in _CLAIM_MARKERS)


@dataclass
class ExtractionResult:
    event_id: str
    neurons: list[Neuron]
    synapses: list[Synapse]
    beliefs: list[Belief]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "status": self.status,
            "neurons": [n.model_dump(mode="json") for n in self.neurons],
            "synapses": [s.model_dump(mode="json") for s in self.synapses],
            "beliefs": [b.model_dump(mode="json") for b in self.beliefs],
        }


class EventExtractor:
    """Deterministic first-pass event extractor.

    This is intentionally conservative: it only converts explicit events into
    evidence-linked episode/topic neurons, light synapses, and inference-level
    beliefs. It never promotes extracted claims to facts.
    """

    def __init__(self, store: RealBrainStore):
        self.store = store

    def run(self, *, limit: int = 20, dry_run: bool = False) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        events = self.store.list_events(status="new", limit=limit)
        results = [self.extract_event(event, dry_run=dry_run).as_dict() for event in events]
        return {"processed": len(results), "dry_run": dry_run, "results": results}

    def extract_event(self, event: BrainEvent, *, dry_run: bool = False) -> ExtractionResult:
        content = event.content.strip()
        if not content:
            if not dry_run:
                self.store.mark_event_status(event.id, "rejected")
            return ExtractionResult(event.id, [], [], [], "rejected")

        evidence_refs = list(dict.fromkeys([event.id, *event.evidence_refs]))
        episode = Neuron(
            type="episode",
            title=_clip(content, 90),
            summary=_clip(content, 300),
            confidence=0.7,
            importance=int(event.metadata.get("importance", 5)) if isinstance(event.metadata.get("importance"), int) else 5,
            created_from_event=event.id,
            evidence_refs=evidence_refs,
            tags=[event.event_type, event.source],
        )
        neurons = [episode]

        topics: list[Neuron] = []
        lowered = content.lower()
        for needle, (kind, title) in _KEYWORD_NEURONS.items():
            if needle in lowered:
                topics.append(
                    self._topic_neuron(
                        title=title,
                        kind=kind,
                        event_id=event.id,
                        evidence_refs=evidence_refs,
                    )
                )
        # stable de-duplication by lower-case title
        seen: set[str] = set()
        topics = [n for n in topics if not (n.title.lower() in seen or seen.add(n.title.lower()))]
        neurons.extend(topics)

        synapses = [
            Synapse(
                source_neuron_id=episode.id,
                target_neuron_id=topic.id,
                relation_type="observed_with",
                weight=0.55,
                confidence=0.65,
                evidence_refs=evidence_refs,
            )
            for topic in topics
        ]

        beliefs: list[Belief] = []
        first_sentence = re.split(r"(?<=[.!?])\s+", content)[0]
        if _contains_claim(first_sentence):
            beliefs.append(
                Belief(
                    claim_text=_clip(first_sentence, 240),
                    status="inference",
                    confidence=0.45,
                    source_quality=0.65,
                    evidence_refs=evidence_refs,
                    owner_domain=str(event.metadata.get("domain")) if event.metadata.get("domain") else None,
                    review_after=datetime.now(UTC) + timedelta(days=14),
                    created_from_event=event.id,
                )
            )

        if not dry_run:
            saved_neurons = [self.store.add_neuron(n) for n in neurons]
            saved_synapses = [self.store.add_synapse(s) for s in synapses]
            saved_beliefs = [self.store.add_belief(b) for b in beliefs]
            self.store.mark_event_status(event.id, "extracted")
            return ExtractionResult(event.id, saved_neurons, saved_synapses, saved_beliefs, "extracted")
        return ExtractionResult(event.id, neurons, synapses, beliefs, "dry_run")

    def _topic_neuron(self, *, title: str, kind: str, event_id: str, evidence_refs: list[str]) -> Neuron:
        existing = [n for n in self.store.find_neurons(query=title, type=kind, limit=10) if n.title.lower() == title.lower()]
        if existing:
            neuron = existing[0]
            neuron.evidence_refs = list(dict.fromkeys([*neuron.evidence_refs, *evidence_refs]))
            neuron.importance = min(10, max(neuron.importance, 6))
            return neuron
        return Neuron(
            type=kind,  # type: ignore[arg-type]
            title=title,
            summary=f"Auto-extracted topic from event {event_id}.",
            confidence=0.6,
            importance=6,
            created_from_event=event_id,
            evidence_refs=evidence_refs,
            tags=["auto-extracted"],
        )


class SynapseHygiene:
    """Reinforce recent evidence-backed edges and decay stale edges."""

    def __init__(self, store: RealBrainStore):
        self.store = store

    def run(self, *, stale_after_days: int = 30, max_decay: float = 0.05, dry_run: bool = False, limit: int = 100) -> dict[str, Any]:
        stale_after_days = max(1, min(int(stale_after_days), 3650))
        max_decay = max(0.0, min(float(max_decay), 0.25))
        limit = max(1, min(int(limit), 500))
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=stale_after_days)
        reviewed = []
        changed = []
        for synapse in self.store.list_synapses(limit=limit):
            anchor = synapse.last_reinforced_at or synapse.created_at
            action = "kept"
            old_weight = synapse.weight
            if anchor < cutoff and synapse.status == "active":
                decay = min(max_decay, synapse.decay_rate * max(1, (now - anchor).days // stale_after_days))
                synapse.weight = max(0.0, round(synapse.weight - decay, 4))
                if synapse.weight < 0.2:
                    synapse.status = "weak"
                action = "decayed"
                if not dry_run:
                    self.store.add_synapse(synapse)
                changed.append({"id": synapse.id, "old_weight": old_weight, "new_weight": synapse.weight, "status": synapse.status})
            reviewed.append({"id": synapse.id, "action": action, "weight": synapse.weight, "status": synapse.status})
        return {"reviewed": len(reviewed), "changed": changed, "dry_run": dry_run, "stale_after_days": stale_after_days}


class ContradictionReview:
    """Surface explicit contradictions/disputed graph edges for review."""

    def __init__(self, store: RealBrainStore, obsidian: ObsidianAdapter):
        self.store = store
        self.obsidian = obsidian

    def run(self, *, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        synapses = [s for s in self.store.list_synapses(limit=limit) if s.relation_type == "contradicts" or s.status == "disputed"]
        beliefs = [b for b in self.store.list_beliefs(limit=limit) if b.contradictions or b.status in {"stale", "disproven"}]
        path = self._write_report(synapses, beliefs)
        event = BrainEvent(
            event_type="contradiction",
            source="real_brain.contradiction_review",
            content=f"Reviewed RealBrain contradictions: {len(synapses)} synapse(s), {len(beliefs)} belief(s) surfaced.",
            metadata={"synapse_count": len(synapses), "belief_count": len(beliefs), "path": path},
            sensitivity="personal",
            evidence_refs=[path],
            processed_status="archived",
        )
        self.store.record_event(event)
        return {
            "path": path,
            "event_id": event.id,
            "contradictory_synapses": [s.model_dump(mode="json") for s in synapses],
            "beliefs_to_review": [b.model_dump(mode="json") for b in beliefs],
        }

    def _write_report(self, synapses: list[Synapse], beliefs: list[Belief]) -> str:
        day = datetime.now(UTC).date().isoformat()
        path = f"brain/reviews/contradictions/{day}.md"
        lines = [
            f"# RealBrain Contradiction Review — {day}",
            "",
            "- authority: review queue only; no fact promotion or external action",
            f"- contradictory_synapses: {len(synapses)}",
            f"- beliefs_to_review: {len(beliefs)}",
            "",
            "## Synapses",
        ]
        if not synapses:
            lines.append("- none surfaced")
        for s in synapses:
            lines.append(f"- `{s.id}` {s.source_neuron_id} --{s.relation_type}/{s.status}/{s.weight:.2f}--> {s.target_neuron_id}")
        lines.extend(["", "## Beliefs"])
        if not beliefs:
            lines.append("- none surfaced")
        for b in beliefs:
            lines.append(f"- `{b.id}` [{b.status}/{b.confidence:.2f}] {b.claim_text}")
        written = self.obsidian.write_markdown(path, "\n".join(lines) + "\n", overwrite=True)
        return written["relative_path"]


class CuriosityQueue:
    """Generate a bounded, evidence-linked curiosity queue from weak spots."""

    def __init__(self, store: RealBrainStore, obsidian: ObsidianAdapter):
        self.store = store
        self.obsidian = obsidian

    def run(self, *, limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        items: list[dict[str, Any]] = []
        for event in self.store.list_events(status="new", limit=limit):
            items.append(
                {
                    "question": f"Should event `{event.id}` be extracted into durable memory?",
                    "why": "new event has not passed extraction/consolidation",
                    "evidence_refs": [event.id, *event.evidence_refs],
                    "priority": 0.6,
                }
            )
        for belief in self.store.list_beliefs(limit=limit):
            if belief.confidence < 0.5 or belief.status in {"hypothesis", "stale"}:
                items.append(
                    {
                        "question": f"What evidence would validate or retire belief `{belief.id}`?",
                        "why": f"belief status={belief.status}, confidence={belief.confidence:.2f}",
                        "evidence_refs": belief.evidence_refs,
                        "priority": 0.7 if belief.status == "stale" else 0.5,
                    }
                )
        for neuron in self.store.find_neurons(limit=limit):
            if not self.store.list_synapses(neuron_id=neuron.id, limit=1):
                items.append(
                    {
                        "question": f"What should `{neuron.title}` connect to in the RealBrain graph?",
                        "why": "active neuron has no synapses yet",
                        "evidence_refs": neuron.evidence_refs,
                        "priority": min(0.9, 0.2 + neuron.importance / 10),
                    }
                )
        items = sorted(items, key=lambda item: item["priority"], reverse=True)[:limit]
        path = self._write_queue(items)
        event = BrainEvent(
            event_type="curiosity",
            source="real_brain.curiosity_queue",
            content=f"Generated {len(items)} RealBrain curiosity item(s).",
            metadata={"path": path, "count": len(items)},
            sensitivity="personal",
            evidence_refs=[path],
            processed_status="archived",
        )
        self.store.record_event(event)
        return {"path": path, "event_id": event.id, "items": items}

    def _write_queue(self, items: list[dict[str, Any]]) -> str:
        lines = [
            "# RealBrain Curiosity Queue — Current",
            "",
            f"- updated: {datetime.now(UTC).isoformat()}",
            "- authority: questions only; no autonomous external action",
            "",
            "## Questions",
        ]
        if not items:
            lines.append("- No current curiosity items.")
        for item in items:
            refs = ", ".join(item.get("evidence_refs") or []) or "none"
            lines.append(f"- [{item['priority']:.2f}] {item['question']}")
            lines.append(f"  - why: {item['why']}")
            lines.append(f"  - evidence_refs: {refs}")
        written = self.obsidian.write_markdown("brain/curiosity/current.md", "\n".join(lines) + "\n", overwrite=True)
        return written["relative_path"]


class NightlyConsolidator:
    """Separate NREM-style nightly consolidation path.

    This bundles extraction, synapse hygiene, contradiction review, curiosity, and
    an NREM dream report into one bounded pass. It remains suggestion-only.
    """

    def __init__(self, store: RealBrainStore, obsidian: ObsidianAdapter):
        self.store = store
        self.obsidian = obsidian

    def run(self, *, budget: int = 20, focus_area: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        budget = max(1, min(int(budget), 100))
        extraction = EventExtractor(self.store).run(limit=budget, dry_run=dry_run)
        hygiene = SynapseHygiene(self.store).run(limit=budget * 5, dry_run=dry_run)
        contradiction = ContradictionReview(self.store, self.obsidian).run(limit=budget) if not dry_run else {"dry_run": True}
        curiosity = CuriosityQueue(self.store, self.obsidian).run(limit=budget) if not dry_run else {"dry_run": True}
        dream = DreamEngine(self.store, self.obsidian).run(mode="nrem_consolidation", budget=min(budget, 20), focus_area=focus_area) if not dry_run else {"dry_run": True}
        summary_path = self._write_summary(extraction, hygiene, contradiction, curiosity, dream, dry_run=dry_run)
        event = None
        if not dry_run:
            event = self.store.record_event(
                BrainEvent(
                    event_type="dream",
                    source="real_brain.nightly_consolidator",
                    content="Nightly RealBrain consolidation completed: extraction, synapse hygiene, contradiction review, curiosity queue, NREM report.",
                    metadata={"summary_path": summary_path, "focus_area": focus_area, "budget": budget},
                    sensitivity="personal",
                    evidence_refs=[summary_path],
                    processed_status="archived",
                )
            )
        return {
            "summary_path": summary_path,
            "event_id": event.id if event else None,
            "dry_run": dry_run,
            "extraction": extraction,
            "synapse_hygiene": hygiene,
            "contradiction_review": contradiction,
            "curiosity_queue": curiosity,
            "nrem_dream": dream,
        }

    def _write_summary(self, extraction: dict, hygiene: dict, contradiction: dict, curiosity: dict, dream: dict, *, dry_run: bool) -> str:
        day = datetime.now(UTC).date().isoformat()
        path = f"brain/sleep-reports/nightly-consolidation-{day}.md"
        lines = [
            f"# RealBrain Nightly Consolidation — {day}",
            "",
            f"- completed_at: {datetime.now(UTC).isoformat()}",
            f"- dry_run: {dry_run}",
            "- authority: internal memory hygiene only; no external or high-impact actions",
            "",
            "## Results",
            f"- events_extracted: {extraction.get('processed', 0)}",
            f"- synapses_changed: {len(hygiene.get('changed', []))}",
            f"- contradiction_report: {contradiction.get('path', 'dry_run')}",
            f"- curiosity_queue: {curiosity.get('path', 'dry_run')}",
            f"- nrem_report: {dream.get('summary_path', 'dry_run')}",
        ]
        written = self.obsidian.write_markdown(path, "\n".join(lines) + "\n", overwrite=True)
        return written["relative_path"]
