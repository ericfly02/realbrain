from pathlib import Path
import tempfile
from datetime import datetime, timedelta

from realbrain.compat import UTC

from realbrain.cognitive_loop import CuriosityQueue, EventExtractor, NightlyConsolidator, SynapseHygiene
from realbrain.models import Belief, BrainEvent, Neuron, Synapse
from realbrain.obsidian_adapter import ObsidianAdapter
from realbrain.store import RealBrainStore


def test_event_extraction_creates_evidence_linked_inferences():
    with tempfile.TemporaryDirectory() as tmp:
        store = RealBrainStore(Path(tmp) / "brain.sqlite")
        event = store.record_event(BrainEvent(event_type="conversation", source="test", content="RealBrain should extract meaningful interactions into memory.", evidence_refs=["test://1"]))
        result = EventExtractor(store).run(limit=5)
        assert result["processed"] == 1
        assert store.get_event(event.id).processed_status == "extracted"
        assert store.find_neurons(query="RealBrain")
        beliefs = store.list_beliefs()
        assert beliefs[0].status == "inference"
        assert event.id in beliefs[0].evidence_refs
        assert store.list_synapses()


def test_synapse_hygiene_decays_stale_edges():
    with tempfile.TemporaryDirectory() as tmp:
        store = RealBrainStore(Path(tmp) / "brain.sqlite")
        a = store.add_neuron(Neuron(type="concept", title="Old A"))
        b = store.add_neuron(Neuron(type="concept", title="Old B"))
        synapse = store.add_synapse(Synapse(source_neuron_id=a.id, target_neuron_id=b.id, relation_type="related_to", weight=0.25, created_at=datetime.now(UTC) - timedelta(days=90), decay_rate=0.1))
        result = SynapseHygiene(store).run(stale_after_days=30, max_decay=0.1)
        assert len(result["changed"]) == 1
        updated = store.get_synapse(synapse.id)
        assert updated.weight < 0.25
        assert updated.status == "weak"


def test_curiosity_and_nightly_consolidation_write_safe_reports():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = RealBrainStore(root / "brain.sqlite")
        obsidian = ObsidianAdapter(root / "vault")
        store.record_event(BrainEvent(event_type="reflection", source="test", content="RealBrain needs curiosity."))
        store.add_belief(Belief(claim_text="A weak hypothesis needs evidence.", status="hypothesis", confidence=0.3))
        curiosity = CuriosityQueue(store, obsidian).run(limit=10)
        assert "brain/curiosity/current.md" in curiosity["path"]
        assert "authority: questions only" in obsidian.read_markdown(curiosity["path"])["content"]
        nightly = NightlyConsolidator(store, obsidian).run(budget=10, focus_area="RealBrain")
        summary = obsidian.read_markdown(nightly["summary_path"])["content"]
        assert "internal memory hygiene only" in summary
        assert "events_extracted" in summary
