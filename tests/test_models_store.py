from pathlib import Path
import tempfile

from pydantic import ValidationError

from realbrain.models import Belief, BrainEvent, DreamRun, Neuron, Synapse
from realbrain.store import RealBrainStore


def test_models_validate_bounds_and_self_edges():
    try:
        Neuron(type="concept", title="Bad confidence", confidence=1.2)
        raise AssertionError("expected validation error")
    except ValidationError:
        pass
    try:
        Synapse(source_neuron_id="n1", target_neuron_id="n1", relation_type="related_to")
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_store_persists_core_records():
    with tempfile.TemporaryDirectory() as tmp:
        store = RealBrainStore(Path(tmp) / "brain.sqlite")
        event = store.record_event(BrainEvent(event_type="conversation", source="unit-test", content="A user wants evidence-backed memory.", evidence_refs=["test://1"]))
        assert store.get_event(event.id).content == event.content
        n1 = store.add_neuron(Neuron(type="concept", title="RealBrain", created_from_event=event.id, evidence_refs=[event.id], importance=8))
        n2 = store.add_neuron(Neuron(type="project", title="Example Project", importance=9))
        assert store.get_neuron(n1.id).title == "RealBrain"
        assert store.find_neurons(query="example project")[0].id == n2.id
        synapse = store.add_synapse(Synapse(source_neuron_id=n1.id, target_neuron_id=n2.id, relation_type="part_of", weight=0.4, confidence=0.7))
        updated = store.reinforce_synapse(synapse.id, delta=0.2, reason="useful retrieval")
        assert updated.weight == 0.6
        belief = store.add_belief(Belief(claim_text="Dream outputs are hypotheses, not facts.", status="fact", evidence_refs=[event.id]))
        assert store.get_belief(belief.id).status == "fact"
        dream = store.add_dream_run(DreamRun(mode="rem_generation", generated_hypotheses=["Try a small workbench."]))
        assert store.get_dream_run(dream.id).mode == "rem_generation"


def test_list_synapses_filters_by_confidence():
    with tempfile.TemporaryDirectory() as tmp:
        store = RealBrainStore(Path(tmp) / "brain.sqlite")
        n1 = store.add_neuron(Neuron(type="concept", title="A"))
        n2 = store.add_neuron(Neuron(type="concept", title="B"))
        n3 = store.add_neuron(Neuron(type="concept", title="C"))
        sure = store.add_synapse(Synapse(source_neuron_id=n1.id, target_neuron_id=n2.id, relation_type="related_to", weight=0.4, confidence=0.9))
        shaky = store.add_synapse(Synapse(source_neuron_id=n1.id, target_neuron_id=n3.id, relation_type="related_to", weight=0.8, confidence=0.3))
        assert [s.id for s in store.list_synapses(min_confidence=0.7)] == [sure.id]
        assert [s.id for s in store.list_synapses(max_confidence=0.4)] == [shaky.id]
        # min and max bracket a confidence band, dropping both edges here
        assert store.list_synapses(min_confidence=0.4, max_confidence=0.8) == []
