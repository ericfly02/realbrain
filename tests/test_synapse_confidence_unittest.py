import tempfile
import unittest
from pathlib import Path

from realbrain.models import Neuron, Synapse
from realbrain.store import RealBrainStore


class ListSynapsesConfidenceTests(unittest.TestCase):
    def test_min_and_max_confidence_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RealBrainStore(Path(tmp) / "brain.sqlite")
            n1 = store.add_neuron(Neuron(type="concept", title="A"))
            n2 = store.add_neuron(Neuron(type="concept", title="B"))
            n3 = store.add_neuron(Neuron(type="concept", title="C"))
            sure = store.add_synapse(
                Synapse(source_neuron_id=n1.id, target_neuron_id=n2.id, relation_type="related_to", weight=0.4, confidence=0.9)
            )
            shaky = store.add_synapse(
                Synapse(source_neuron_id=n1.id, target_neuron_id=n3.id, relation_type="related_to", weight=0.8, confidence=0.3)
            )

            self.assertEqual([s.id for s in store.list_synapses(min_confidence=0.7)], [sure.id])
            self.assertEqual([s.id for s in store.list_synapses(max_confidence=0.4)], [shaky.id])
            band = store.list_synapses(min_confidence=0.2, max_confidence=0.5)
            self.assertEqual([s.id for s in band], [shaky.id])


if __name__ == "__main__":
    unittest.main()
