import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from realbrain.dream_engine import DreamEngine
from realbrain.global_workspace import GlobalWorkspace
from realbrain.models import BrainEvent, Neuron, Synapse
from realbrain.obsidian_adapter import ObsidianAdapter, ObsidianSafetyError
from realbrain.store import RealBrainStore
from realbrain_server.tools import RealBrainToolContext, activate, dream, extract_events, record_event, search_memory


class RealBrainSmokeTests(unittest.TestCase):
    def test_store_workspace_and_dream_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RealBrainStore(root / "realbrain.sqlite")
            obsidian = ObsidianAdapter(root / "vault")

            event = store.record_event(
                BrainEvent(
                    event_type="conversation",
                    source="unit-test",
                    content="RealBrain should preserve evidence and keep dreams as hypotheses.",
                    evidence_refs=["test://event/1"],
                )
            )
            a = store.add_neuron(Neuron(type="project", title="RealBrain", importance=9, created_from_event=event.id))
            b = store.add_neuron(Neuron(type="concept", title="Dream Engine", importance=8, created_from_event=event.id))
            store.add_synapse(Synapse(source_neuron_id=a.id, target_neuron_id=b.id, relation_type="depends_on", weight=0.7))

            workspace = GlobalWorkspace(store, obsidian).activate("RealBrain", depth=1, budget=5)
            self.assertTrue(workspace["activated_neurons"])
            self.assertIn("brain/global-workspace/current.md", workspace["workspace_path"])

            result = DreamEngine(store, obsidian).run(mode="rem_generation", budget=5, focus_area="RealBrain")
            run = result["dream_run"]
            self.assertEqual(run["accepted_changes"], [])
            self.assertIn("hypotheses_not_facts", run["safety_flags"])
            self.assertIn("not fact", obsidian.read_markdown(run["summary_path"])["content"].lower())

    def test_obsidian_adapter_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = ObsidianAdapter(tmp)
            adapter.write_markdown("notes/example.md", "# Example", evidence_refs=["test://source"])
            self.assertIn("Example", adapter.read_markdown("notes/example.md")["content"])
            with self.assertRaises(ObsidianSafetyError):
                adapter.write_markdown("../escape.md", "nope")

    def test_tool_layer_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vault"
            ctx = RealBrainToolContext(brain_root=root, db_path=root / "ops" / "brain" / "realbrain.sqlite")

            event_result = record_event(
                {
                    "event_type": "conversation",
                    "source": "unit-test",
                    "content": "RealBrain records useful events for later retrieval.",
                    "evidence_refs": ["test://tool/1"],
                },
                ctx=ctx,
            )
            self.assertTrue(event_result["success"])

            extraction = extract_events(limit=5, ctx=ctx)
            self.assertTrue(extraction["success"])
            self.assertGreaterEqual(extraction["result"]["processed"], 1)

            search = search_memory("useful events retrieval", ctx=ctx)
            self.assertTrue(search["success"])

            workspace = activate("RealBrain", ctx=ctx)
            self.assertTrue(workspace["success"])

            dream_result = dream(mode="rem_generation", budget=3, focus_area="RealBrain", ctx=ctx)
            self.assertTrue(dream_result["success"])
            self.assertTrue(any("hypothesis" in warning.lower() for warning in dream_result["warnings"]))

    def test_demo_runs_in_temp_dir(self):
        demo = Path(__file__).resolve().parents[1] / "examples" / "demo.py"
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, str(demo)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            db = Path(tmp) / "demo_vault" / "ops" / "brain" / "realbrain.sqlite"
            self.assertTrue(db.exists())


if __name__ == "__main__":
    unittest.main()
