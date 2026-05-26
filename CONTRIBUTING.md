# Contributing to RealBrain

Thanks for helping make local, evidence-linked agent memory safer and more useful.

## Ground rules

- Keep RealBrain local-first by default.
- Do not add network calls, hosted services, telemetry, or cloud dependencies without an explicit design discussion.
- Do not store secrets, private vault contents, personal data dumps, or raw transcripts in examples or tests.
- Treat dream/consolidation output as hypothesis/suggestion only.
- Keep high-impact external actions owned by host runtimes, not RealBrain.
- Prefer small, reviewable pull requests.

## Good first contributions

- Improve docs and examples.
- Add adapter examples for MCP hosts, OpenClaw, CLI agents, or FastAPI bridges.
- Add safety tests for path confinement, sensitivity labels, and hypothesis/fact separation.
- Add benchmark fixtures for event extraction, search, or consolidation quality.
- Improve package metadata and PyPI readiness.
- Add demo screenshots/GIFs using fake data.

## Development setup

```bash
git clone https://github.com/ericfly02/realbrain.git
cd realbrain
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
python -m compileall realbrain realbrain_server examples tests
python examples/demo.py
```

Quickstart smoke check tip:
- run `python examples/demo.py` from a temp/throwaway folder so it writes `./demo_vault` there.
- demo uses fake local data only and should run without secrets or network access.

## Pull request checklist

Before opening a PR:

- [ ] Tests pass with `python -m unittest discover -s tests`.
- [ ] Compile check passes with `python -m compileall realbrain realbrain_server examples tests`.
- [ ] Demo runs with fake local data only.
- [ ] No generated SQLite DBs, vault data, `.env` files, tokens, credentials, or private paths are committed.
- [ ] New behavior is documented.
- [ ] Safety boundaries are preserved.

## Issue etiquette

Please include:

- What you expected.
- What happened.
- Minimal reproduction steps or a small fake-data example.
- Python version and OS.
- Whether you are using the core library, tool layer, OpenClaw, MCP, or another host.

Do not paste secrets, private notes, personal vault contents, API tokens, or production host config into issues.

## Design principles

RealBrain should make memory more trustworthy by being explicit, inspectable, and conservative:

- evidence refs over vibes;
- typed records over raw transcript slop;
- human-readable markdown as durable truth;
- SQLite as an operational index;
- hypotheses never promoted to facts automatically;
- host-owned authority for external actions.
