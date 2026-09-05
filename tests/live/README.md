# Live smoke scripts

These files are direct runnable local experiments, not pytest tests. Running
the script is the explicit opt-in to contact live APIs or a local Ollama
server.

## AgenticMapper smoke scripts

Edit the constants near the top of each script, keep credentials in environment
variables, and run:

```bash
uv run python tests/live/agentic_openai_smoke.py
uv run python tests/live/loinc_smoke.py
uv run python tests/live/agentic_ollama_smoke.py
uv run python tests/live/agentic_experiment_smoke.py
```

Required credentials:

```bash
export OPENAI_API_KEY="..."       # OpenAI scripts
export LOINC_USERNAME="..."       # LOINC cases
export LOINC_PASSWORD="..."
```

Local Ollama scripts normally require no secret, but Ollama must be running and
the selected model must support tool calling.

## PlannedPipeline smoke scripts

These are direct runnable manual smoke scripts for the new planned pipeline,
not pytest tests. They intentionally use:

```python
OntologyMapper(use_planned_pipeline=True, retrieval_mode=...)
```

`AgenticMapper` is intentionally not used.

Run:

```bash
uv run python tests/live/planned_public_smoke.py
uv run python tests/live/planned_local_smoke.py
uv run python tests/live/planned_disabled_smoke.py
```

For the planned smoke scripts, OpenAI runs require:

```bash
export OPENAI_API_KEY="..."
```

Public planned LOINC retrieval can also use:

```bash
export LOINC_USERNAME="..."
export LOINC_PASSWORD="..."
```

All other settings are edited at the top of each planned smoke file:

- Provider switching: edit `PROVIDER = "openai"` or `PROVIDER = "ollama"`.
- Model switching: edit `OPENAI_MODEL` or `OLLAMA_MODEL`.
- Ollama URL: edit `OLLAMA_BASE_URL`.
- Source term, label, target ontology, clinical area, and retrieval mode: edit
  the corresponding constants.
- Local SapBERT URL: edit `SAPBERT_URL` in `planned_local_smoke.py`.

`planned_public_smoke.py` passes `LOINC_USERNAME` and `LOINC_PASSWORD` to
`SearchTools` for LOINC public retrieval. If those variables are missing, LOINC
retrieval may return no candidates and the result may be `UNKNOWN:UNMAPPED`.
