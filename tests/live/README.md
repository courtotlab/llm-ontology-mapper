# Live smoke scripts

These files are direct runnable local experiments, not pytest tests. Running
the script is the explicit opt-in to contact live APIs or a local Ollama
server.

The only supported mapping architecture is the seven-stage planned pipeline;
these scripts exercise it directly via:

```python
OntologyMapper(retrieval_mode=...)
```

## LOINC smoke script

```bash
uv run python tests/live/loinc_smoke.py
```

Required credentials:

```bash
export LOINC_USERNAME="..."
export LOINC_PASSWORD="..."
```

## PlannedPipeline smoke scripts

These are direct runnable manual smoke scripts for the planned pipeline, not
pytest tests. Run:

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
