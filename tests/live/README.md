# Live smoke scripts

These files are direct runnable local experiments, not pytest tests. Running
the script is the explicit opt-in to contact live APIs or a local Ollama
server.

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
