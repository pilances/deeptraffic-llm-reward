# LLM communication

The separately implemented strict response parser and explicit fallback are
documented in [llm_action.md](llm_action.md). They are not yet wired to this
client or the simulation.

`llm_client.OllamaClient` sends prepared role/content messages to Ollama's
`POST /api/chat` endpoint. It returns the complete request and server response,
including raw text and server timing/token metrics. It performs no action
parsing, simulation calls, learning, fallback policy or retries. Communication
failures, empty/incomplete replies and output-budget exhaustion raise
`LLMCommunicationError`. A request has a configurable finite timeout.

```python
from deeptraffic_llm_reward.llm_client import OllamaClient
from deeptraffic_llm_reward.llm_state import prepare_llm_request

prepared = prepare_llm_request(observation, config)
exchange = OllamaClient().chat(prepared['messages'])
print(exchange['response']['message']['content'])
```

The existing native Ollama server on this Mac already has `qwen3:8b` installed.
No container or download is needed. Defaults: localhost:11434, temperature 0,
seed 42, thinking off, JSON output, 64 output tokens, context 8192, non-streaming.
Each call contains only explicitly supplied messages. The model is unloaded
after the reply (`keep_alive=0`) to release memory. These settings aid repeatability
but do not guarantee identical output across model/runtime/hardware versions.

The live smoke command sends exactly one prompt, using a fictional 52x13 grid
with a slower vehicle ahead. It never creates an environment or executes an
action. Model-list and version GET requests record metadata without inference.

```sh
.venv/bin/python scripts/test_llm_connection.py
# Repeating requires a new output directory:
.venv/bin/python scripts/test_llm_connection.py --out results/llm_connection_repeat
```

`--url` and `--model` select an Ollama-compatible endpoint and installed model.
Only Ollama transport is implemented in this step, not a vLLM client.
Existing output directories are refused. Generated artifacts:

- `prepared.json`: synthetic state, prompt messages and state/prompt hashes.
- `exchange.json`: exact request/response, latency, UTC timestamp, Ollama version,
  installed model digest/quantization, decoding settings and synthetic config.

Artifacts are reproducible diagnostic output, safe to remove and regenerate
in a new directory. A successful smoke test validates transport and receipt of
a completed textual reply; it does not measure driving quality or validate an
action for execution. Unit tests use mocked HTTP responses and perform no LLM
inference. No dependencies beyond the existing package and Python standard
library were added.

API reference: https://docs.ollama.com/api/chat

## Executed smoke test

The single live inference request completed successfully on 2026-11-14 using
`qwen3:8b` (Q4_K_M, model digest recorded in exchange.json). Wall time was
20.94165 seconds including model loading. Raw reply: `{"action":1}`.
No action was executed and no additional live prompt was sent. The 13 targeted
communication/state tests passed. Results are under `results/llm_connection_v1/`.
