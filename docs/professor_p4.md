# Professor P4 reconstruction

The visible system instruction in IMG_8098.jpg is transcribed as `SYS_P4` in
`llm_professor.py`. The original `prompt_lab.py`, `state_tool.py`, and
`disagree.py` are not supplied: their complete implementations cannot be recovered
from a photograph. The renderer and runner here reconstruct the visible contract.

The current v2 evaluation uses eleven controlled cars, a raw 315-value observation per callback
(3 lanes each side, 40 patches ahead, 5 including the ego row behind), and the
same five action IDs. Only the teacher sees the natural-language summary.
Forward FAST is >=70 mph, CLOSE <=14 patches; rear FAST >=62 mph,
CLOSE <=5 patches. Five configured rear rows contain four strictly rearward rows.
Zero cells remain ambiguous rather than being assigned a made-up vehicle speed.
Lane number and available actions use fresh native engine telemetry for the
current callback owner. `legal_actions()` includes left/right only when centered,
on-road and permitted by the engine's target-region gate. The message reports
CHANGE LEGAL/BLOCKED and rejection reasons. This is an extension to the photo's
road-only mask, not an exact replication of its observation-only teacher.
The speed objective is updated to the controlled fleet average.

Responses contain exactly `reason` (short sentence) and `action` (lowercase named
action). The Ollama JSON schema restricts the action enum to the current legal
actions before generation. The parser validates the same set; unavailable choices
raise an error and are never silently replaced. `masked_argmax` accepts the same
legal action list for a student network. Native execution remains unchanged.
The existing frozen 676-input submission is not resized or retrained.

The runner defaults to qwen3:8b, 200 frames, temperature 0, exact-message caching of valid
responses, and seeds 2000100, 2000101, 2000102. Cache is process-local and fresh on
each launch. Temperature 0 aids repeatability but is not a cross-runtime identity
guarantee. Prompt, original responses, action IDs and safety outcomes are logged.
No live simulation or model download is triggered by installing this change.

```sh
.venv/bin/python -u scripts/evaluate_professor.py --model qwen3:8b --frames 200 --seeds 2000100 2000101 2000102 --out results/llm_compact/professor_p4_v2
```

The model must already be installed in the local Ollama server. The native mask
is checked at callback time; a past legal status is no guarantee for subsequent
propagated action applications. The earlier v1 results used one car and are not
a matched comparison for v2. The page
does not provide enough information to reproduce the professor's original score
or a full student-training/distillation procedure.

## Spark inference

The runner accepts `--base-url` and records it in the manifest. For the separate
Spark container `deeptraffic-qwen3-8b`, expose Ollama on Spark loopback port 11436
and use a tunnel from the Mac (keep this terminal open):

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:11436:127.0.0.1:11436 luis@100.118.208.49
```

In another Mac terminal, first check installation with
`curl -fsS http://127.0.0.1:11436/api/tags`. Once `qwen3:8b` is present:

```sh
.venv/bin/python -u scripts/evaluate_professor.py \
  --base-url http://127.0.0.1:11436 --model qwen3:8b \
  --frames 2000 --seeds 2000100 2000101 2000102 \
  --out results/llm_compact/professor_p4_spark_v1
```

The simulation and result files remain on the Mac. GPU inference runs on Spark.
Compare the model digest in `/api/tags` with the Mac's checkpoint environment;
the same weights and sampling settings can still produce different decisions
across GPU backends.
